"""Compare ordinary inference with dihedral test-time augmentation on validation."""
from __future__ import annotations

import argparse
import csv
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from sen12mscr.losses import cloud_mask, per_image_metrics
from sen12mscr.model import ResUNet
from sen12mscr.parquet_dataset import Sen12msCRParquet
from sen12mscr.dataset import read_manifest


METRIC_KEYS = (
    "PSNR_full", "PSNR_cloud", "RMSE_full", "RMSE_cloud",
    "SSIM_full", "SSIM_cloud", "SAM_full", "SAM_cloud",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", choices=("val",), default="val")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--amp", default="auto", choices=("auto", "bf16", "fp16", "off"))
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Optional validation cap for a pilot; 0 evaluates the whole split")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def transform(x: torch.Tensor, k: int, mirror: bool) -> torch.Tensor:
    if mirror:
        x = torch.flip(x, dims=(-1,))
    return torch.rot90(x, k, dims=(-2, -1)) if k else x


def inverse_transform(x: torch.Tensor, k: int, mirror: bool) -> torch.Tensor:
    if k:
        x = torch.rot90(x, -k, dims=(-2, -1))
    return torch.flip(x, dims=(-1,)) if mirror else x


def summarize(records: list[dict], group_key: str | None):
    groups = {"all": records}
    if group_key:
        for value in sorted({row[group_key] for row in records}):
            groups[str(value)] = [row for row in records if row[group_key] == value]

    summary = []
    for group, rows in groups.items():
        item = {"group": group, "samples": len(rows)}
        for key in METRIC_KEYS:
            for mode in ("baseline", "tta"):
                values = np.asarray([row[f"{mode}_{key}"] for row in rows], dtype=np.float64)
                finite = values[np.isfinite(values)]
                item[f"{mode}_{key}"] = float(finite.mean()) if finite.size else math.nan
            item[f"delta_{key}"] = item[f"tta_{key}"] - item[f"baseline_{key}"]
        summary.append(item)
    return summary


def main():
    args = parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or args.max_samples < 0:
        raise SystemExit("batch_size must be positive; workers and max_samples cannot be negative")
    if os.path.exists(args.out):
        raise SystemExit(f"Output path already exists; choose a new --out: {args.out}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but is not available")

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    amp = args.amp
    if device.type != "cuda" or amp == "off":
        amp = "off"
    elif amp == "auto":
        amp = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    elif amp == "bf16" and not torch.cuda.is_bf16_supported():
        amp = "fp16"
    amp_dtype = torch.bfloat16 if amp == "bf16" else torch.float16

    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    saved_args = checkpoint.get("args", {})
    use_sar = not saved_args.get("no_sar", False)
    model = ResUNet(
        base=saved_args.get("base", 48),
        use_sar=use_sar,
        attention_mode=saved_args.get("attention_mode", "self"),
    ).to(device).eval()
    weights = checkpoint.get("ema", checkpoint.get("net"))
    if weights is None:
        raise RuntimeError("Checkpoint does not contain EMA or network weights")
    model.load_state_dict(weights)

    manifest = read_manifest(args.manifest)
    items = manifest["splits"][args.split]
    if not items:
        raise RuntimeError(f"Manifest split '{args.split}' is empty")
    dataset = Sen12msCRParquet(
        args.data_root,
        items,
        use_sar=use_sar,
        s2_scale=saved_args.get("s2_scale", 10000.0),
        sar_min_db=saved_args.get("sar_min_db", -25.0),
        sar_max_db=saved_args.get("sar_max_db", 0.0),
        batch_size=args.batch_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )

    def predict(cloudy, sar):
        if device.type == "cuda" and amp != "off":
            with torch.autocast("cuda", dtype=amp_dtype):
                return model(cloudy, sar)
        return model(cloudy, sar)

    os.makedirs(args.out, exist_ok=False)
    csv_path = os.path.join(args.out, "per_image_val.csv")
    transforms = [(k, mirror) for mirror in (False, True) for k in range(4)]
    records = []
    baseline_seconds = 0.0
    tta_seconds = 0.0
    started = time.perf_counter()
    sample_count = 0

    print(f"Device: {device} | AMP: {amp} | SAR: {use_sar}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Validation row groups: {len(items)} | TTA views: {len(transforms)}")

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        fieldnames = ["image", "roi", "scene", "season", "patch"]
        fieldnames += [f"{mode}_{key}" for mode in ("baseline", "tta") for key in METRIC_KEYS]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        with torch.inference_mode():
            for batch_index, batch in enumerate(loader, start=1):
                cloudy = batch["cloudy"].to(device, non_blocking=True)
                clear = batch["clear"].to(device, non_blocking=True)
                sar = batch["s1"].to(device, non_blocking=True) if use_sar else None

                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                tick = time.perf_counter()
                baseline_prediction = predict(cloudy, sar).float()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                baseline_seconds += time.perf_counter() - tick

                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                tick = time.perf_counter()
                tta_prediction = None
                for k, mirror in transforms:
                    cloudy_view = transform(cloudy, k, mirror)
                    sar_view = transform(sar, k, mirror) if sar is not None else None
                    prediction = predict(cloudy_view, sar_view).float()
                    prediction = inverse_transform(prediction, k, mirror)
                    tta_prediction = prediction if tta_prediction is None else tta_prediction + prediction
                tta_prediction = tta_prediction / len(transforms)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                tta_seconds += time.perf_counter() - tick

                mask = cloud_mask(cloudy.float(), clear.float())
                baseline_metrics = per_image_metrics(baseline_prediction, clear.float(), mask)
                tta_metrics = per_image_metrics(tta_prediction, clear.float(), mask)
                batch_size = cloudy.shape[0]
                for index in range(batch_size):
                    row = {
                        "image": batch["name"][index],
                        "roi": batch["roi"][index],
                        "scene": batch["scene"][index],
                        "season": batch.get("season", batch["roi"])[index],
                        "patch": batch.get("patch", [""] * batch_size)[index],
                    }
                    for key in METRIC_KEYS:
                        row[f"baseline_{key}"] = float(baseline_metrics[key][index].cpu())
                        row[f"tta_{key}"] = float(tta_metrics[key][index].cpu())
                    records.append(row)
                    writer.writerow(row)
                sample_count += batch_size

                if batch_index % 50 == 0:
                    handle.flush()
                    print(f"Processed {sample_count:,} validation samples", flush=True)
                if args.max_samples and sample_count >= args.max_samples:
                    break

    elapsed = time.perf_counter() - started
    summary = summarize(records, "season")
    summary_path = os.path.join(args.out, "summary_val.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    timing_path = os.path.join(args.out, "timing.csv")
    with open(timing_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "samples", "views", "baseline_model_seconds", "tta_model_seconds",
            "baseline_ms_per_sample", "tta_ms_per_sample", "tta_over_baseline", "total_wall_seconds",
        ))
        writer.writeheader()
        writer.writerow({
            "samples": sample_count,
            "views": len(transforms),
            "baseline_model_seconds": baseline_seconds,
            "tta_model_seconds": tta_seconds,
            "baseline_ms_per_sample": 1000 * baseline_seconds / max(sample_count, 1),
            "tta_ms_per_sample": 1000 * tta_seconds / max(sample_count, 1),
            "tta_over_baseline": tta_seconds / max(baseline_seconds, 1e-9),
            "total_wall_seconds": elapsed,
        })

    print(f"Evaluated {sample_count:,} samples in {elapsed/60:.1f} min.")
    print(f"Per-image results: {csv_path}")
    print(f"Summary: {summary_path}")
    print(f"Timing: {timing_path}")
    for row in summary:
        print(
            f"[{row['group']}] n={row['samples']} "
            f"PSNR {row['baseline_PSNR_full']:.4f}->{row['tta_PSNR_full']:.4f} "
            f"({row['delta_PSNR_full']:+.4f} dB), "
            f"SSIM {row['baseline_SSIM_full']:.5f}->{row['tta_SSIM_full']:.5f} "
            f"({row['delta_SSIM_full']:+.5f}), "
            f"SAM {row['baseline_SAM_full']:.4f}->{row['tta_SAM_full']:.4f} "
            f"({row['delta_SAM_full']:+.4f} deg)"
        )
    print(f"Model-time ratio (TTA / baseline): {tta_seconds/max(baseline_seconds, 1e-9):.2f}x")
    print("Cloud-region scores use the project's RGB-difference pseudo-mask.")


if __name__ == "__main__":
    main()

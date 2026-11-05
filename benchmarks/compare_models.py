"""Compare two SEN12MS-CR checkpoints on the same manifest split.

Produces per-image metrics, grouped metric summaries, and four-panel RGB
comparisons: cloudy input | baseline | spectral-weighted | clear reference.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from sen12mscr.dataset import Sen12msCR, read_manifest
from sen12mscr.losses import LPIPSLoss, cloud_mask, per_image_metrics
from sen12mscr.model import ResUNet
from sen12mscr.parquet_dataset import Sen12msCRParquet

RGB = [3, 2, 1]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", required=True)
    p.add_argument("--data_format", choices=["geotiff", "parquet"], default="geotiff")
    p.add_argument("--manifest", required=True)
    p.add_argument("--baseline_ckpt", required=True)
    p.add_argument("--spectral_ckpt", "--candidate_ckpt", dest="spectral_ckpt", required=True)
    p.add_argument("--out", default="runs/final_comparison")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--amp", choices=["auto", "bf16", "fp16", "off"], default="auto")
    p.add_argument("--save_visuals", type=int, default=12)
    p.add_argument("--baseline_label", default="Baseline")
    p.add_argument("--candidate_label", default="Candidate")
    p.add_argument("--lpips", action="store_true")
    p.add_argument("--lpips_net", choices=["alex", "vgg"], default="alex")
    return p.parse_args()


def load_model(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    use_sar = not args.get("no_sar", False)
    base = int(args.get("base", 48))
    model = ResUNet(base=base, use_sar=use_sar).to(device).eval()
    weights = ckpt.get("ema", ckpt.get("net"))
    if weights is None:
        raise RuntimeError(f"Checkpoint has neither 'ema' nor 'net' weights: {path}")
    model.load_state_dict(weights)
    return model, use_sar, base, int(ckpt.get("epoch", -1))


def rgb_image(t: torch.Tensor) -> Image.Image:
    # Match the demo's fixed exposure and sRGB transfer curve.
    linear = (t[RGB].detach().float().cpu() * 2.5).clamp(0, 1).permute(1, 2, 0).numpy()
    arr = np.where(linear <= 0.0031308, 12.92 * linear,
                   1.055 * np.power(linear, 1 / 2.4) - 0.055)
    return Image.fromarray((arr * 255.0).round().astype(np.uint8), mode="RGB")


def save_four_panel(cloudy, baseline, candidate, clear, path: Path, baseline_label, candidate_label):
    images = [rgb_image(x) for x in (cloudy, baseline, candidate, clear)]
    save_four_panel_images(images, path, baseline_label, candidate_label)


def save_four_panel_images(images, path: Path, baseline_label, candidate_label):
    w, h = images[0].size
    header = 30
    canvas = Image.new("RGB", (w * 4, h + header), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    labels = ["Cloudy input", baseline_label, candidate_label, "Clear reference"]
    for i, (img, label) in enumerate(zip(images, labels)):
        canvas.paste(img, (i * w, header))
        draw.text((i * w + 8, 8), label, fill="black", font=font)
    canvas.save(path)


def finite_mean(values):
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def main():
    a = parse_args()
    if a.batch_size < 1 or a.num_workers < 0 or a.save_visuals < 0:
        raise ValueError("batch_size must be positive; num_workers/save_visuals cannot be negative")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = a.amp
    if device.type != "cuda" or amp == "off":
        amp = "off"
    elif amp == "auto":
        amp = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    elif amp == "bf16" and not torch.cuda.is_bf16_supported():
        amp = "fp16"
    amp_dtype = torch.bfloat16 if amp == "bf16" else torch.float16

    baseline, sar_a, base_a, epoch_a = load_model(a.baseline_ckpt, device)
    candidate, sar_b, base_b, epoch_b = load_model(a.spectral_ckpt, device)
    if (sar_a, base_a) != (sar_b, base_b):
        raise RuntimeError(
            f"Checkpoint architectures differ: baseline SAR/base={sar_a}/{base_a}, "
            f"spectral SAR/base={sar_b}/{base_b}"
        )
    use_sar = sar_a

    manifest = read_manifest(a.manifest)
    items = manifest["splits"][a.split]
    if not items:
        raise RuntimeError(f"Manifest split '{a.split}' is empty")
    if a.data_format == "parquet":
        dataset = Sen12msCRParquet(
            a.data_root, items, use_sar=use_sar, batch_size=a.batch_size,
        )
        kwargs = dict(
            batch_size=None,
            num_workers=a.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=False,
        )
    else:
        dataset = Sen12msCR(a.data_root, items, use_sar=use_sar)
        kwargs = dict(
            batch_size=a.batch_size,
            shuffle=False,
            num_workers=a.num_workers,
            pin_memory=device.type == "cuda",
        )
        if a.num_workers > 0:
            kwargs["persistent_workers"] = True
    if a.num_workers > 0:
        kwargs["prefetch_factor"] = 2
    loader = DataLoader(dataset, **kwargs)
    total_samples = dataset.sample_count if a.data_format == "parquet" else len(dataset)

    lp = LPIPSLoss(a.lpips_net, spatial=True).to(device) if a.lpips else None
    out_dir = Path(a.out)
    visual_dir = out_dir / "visualizations"
    visual_dir.mkdir(parents=True, exist_ok=True)
    records = []
    visuals_saved = 0
    visuals_by_season: dict[str, int] = {}
    parquet_season_quota = max(1, math.ceil(a.save_visuals / 4))
    # Cache only one preview per season/scene/cloud-coverage bin. This avoids
    # showing neighboring patches from the first scene in each season.
    parquet_visual_candidates = {}

    print(f"Device={device} AMP={amp} SAR={use_sar} | format={a.data_format} | split={a.split} samples={total_samples:,}")
    print(f"Baseline checkpoint epoch={epoch_a}; candidate checkpoint epoch={epoch_b}")

    with torch.inference_mode():
        for batch in loader:
            cloudy = batch["cloudy"].to(device, non_blocking=True)
            clear = batch["clear"].to(device, non_blocking=True)
            sar = batch["s1"].to(device, non_blocking=True) if use_sar else None

            if device.type == "cuda" and amp != "off":
                with torch.autocast("cuda", dtype=amp_dtype):
                    pred_base = baseline(cloudy, sar)
                    pred_candidate = candidate(cloudy, sar)
            else:
                pred_base = baseline(cloudy, sar)
                pred_candidate = candidate(cloudy, sar)

            cloudy_f = cloudy.float().clamp(0, 1)
            clear_f = clear.float().clamp(0, 1)
            pred_base = pred_base.float().clamp(0, 1)
            pred_candidate = pred_candidate.float().clamp(0, 1)
            mask = cloud_mask(cloudy_f, clear_f)

            base_metrics = per_image_metrics(pred_base, clear_f, mask, lp)
            candidate_metrics = per_image_metrics(pred_candidate, clear_f, mask, lp)

            for i, name in enumerate(batch["name"]):
                row = {
                    "image": name,
                    "roi": batch["roi"][i],
                    "scene": batch["scene"][i],
                    "pseudo_cloud_coverage": float(mask[i].mean().cpu()),
                }
                for key, values in base_metrics.items():
                    row[f"baseline_{key}"] = float(values[i].cpu())
                for key, values in candidate_metrics.items():
                    row[f"candidate_{key}"] = float(values[i].cpu())
                records.append(row)

                season = batch.get("season", [""] * len(batch["name"]))[i]
                if a.data_format == "parquet":
                    if a.save_visuals > 0:
                        coverage_bin = min(5, int(row["pseudo_cloud_coverage"] * 6))
                        key = (str(season), str(row["scene"]), coverage_bin)
                        if key not in parquet_visual_candidates:
                            parquet_visual_candidates[key] = (
                                row["image"],
                                [rgb_image(x) for x in (
                                    cloudy_f[i], pred_base[i], pred_candidate[i], clear_f[i]
                                )],
                            )
                    save_this = False
                else:
                    save_this = visuals_saved < a.save_visuals
                if save_this:
                    safe_name = Path(name).stem
                    save_four_panel(
                        cloudy_f[i], pred_base[i], pred_candidate[i], clear_f[i],
                        visual_dir / f"{visuals_saved + 1:02d}_{safe_name}.png",
                        a.baseline_label,
                        a.candidate_label,
                    )
                    visuals_saved += 1
                    visuals_by_season[season] = visuals_by_season.get(season, 0) + 1

            done = len(records)
            print(f"\rProcessed {done:,}/{total_samples:,}", end="", flush=True)

    print()
    if a.data_format == "parquet" and a.save_visuals > 0:
        season_order = ("spring", "summer", "fall", "winter")
        for season in season_order:
            keys = [key for key in parquet_visual_candidates if key[0] == season]
            selected = []
            # Round-robin coverage bins and scenes, giving scene variety first.
            scenes = sorted({key[1] for key in keys})
            for coverage_bin in range(6):
                for scene in scenes:
                    key = (season, scene, coverage_bin)
                    if key in parquet_visual_candidates and key not in selected:
                        selected.append(key)
                        if len(selected) >= parquet_season_quota:
                            break
                if len(selected) >= parquet_season_quota:
                    break
            # Fill any gaps with remaining available scene/bin examples.
            for key in sorted(keys, key=lambda x: (x[2], x[1])):
                if len(selected) >= parquet_season_quota:
                    break
                if key not in selected:
                    selected.append(key)
            for key in selected:
                name, images = parquet_visual_candidates[key]
                safe_name = Path(name).stem
                save_four_panel_images(
                    images,
                    visual_dir / f"{visuals_saved + 1:02d}_{safe_name}_cloudbin{key[2]}.png",
                    a.baseline_label,
                    a.candidate_label,
                )
                visuals_saved += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    per_image_path = out_dir / f"per_image_{a.split}.csv"
    fields = list(records[0].keys())
    with per_image_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    metric_names = [k for k in records[0] if k.startswith("baseline_")]
    summary_rows = []
    buckets = {
        "all": lambda r: True,
        "coverage_1_10_percent": lambda r: 0.01 <= r["pseudo_cloud_coverage"] < 0.10,
        "coverage_10_50_percent": lambda r: 0.10 <= r["pseudo_cloud_coverage"] < 0.50,
        "coverage_50_100_percent": lambda r: r["pseudo_cloud_coverage"] >= 0.50,
    }
    for bucket, selector in buckets.items():
        subset = [r for r in records if selector(r)]
        if not subset:
            continue
        for prefix in ("baseline", "candidate"):
            row = {"group": bucket, "model": prefix, "samples": len(subset)}
            for key in metric_names:
                metric = key.removeprefix("baseline_")
                row[metric] = finite_mean([r.get(f"{prefix}_{metric}", float("nan")) for r in subset])
            summary_rows.append(row)

    summary_path = out_dir / f"summary_{a.split}.csv"
    summary_fields = list(summary_rows[0].keys())
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Per-image metrics: {per_image_path}")
    print(f"Grouped summary:   {summary_path}")
    print(f"Visual comparisons: {visual_dir} ({visuals_saved} images)")
    if a.data_format == "parquet" and a.save_visuals > 0:
        print("Parquet previews are selected across scenes and pseudo-cloud-coverage bins.")
    print("Cloud-region metrics use the RGB-difference pseudo-mask, not ground-truth cloud labels.")


if __name__ == "__main__":
    main()

"""Evaluate a SEN12MS-CR checkpoint on the held-out manifest split."""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
from .dataset import Sen12msCR, read_manifest
from .losses import LPIPSLoss, cloud_mask, per_image_metrics
from .model import ResUNet


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--amp", default="auto", choices=["auto", "bf16", "fp16", "off"])
    p.add_argument("--s2_scale", type=float, default=10000.0)
    p.add_argument("--sar_min_db", type=float, default=-25.0)
    p.add_argument("--sar_max_db", type=float, default=0.0)
    p.add_argument("--min_cov", type=float, default=0.01)
    p.add_argument("--lpips", action="store_true")
    p.add_argument("--lpips_net", default="alex", choices=["vgg", "alex"])
    p.add_argument("--out", default="eval")
    p.add_argument("--save_visuals", type=int, default=12)
    return p.parse_args()


def main():
    def save_comparison(cloudy, prediction, clear, path):
        def to_rgb(tensor):
            # Use Sentinel-2 RGB bands: B4, B3, B2 (indices 3, 2, 1).
            image = tensor[[3, 2, 1]].detach().float().cpu()
            image = image.clamp(0, 1).permute(1, 2, 0).numpy()
            return (image * 255).round().astype(np.uint8)

        cloudy_img = to_rgb(cloudy)
        prediction_img = to_rgb(prediction)
        clear_img = to_rgb(clear)

        # Place all three images side by side.
        comparison = np.concatenate(
            [cloudy_img, prediction_img, clear_img], axis=1
        )

        Image.fromarray(comparison).save(path)
    a = parse_args()
    if a.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else ("cpu" if a.device == "auto" else a.device))
    amp = a.amp
    if device.type != "cuda" or amp == "off":
        amp = "off"
    elif amp == "auto":
        amp = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    elif amp == "bf16" and not torch.cuda.is_bf16_supported():
        amp = "fp16"
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16

    checkpoint = torch.load(a.ckpt, map_location=device, weights_only=False)
    saved_args = checkpoint.get("args", {})
    use_sar = not saved_args.get("no_sar", False)
    model = ResUNet(base=saved_args.get("base", 48), use_sar=use_sar).to(device).eval()
    weights = checkpoint.get("ema", checkpoint.get("net"))
    if weights is None:
        raise RuntimeError("Checkpoint must contain either 'ema' or 'net' weights")
    model.load_state_dict(weights)

    manifest = read_manifest(a.manifest)
    items = manifest["splits"][a.split]
    if not items:
        raise RuntimeError(f"Manifest split '{a.split}' is empty")
    dataset = Sen12msCR(
        a.data_root, items, use_sar=use_sar, s2_scale=a.s2_scale,
        sar_min_db=a.sar_min_db, sar_max_db=a.sar_max_db,
    )
    loader = DataLoader(
        dataset, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers,
        pin_memory=device.type == "cuda", persistent_workers=a.num_workers > 0,
    )
    lp = LPIPSLoss(a.lpips_net, spatial=True).to(device) if a.lpips else None

    records = []
    visual_dir = os.path.join(a.out, "visualizations")
    os.makedirs(visual_dir, exist_ok=True)

    saved_visuals = 0
    with torch.no_grad():

        for batch in loader:

            cloudy = batch["cloudy"].to(device, non_blocking=True)
            clear = batch["clear"].to(device, non_blocking=True)
            sar = batch["s1"].to(device, non_blocking=True) if use_sar else None
            if device.type == "cuda" and amp != "off":
                with torch.autocast("cuda", dtype=dtype):
                    prediction = model(cloudy, sar)
            else:
                prediction = model(cloudy, sar)
            mask = cloud_mask(cloudy, clear)

            baseline_metrics = per_image_metrics(cloudy.float(), clear, mask, lp)
            model_metrics = per_image_metrics(prediction.float(), clear, mask, lp)
            for idx, name in enumerate(batch["name"]):
                row = {
                    "image": name,
                    "roi": batch["roi"][idx],
                    "scene": batch["scene"][idx],
                }
                row.update({
    f"baseline_{key}": float(value[idx].cpu())
    for key, value in baseline_metrics.items()
})
                row.update({
    f"model_{key}": float(value[idx].cpu())
    for key, value in model_metrics.items()
})
                records.append(row)
                if saved_visuals < a.save_visuals:
                    save_comparison(
                        cloudy[idx],
                        prediction[idx],
                        clear[idx],
                        os.path.join(visual_dir, f"{name}.png"),
                    )
                    saved_visuals += 1

    os.makedirs(a.out, exist_ok=True)
    csv_path = os.path.join(a.out, f"per_image_{a.split}.csv")
    metric_keys = [k for k in records[0] if k not in ("image", "roi", "scene")]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "roi", "scene"] + metric_keys)
        writer.writeheader()
        writer.writerows(records)

    print(f"Device: {device} | AMP: {amp} | SAR: {use_sar}")
    print(f"Evaluated {len(records):,} samples from split '{a.split}'.")
    for bucket_name, selector in (
        ("all", lambda r: True),
        ("cloud coverage >= min_cov", lambda r: r["model_mask_cov"] >= a.min_cov),
        ("1-10%", lambda r: 0.01 <= r["model_mask_cov"] < 0.10),
        ("10-50%", lambda r: 0.10 <= r["model_mask_cov"] < 0.50),
        (">=50%", lambda r: r["model_mask_cov"] >= 0.50),
    ):
        selected = [r for r in records if selector(r)]
        if not selected:
            continue
        print(f"\n[{bucket_name}] n={len(selected)}")
        for key in metric_keys:
            values = np.asarray([r[key] for r in selected], dtype=np.float64)
            finite = values[np.isfinite(values)]
            if finite.size:
                print(f"  {key:14s} {finite.mean():.5f}")
    print(f"Per-image results: {csv_path}")
    print(f"Saved visualizations: {saved_visuals} ({visual_dir})")
    print("Cloud-region scores use a pseudo-mask derived from cloudy/clear RGB differences, not a supplied ground-truth cloud mask.")


if __name__ == "__main__":
    main()

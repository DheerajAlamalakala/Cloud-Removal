
"""Global and cloud-region per-band SEN12MS-CR evaluation."""
import argparse
import csv
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import Sen12msCR, read_manifest
from .losses import cloud_mask
from .model import ResUNet


BANDS = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B10", "B11", "B12"
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--amp", default="bf16",
                   choices=["bf16", "fp16", "off"])
    p.add_argument("--out", default="./runs/band_evaluation")
    a = p.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        a.ckpt, map_location=device, weights_only=False
    )

    saved_args = checkpoint.get("args", {})
    use_sar = not saved_args.get("no_sar", False)

    model = ResUNet(
        base=saved_args.get("base", 48),
        use_sar=use_sar
    ).to(device).eval()

    weights = checkpoint.get("ema", checkpoint.get("net"))
    if weights is None:
        raise RuntimeError("Checkpoint has no EMA or network weights")

    model.load_state_dict(weights)

    manifest = read_manifest(a.manifest)
    items = manifest["splits"]["test"]

    dataset = Sen12msCR(
        a.data_root,
        items,
        use_sar=use_sar,
        s2_scale=10000.0,
        sar_min_db=-25.0,
        sar_max_db=0.0,
    )

    loader = DataLoader(
        dataset,
        batch_size=a.batch_size,
        shuffle=False,
        num_workers=a.num_workers,
        pin_memory=device.type == "cuda",
    )

    # Global error sums.
    global_sums = {
        name: {
            "squared_error": np.zeros(13, dtype=np.float64),
            "absolute_error": np.zeros(13, dtype=np.float64),
            "signed_error": np.zeros(13, dtype=np.float64),
        }
        for name in ["cloudy", "model"]
    }

    # Cloud-region error sums.
    cloud_sums = {
        name: {
            "squared_error": np.zeros(13, dtype=np.float64),
            "absolute_error": np.zeros(13, dtype=np.float64),
            "signed_error": np.zeros(13, dtype=np.float64),
        }
        for name in ["cloudy", "model"]
    }
    coverage_bins = {
        "low_1_10": (0.01, 0.10),
        "medium_10_50": (0.10, 0.50),
        "high_50_100": (0.50, 1.01),
    }

    coverage_sums = {
        bin_name: {
            name: {
                "squared_error": np.zeros(13, dtype=np.float64),
                "absolute_error": np.zeros(13, dtype=np.float64),
                "signed_error": np.zeros(13, dtype=np.float64),
            }
            for name in ["cloudy", "model"]
        }
        for bin_name in coverage_bins
    }

    coverage_pixels = {name: 0 for name in coverage_bins}
    coverage_samples = {name: 0 for name in coverage_bins}

    global_pixels = 0
    cloud_pixels = 0
    total_samples = 0

    amp_enabled = device.type == "cuda" and a.amp != "off"

    if amp_enabled and a.amp == "bf16":
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16

    if amp_enabled and a.amp == "bf16" and not torch.cuda.is_bf16_supported():
        amp_dtype = torch.float16

    with torch.no_grad():
        for batch in loader:
            cloudy = batch["cloudy"].to(device, non_blocking=True)
            clear = batch["clear"].to(device, non_blocking=True)
            sar = batch["s1"].to(device, non_blocking=True) if use_sar else None

            if amp_enabled:
                with torch.autocast("cuda", dtype=amp_dtype):
                    prediction = model(cloudy, sar)
            else:
                prediction = model(cloudy, sar)

            cloudy = cloudy.float().clamp(0, 1)
            clear = clear.float().clamp(0, 1)
            prediction = prediction.float().clamp(0, 1)

            mask = cloud_mask(cloudy, clear).float()

            # Handle expected Bx1xHxW or BxHxW mask formats.
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

            if mask.shape[1] != 1:
                mask = mask[:, :1]

            mask = mask > 0.5
                        # Calculate pseudo-cloud coverage separately for each image.
            sample_coverage = mask.float().mean(dim=(1, 2, 3))

            for bin_name, limits in coverage_bins.items():
                if bin_name == "low_1_10":
                    selected = (
                        (sample_coverage >= 0.01) &
                        (sample_coverage < 0.10)
                    )
                elif bin_name == "medium_10_50":
                    selected = (
                        (sample_coverage >= 0.10) &
                        (sample_coverage < 0.50)
                    )
                else:
                    selected = sample_coverage >= 0.50

                if not selected.any():
                    continue

                selected_mask = mask[selected].double()
                coverage_samples[bin_name] += int(selected.sum().item())
                coverage_pixels[bin_name] += int(selected_mask.sum().item())

                for name, error in {
                    "cloudy": cloudy - clear,
                    "model": prediction - clear,
                }.items():
                    selected_error = error[selected].double()

                    coverage_sums[bin_name][name]["squared_error"] += (
                        (selected_error.square() * selected_mask)
                        .sum(dim=(0, 2, 3)).cpu().numpy()
                    )

                    coverage_sums[bin_name][name]["absolute_error"] += (
                        (selected_error.abs() * selected_mask)
                        .sum(dim=(0, 2, 3)).cpu().numpy()
                    )

                    coverage_sums[bin_name][name]["signed_error"] += (
                        (selected_error * selected_mask)
                        .sum(dim=(0, 2, 3)).cpu().numpy()
                    )

            batch_pixels = clear.shape[0] * clear.shape[2] * clear.shape[3]
            batch_cloud_pixels = int(mask.sum().item())

            errors = {
                "cloudy": cloudy - clear,
                "model": prediction - clear,
            }

            for name, error in errors.items():
                global_sums[name]["squared_error"] += (
                    error.double().square().sum(dim=(0, 2, 3)).cpu().numpy()
                )
                global_sums[name]["absolute_error"] += (
                    error.double().abs().sum(dim=(0, 2, 3)).cpu().numpy()
                )
                global_sums[name]["signed_error"] += (
                    error.double().sum(dim=(0, 2, 3)).cpu().numpy()
                )

                error_double = error.double()
                mask_double = mask.double()

                cloud_sums[name]["squared_error"] += (
                    (error_double.square() * mask_double)
                    .sum(dim=(0, 2, 3))
                    .cpu().numpy()
                )

                cloud_sums[name]["absolute_error"] += (
                    (error_double.abs() * mask_double)
                    .sum(dim=(0, 2, 3))
                    .cpu().numpy()
                )

                cloud_sums[name]["signed_error"] += (
                    (error_double * mask_double)
                    .sum(dim=(0, 2, 3))
                    .cpu().numpy()
                )

            global_pixels += batch_pixels
            cloud_pixels += batch_cloud_pixels
            total_samples += clear.shape[0]

            print(
                f"\rProcessed {total_samples:,}/{len(dataset):,}",
                end="",
                flush=True
            )

    print()

    if global_pixels == 0:
        raise RuntimeError("No test pixels evaluated")

    if cloud_pixels == 0:
        raise RuntimeError("No pseudo-cloud pixels selected")

    def calculate(sums, pixels):
        results = {}

        for name in ["cloudy", "model"]:
            mse = sums[name]["squared_error"] / pixels
            rmse = np.sqrt(mse)
            mae = sums[name]["absolute_error"] / pixels
            bias = sums[name]["signed_error"] / pixels
            psnr = -10 * np.log10(np.maximum(mse, 1e-10))

            results[name] = {
                "PSNR": psnr,
                "RMSE": rmse,
                "MAE": mae,
                "Bias": bias,
            }

        return results

    global_results = calculate(global_sums, global_pixels)
    cloud_results = calculate(cloud_sums, cloud_pixels)
    coverage_results = {}

    for bin_name in coverage_bins:
        pixels = coverage_pixels[bin_name]

        if pixels == 0:
            print(f"WARNING: No samples in coverage bin {bin_name}")
            continue

        coverage_results[bin_name] = calculate(
            coverage_sums[bin_name], pixels
        )
    os.makedirs(a.out, exist_ok=True)

    output = os.path.join(a.out, "per_band_global_and_cloud.csv")

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "band",
            "global_cloudy_PSNR", "global_model_PSNR",
            "global_cloudy_RMSE", "global_model_RMSE",
            "global_cloudy_MAE", "global_model_MAE",
            "global_model_bias",
            "cloud_cloudy_PSNR", "cloud_model_PSNR",
            "cloud_cloudy_RMSE", "cloud_model_RMSE",
            "cloud_cloudy_MAE", "cloud_model_MAE",
            "cloud_model_bias",
        ])

        for i, band in enumerate(BANDS):
            writer.writerow([
                band,
                global_results["cloudy"]["PSNR"][i],
                global_results["model"]["PSNR"][i],
                global_results["cloudy"]["RMSE"][i],
                global_results["model"]["RMSE"][i],
                global_results["cloudy"]["MAE"][i],
                global_results["model"]["MAE"][i],
                global_results["model"]["Bias"][i],
                cloud_results["cloudy"]["PSNR"][i],
                cloud_results["model"]["PSNR"][i],
                cloud_results["cloudy"]["RMSE"][i],
                cloud_results["model"]["RMSE"][i],
                cloud_results["cloudy"]["MAE"][i],
                cloud_results["model"]["MAE"][i],
                cloud_results["model"]["Bias"][i],
            ])

    coverage_output = os.path.join(
        a.out, "per_band_coverage_bins.csv"
    )

    with open(coverage_output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "coverage_bin",
            "samples",
            "pseudo_cloud_pixels",
            "band",
            "cloudy_PSNR",
            "model_PSNR",
            "cloudy_RMSE",
            "model_RMSE",
            "cloudy_MAE",
            "model_MAE",
            "model_bias",
        ])

        for bin_name in coverage_bins:
            if bin_name not in coverage_results:
                continue

            results = coverage_results[bin_name]

            for i, band in enumerate(BANDS):
                writer.writerow([
                    bin_name,
                    coverage_samples[bin_name],
                    coverage_pixels[bin_name],
                    band,
                    results["cloudy"]["PSNR"][i],
                    results["model"]["PSNR"][i],
                    results["cloudy"]["RMSE"][i],
                    results["model"]["RMSE"][i],
                    results["cloudy"]["MAE"][i],
                    results["model"]["MAE"][i],
                    results["model"]["Bias"][i],
                ])

    print(f"Coverage-bin results saved: {coverage_output}")
    print(f"Evaluated samples: {total_samples:,}")
    print(f"Global pixels per band: {global_pixels:,}")
    print(f"Pseudo-cloud pixels per band: {cloud_pixels:,}")

    print("\nCloud-region results:")
    print(
        f"{'Band':5s} {'Cloudy MAE':>12s} {'Model MAE':>12s} "
        f"{'Model PSNR':>12s} {'Model Bias':>12s}"
    )

    for i, band in enumerate(BANDS):
        print(
            f"{band:5s} "
            f"{cloud_results['cloudy']['MAE'][i]:12.5f} "
            f"{cloud_results['model']['MAE'][i]:12.5f} "
            f"{cloud_results['model']['PSNR'][i]:12.3f} "
            f"{cloud_results['model']['Bias'][i]:12.5f}"
        )

    print(f"\nSaved: {output}")
    print(
        "Cloud-region metrics use the RGB-derived pseudo-mask, "
        "not a ground-truth cloud mask."
    )


if __name__ == "__main__":
    main()


"""Plot spectral signatures and calculate per-band reconstruction errors."""
import argparse
import csv
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from sen12mscr.dataset import Sen12msCR, read_manifest
from sen12mscr.losses import cloud_mask
from sen12mscr.model import ResUNet


BANDS = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B10", "B11", "B12"
]

TARGETS = {
    "ROIs1158_spring_s2_cloudy_44_p343.tif",
    "ROIs1158_spring_s2_cloudy_140_p866.tif",
    "ROIs1158_spring_s2_cloudy_140_p252.tif",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", default="spectral_signatures")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "off"])
    return p.parse_args()


def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(
        a.ckpt, map_location=device, weights_only=False
    )
    saved_args = checkpoint.get("args", {})
    use_sar = not saved_args.get("no_sar", False)

    model = ResUNet(
        base=saved_args.get("base", 48),
        use_sar=use_sar,
    ).to(device).eval()

    weights = checkpoint.get("ema", checkpoint.get("net"))
    if weights is None:
        raise RuntimeError("Checkpoint has no EMA or network weights.")

    model.load_state_dict(weights)

    amp = a.amp
    if device.type != "cuda":
        amp = "off"
    elif amp == "bf16" and not torch.cuda.is_bf16_supported():
        amp = "fp16"

    dtype = torch.bfloat16 if amp == "bf16" else torch.float16

    manifest = read_manifest(a.manifest)
    items = [
        item for item in manifest["splits"]["test"]
        if item["name"] in TARGETS
    ]

    found = {item["name"] for item in items}
    missing = TARGETS - found
    if missing:
        raise RuntimeError(f"Samples missing from test manifest: {missing}")

    dataset = Sen12msCR(
        a.data_root,
        items,
        use_sar=use_sar,
    )

    loader = DataLoader(
        dataset,
        batch_size=a.batch_size,
        shuffle=False,
        num_workers=a.num_workers,
        pin_memory=device.type == "cuda",
    )

    summary = []
    band_records = []

    with torch.no_grad():
        for batch in loader:
            cloudy = batch["cloudy"].to(device)
            clear = batch["clear"].to(device)
            sar = batch["s1"].to(device) if use_sar else None

            if device.type == "cuda" and amp != "off":
                with torch.autocast("cuda", dtype=dtype):
                    prediction = model(cloudy, sar)
            else:
                prediction = model(cloudy, sar)

            masks = cloud_mask(cloudy, clear)

            for i, name in enumerate(batch["name"]):
                mask = masks[i].detach().float().squeeze().cpu().numpy()

                selected = mask > 0.5

                if selected.sum() < 10:
                    count = max(10, int(mask.size * 0.05))
                    threshold = np.partition(
                        mask.ravel(), -count
                    )[-count]
                    selected = mask >= threshold

                cloudy_np = cloudy[i].detach().float().cpu().numpy()
                prediction_np = prediction[i].detach().float().cpu().numpy()
                clear_np = clear[i].detach().float().cpu().numpy()

                def spectrum(array):
                    return array[:, selected].mean(axis=1)

                cloudy_spectrum = spectrum(cloudy_np)
                predicted_spectrum = spectrum(prediction_np)
                clear_spectrum = spectrum(clear_np)

                # Calculate per-band errors over selected pixels.
                for band_idx, band_name in enumerate(BANDS):
                    cloudy_error = (
                        cloudy_np[band_idx][selected]
                        - clear_np[band_idx][selected]
                    )
                    model_error = (
                        prediction_np[band_idx][selected]
                        - clear_np[band_idx][selected]
                    )

                    band_records.append({
                        "image": name,
                        "band": band_name,
                        "cloudy_MAE": float(
                            np.abs(cloudy_error).mean()
                        ),
                        "model_MAE": float(
                            np.abs(model_error).mean()
                        ),
                        "model_bias": float(model_error.mean()),
                    })

                # Plot spectral signatures.
                plt.figure(figsize=(11, 6))

                plt.plot(
                    BANDS, cloudy_spectrum,
                    marker="o", label="Cloudy input"
                )
                plt.plot(
                    BANDS, predicted_spectrum,
                    marker="o", label="Model prediction"
                )
                plt.plot(
                    BANDS, clear_spectrum,
                    marker="o", label="Clear reference"
                )

                plt.title(
                    f"Mean spectral signature - {name}\n"
                    f"Selected pixels: {selected.sum():,}"
                )
                plt.xlabel("Sentinel-2 spectral band")
                plt.ylabel("Normalized reflectance")
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()

                output_name = name.replace(".tif", "_spectrum.png")
                plt.savefig(
                    os.path.join(a.out, output_name),
                    dpi=180
                )
                plt.close()

                summary.append({
                    "image": name,
                    "selected_pixels": int(selected.sum()),
                    "mask_coverage": float(selected.mean()),
                })

                print(
                    f"Processed {name} | "
                    f"selected pixels: {selected.sum():,}"
                )

    # Save selected-pixel summary.
    with open(
        os.path.join(a.out, "summary.csv"),
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "selected_pixels", "mask_coverage"],
        )
        writer.writeheader()
        writer.writerows(summary)

    # Save per-band errors.
    with open(
        os.path.join(a.out, "per_band_errors.csv"),
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image", "band", "cloudy_MAE",
                "model_MAE", "model_bias"
            ],
        )
        writer.writeheader()
        writer.writerows(band_records)

    print(f"\nSpectral plots and CSV files saved in: {a.out}")
    print(f"Per-band error records: {len(band_records)}")


if __name__ == "__main__":
    main()

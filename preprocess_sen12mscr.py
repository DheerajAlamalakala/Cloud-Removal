"""
Convert SEN12MS-CR (or SEN12MS-CR-TS) GeoTIFF tiles into PNG patches laid out
for CloudDataset (trainA = cloudy, trainB = cloud-free).

SEN12MS-CR's Sentinel-2 tiles are 13-band GeoTIFFs. This script extracts
bands B4, B3, B2 (Red, Green, Blue -> indices 3, 2, 1 in a 0-indexed 13-band
stack) and rescales them to 8-bit for a standard RGB CycleGAN. If you want
to train on all 13 bands instead, set --channels 13, which saves .npy files
instead of PNGs (datasets.py already knows how to load these).

Requires: rasterio (pip install rasterio) or tifffile as a fallback.

Usage:
    python preprocess_sen12mscr.py \
        --sen12mscr_root /path/to/SEN12MSCR \
        --output_root ./data \
        --split train \
        --channels 3
"""

import argparse
import glob
import os

import numpy as np

try:
    import rasterio
    HAVE_RASTERIO = True
except ImportError:
    HAVE_RASTERIO = False
    try:
        import tifffile
    except ImportError:
        raise SystemExit(
            "Need either 'rasterio' or 'tifffile' installed to read the GeoTIFFs.\n"
            "Install with: pip install rasterio   (preferred)\n"
            "         or:  pip install tifffile"
        )

from PIL import Image


def read_tif(path):
    if HAVE_RASTERIO:
        with rasterio.open(path) as src:
            arr = src.read()  # (bands, H, W)
        return np.transpose(arr, (1, 2, 0))  # -> (H, W, bands)
    else:
        arr = tifffile.imread(path)
        if arr.ndim == 3 and arr.shape[0] <= 20:
            arr = np.transpose(arr, (1, 2, 0))
        return arr


def to_rgb_uint8(arr, bands=(3, 2, 1), clip_percentile=2):
    """Sentinel-2 L2A bands are reflectance*10000. Percentile-stretch to 8-bit RGB."""
    rgb = arr[:, :, bands].astype(np.float32)
    out = np.zeros_like(rgb, dtype=np.uint8)
    for c in range(3):
        band = rgb[:, :, c]
        lo, hi = np.percentile(band, clip_percentile), np.percentile(band, 100 - clip_percentile)
        band = np.clip((band - lo) / max(hi - lo, 1e-6), 0, 1)
        out[:, :, c] = (band * 255).astype(np.uint8)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sen12mscr_root", required=True, help="root folder of the extracted SEN12MS-CR dataset")
    p.add_argument("--output_root", default="./data")
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--channels", type=int, default=3, choices=[3, 13])
    p.add_argument("--max_tiles", type=int, default=0, help="0 = no limit, otherwise cap number of tiles (useful for a quick test run)")
    args = p.parse_args()

    domain_A = "trainA" if args.split == "train" else "testA"  # cloudy
    domain_B = "trainB" if args.split == "train" else "testB"  # cloud-free
    out_A = os.path.join(args.output_root, domain_A)
    out_B = os.path.join(args.output_root, domain_B)
    os.makedirs(out_A, exist_ok=True)
    os.makedirs(out_B, exist_ok=True)

    # SEN12MS-CR ships as scene folders containing s2_cloudy/ and s2_cloudfree/ subfolders
    # (naming can vary slightly by release; adjust the glob patterns below if needed).
    cloudy_files = sorted(glob.glob(os.path.join(args.sen12mscr_root, "**", "*s2_cloudy*", "*.tif"), recursive=True))
    clear_files = sorted(glob.glob(os.path.join(args.sen12mscr_root, "**", "*s2_cloudfree*", "*.tif"), recursive=True))

    if not cloudy_files or not clear_files:
        raise SystemExit(
            f"Found {len(cloudy_files)} cloudy and {len(clear_files)} cloud-free tifs under "
            f"{args.sen12mscr_root}. Check the folder naming in the extracted dataset and adjust "
            "the glob patterns in this script if it doesn't match 's2_cloudy' / 's2_cloudfree'."
        )

    if args.max_tiles:
        cloudy_files = cloudy_files[: args.max_tiles]
        clear_files = clear_files[: args.max_tiles]

    print(f"Converting {len(cloudy_files)} cloudy tiles -> {out_A}")
    for path in cloudy_files:
        arr = read_tif(path)
        name = os.path.splitext(os.path.basename(path))[0]
        if args.channels == 3:
            rgb = to_rgb_uint8(arr)
            Image.fromarray(rgb).save(os.path.join(out_A, f"{name}.png"))
        else:
            np.save(os.path.join(out_A, f"{name}.npy"), arr.astype(np.float32))

    print(f"Converting {len(clear_files)} cloud-free tiles -> {out_B}")
    for path in clear_files:
        arr = read_tif(path)
        name = os.path.splitext(os.path.basename(path))[0]
        if args.channels == 3:
            rgb = to_rgb_uint8(arr)
            Image.fromarray(rgb).save(os.path.join(out_B, f"{name}.png"))
        else:
            np.save(os.path.join(out_B, f"{name}.npy"), arr.astype(np.float32))

    print("Done.")


if __name__ == "__main__":
    main()

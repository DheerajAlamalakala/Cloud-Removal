"""Audit one decoded Hugging Face Parquet sample from each season."""
from __future__ import annotations

import argparse
import os

import numpy as np

from sen12mscr.parquet_dataset import Sen12msCRParquet, make_parquet_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--manifest", default="./runs/run2_parquet/split_manifest.json")
    parser.add_argument("--rebuild_manifest", action="store_true")
    args = parser.parse_args()

    manifest = make_parquet_manifest(args.data_root, args.manifest, rebuild=args.rebuild_manifest)
    print("Hub-provided split counts:")
    for split, groups in manifest["splits"].items():
        rows = sum(int(group["rows"]) for group in groups)
        print(f"  {split}: {rows:,} patches in {len(groups)} row groups")

    groups_by_season = {}
    for group in manifest["splits"]["train"]:
        groups_by_season.setdefault(group["season"], group)
    missing = {"spring", "summer", "fall", "winter"}.difference(groups_by_season)
    if missing:
        raise RuntimeError(f"Training split is missing seasons: {sorted(missing)}")

    for season in ("spring", "summer", "fall", "winter"):
        group = groups_by_season[season]
        dataset = Sen12msCRParquet(args.data_root, [group], train=False, use_sar=True, batch_size=1)
        sample = next(iter(dataset))
        for key, expected in (("cloudy", (13, 256, 256)), ("clear", (13, 256, 256)), ("s1", (2, 256, 256))):
            tensor = sample[key][0]
            if tuple(tensor.shape) != expected:
                raise RuntimeError(f"{season} {key}: expected {expected}, got {tuple(tensor.shape)}")
            array = tensor.numpy()
            if not np.isfinite(array).all():
                raise RuntimeError(f"{season} {key} contains NaN or infinite values")
        if float(sample["cloudy"].min()) < 0 or float(sample["cloudy"].max()) > 1:
            raise RuntimeError(f"{season} cloudy reflectance is outside [0, 1]")
        if float(sample["clear"].min()) < 0 or float(sample["clear"].max()) > 1:
            raise RuntimeError(f"{season} clear reflectance is outside [0, 1]")
        print(
            f"{season:6s} {sample['name'][0]}: cloudy={tuple(sample['cloudy'][0].shape)} "
            f"clear={tuple(sample['clear'][0].shape)} SAR={tuple(sample['s1'][0].shape)} "
            f"S2 range=[{sample['cloudy'].min():.3f}, {sample['cloudy'].max():.3f}] "
            f"SAR range=[{sample['s1'].min():.3f}, {sample['s1'].max():.3f}]"
        )
    print(f"Manifest: {os.path.abspath(args.manifest)}")
    print("Sample audit passed. This checks one training row group per season; it does not read the full dataset.")


if __name__ == "__main__":
    main()

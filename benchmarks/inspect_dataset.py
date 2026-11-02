"""Quickly audit a few SEN12MS-CR TIFF triplets before starting a long run."""
import argparse
import os
import numpy as np

from sen12mscr.dataset import scan, _read_tif


def stats(label, arr):
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        print(f"{label}: no finite pixels")
        return
    q = np.percentile(finite, [0, 1, 50, 99, 100])
    print(
        f"{label}: shape={arr.shape}, dtype={arr.dtype}, "
        f"min={q[0]:.3f}, p1={q[1]:.3f}, median={q[2]:.3f}, p99={q[3]:.3f}, max={q[4]:.3f}"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", required=True)
    p.add_argument("--rois", nargs="+", default=["ROIs1158_spring"])
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--no_sar", action="store_true")
    a = p.parse_args()
    items = scan(a.data_root, a.rois, include_sar=not a.no_sar)
    print(f"\nFound {len(items):,} samples. Inspecting first {min(a.samples, len(items))}.")
    for item in items[:a.samples]:
        print(f"\n{item['roi']} / scene {item['scene']} / {item['name']}")
        cloudy = _read_tif(os.path.join(a.data_root, item["cloudy"]))
        clear = _read_tif(os.path.join(a.data_root, item["clear"]))
        stats("cloudy S2 raw", cloudy)
        stats("clear S2 raw", clear)
        if item.get("s1"):
            sar = _read_tif(os.path.join(a.data_root, item["s1"]))
            stats("S1 raw dB", sar)
        if cloudy.shape != clear.shape or cloudy.shape[0] != 13:
            print("WARNING: cloudy/clear S2 band count or dimensions do not match expected 13-band layout")


if __name__ == "__main__":
    main()

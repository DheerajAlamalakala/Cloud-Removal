import time
import argparse
import os
import random
import rasterio
from sen12mscr.dataset import scan

def main():
    parser = argparse.ArgumentParser(description="Benchmark paired SEN12MS-CR TIFF reads.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--rois", nargs="+", default=["ROIs1158_spring"])
    parser.add_argument("--samples", type=int, default=32)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("samples must be positive")

    items = scan(args.data_root, args.rois, include_sar=True)
    if not items:
        parser.error("No complete triplets found for the selected ROIs")
    random.seed(42)
    chosen = random.sample(items, min(args.samples, len(items)))
    read_times = []
    start_total = time.perf_counter()
    for i, item in enumerate(chosen, 1):
        start = time.perf_counter()
        for key in ("s1", "clear", "cloudy"):
            with rasterio.open(os.path.join(args.data_root, item[key])) as src:
                src.read()
        elapsed = time.perf_counter() - start
        read_times.append(elapsed)
        print(f"Triplet {i:02d}/{len(chosen)} | {elapsed:.4f}s")

    total = time.perf_counter() - start_total
    print("\n--- PAIRED TIFF READ RESULTS ---")
    print(f"Triplets: {len(chosen)}")
    print(f"Total time: {total:.3f}s")
    print(f"Average time/triplet: {total/len(chosen):.4f}s")
    print(f"Throughput: {len(chosen)/total:.3f} triplets/sec")
    print(f"Average time per TIFF: {sum(read_times)/(len(chosen)*3):.4f}s")


if __name__ == "__main__":
    main()

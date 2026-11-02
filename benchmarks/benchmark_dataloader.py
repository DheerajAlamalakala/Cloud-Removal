import os
import argparse
import time
import torch
from torch.utils.data import DataLoader
from sen12mscr.dataset import Sen12msCR, read_manifest

def main():
    parser = argparse.ArgumentParser(description="Benchmark SEN12MS-CR TIFF loading throughput.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.samples < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("samples and batch_size must be positive; workers cannot be negative")
    items = read_manifest(args.manifest)["splits"]["train"][:args.samples]
    if not items:
        parser.error("The manifest train split is empty")

    ds = Sen12msCR(
        args.data_root, items,
        train=True,
        crop=128,
        use_sar=True,
        s2_scale=10000.0,
        sar_min_db=-25.0,
        sar_max_db=0.0,
    )

    print(f"Benchmark samples: {len(ds)}")
    print(f"Batch size: {args.batch_size} | Crop: 128 | SAR: ON")
    print()

    for workers in (args.workers,):
        kwargs = {
            "batch_size": args.batch_size,
            "shuffle": True,
            "num_workers": workers,
            "pin_memory": torch.cuda.is_available(),
        }

        if workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = 2

        loader = DataLoader(ds, **kwargs)
        iterator = iter(loader)

        warmup = min(10, len(loader) - 1)
        for _ in range(warmup):
            next(iterator)

        start = time.perf_counter()
        batches = min(100, len(loader) - warmup)
        if batches < 1:
            raise RuntimeError("Need at least one batch after warmup; increase --samples")

        for _ in range(batches):
            next(iterator)

        elapsed = time.perf_counter() - start
        samples = batches * args.batch_size

        print(
            f"Workers={workers} | "
            f"Time={elapsed:.2f}s | "
            f"Average batch={elapsed/batches:.4f}s | "
            f"Throughput={samples/elapsed:.2f} samples/sec"
        )

        del iterator, loader

if __name__ == "__main__":
    main()

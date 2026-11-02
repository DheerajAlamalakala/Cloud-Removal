import time
import argparse
import torch
from torch.utils.data import DataLoader
from sen12mscr.dataset import Sen12msCR, read_manifest

def main():
    parser = argparse.ArgumentParser(description="Benchmark SEN12MS-CR batch transfer to CUDA.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.samples < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("samples and batch_size must be positive; workers cannot be negative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.device("cuda")
    items = read_manifest(args.manifest)["splits"]["train"][:args.samples]
    if not items:
        parser.error("The manifest train split is empty")

    ds = Sen12msCR(args.data_root, items, train=True, crop=128, use_sar=True)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        **({"prefetch_factor": 2} if args.workers > 0 else {}),
    )

    iterator = iter(loader)

    warmup = min(10, len(loader) - 1)
    for _ in range(warmup):
        batch = next(iterator)
        batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                 for k, v in batch.items()}

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    batches = min(100, len(loader) - warmup)
    if batches < 1:
        raise RuntimeError("Need at least one batch after warmup; increase --samples")

    for _ in range(batches):
        batch = next(iterator)
        batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                 for k, v in batch.items()}

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print(f"GPU transfer time: {elapsed:.2f}s")
    print(f"Throughput: {batches * args.batch_size / elapsed:.2f} samples/sec")
    print(f"Peak allocated VRAM: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    print(f"Peak reserved VRAM: {torch.cuda.max_memory_reserved()/1024**3:.2f} GB")

    del iterator, loader, batch
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

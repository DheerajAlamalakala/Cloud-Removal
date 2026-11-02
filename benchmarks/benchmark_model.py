
import argparse
import gc
import time
import torch
from torch.utils.data import DataLoader

from sen12mscr.dataset import Sen12msCR, read_manifest
from sen12mscr.model import ResUNet
from sen12mscr.losses import CRLoss

def run_benchmark(batch_size, items, data_root, workers):
    print(f"\n{'='*45}")
    print(f"Testing batch size: {batch_size}")
    print(f"{'='*45}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    ds = Sen12msCR(
        data_root, items, train=True, crop=128, use_sar=True
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        **({"prefetch_factor": 2} if workers > 0 else {}),
    )

    model = ResUNet(base=48, use_sar=True).cuda()
    criterion = CRLoss(
        l_ssim=0.5,
        l_edge=2.0,
        l_lpips=0.1,
        cloud_w=4.0,
        lpips_net="alex",
    ).cuda()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-4
    )

    iterator = iter(loader)

    def train_step(batch):
        cloudy = batch["cloudy"].cuda(non_blocking=True)
        clear = batch["clear"].cuda(non_blocking=True)
        sar = batch["s1"].cuda(non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(cloudy, sar)

        loss, parts = criterion(
            prediction.float(),
            clear.float(),
            cloudy.float(),
        )

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss: {loss.item()}")

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0
        )
        optimizer.step()

        return loss.item()

    model.train()

    try:
        print("Warming up...")
        for _ in range(5):
            train_step(next(iterator))

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        print("Measuring...")
        steps = 30
        start = time.perf_counter()

        for _ in range(steps):
            train_step(next(iterator))

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        allocated = torch.cuda.max_memory_allocated() / 1024**3
        reserved = torch.cuda.max_memory_reserved() / 1024**3

        print("\nRESULT")
        print(f"Batch size: {batch_size}")
        print(f"Steps: {steps}")
        print(f"Time: {elapsed:.2f} seconds")
        print(f"Average step: {elapsed/steps:.4f} seconds")
        print(f"Throughput: {steps*batch_size/elapsed:.2f} samples/sec")
        print(f"Peak allocated VRAM: {allocated:.2f} GB")
        print(f"Peak reserved VRAM: {reserved:.2f} GB")

    finally:
        del iterator, loader, optimizer, criterion, model, ds
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Benchmark U-Net training throughput and peak VRAM.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.samples < 1 or args.workers < 0:
        parser.error("samples must be positive; workers cannot be negative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    torch.backends.cudnn.benchmark = True

    items = read_manifest(args.manifest)["splits"]["train"][:args.samples]
    if not items:
        parser.error("The manifest train split is empty")
    if len(items) < 280:
        parser.error("--samples must provide at least 280 patches for warmup and measurement")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Dataset samples: {len(items)}")
    print("Loss: L1 + SSIM + Edge + LPIPS (AlexNet)")
    print("Precision: BF16")

    run_benchmark(4, items, args.data_root, args.workers)
    run_benchmark(8, items, args.data_root, args.workers)


if __name__ == "__main__":
    main()

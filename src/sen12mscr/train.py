"""Train the SAR-assisted 13-band residual U-Net on SEN12MS-CR."""
from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import random
import shutil
import time
from contextlib import nullcontext

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from .dataset import RGB, Sen12msCR, make_splits
from .losses import CRLoss, cloud_mask, per_image_metrics
from .model import ResUNet


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", required=True, help="Root containing the extracted TUM season folders")
    p.add_argument("--rois", nargs="+", default=["ROIs1158_spring"], help="Start with spring; add summer/fall/winter for all seasons")
    p.add_argument("--manifest", default=None, help="Split manifest path (default: OUT/split_manifest.json)")
    p.add_argument("--rebuild_manifest", action="store_true", help="Regenerate the split manifest")
    p.add_argument("--split_mode", default="official", choices=["official", "scene"], help="Official author ROI split (default) or custom random scene split")
    p.add_argument("--max_train", type=int, default=0, help="Debug cap on train patches; 0 uses all training patches")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    p.add_argument("--crop", type=int, default=128)
    p.add_argument("--base", type=int, default=48, help="U-Net base width; 32/48 is safer on 8GB VRAM")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--no_sar", action="store_true", help="Optical-only ablation; does not require S1 files")
    p.add_argument("--s2_scale", type=float, default=10000.0, help="Raw S2 reflectance divisor")
    p.add_argument("--sar_min_db", type=float, default=-25.0)
    p.add_argument("--sar_max_db", type=float, default=0.0)
    p.add_argument("--lambda_ssim", type=float, default=0.5)
    p.add_argument("--lambda_edge", type=float, default=2.0)
    p.add_argument("--lambda_lpips", type=float, default=0.1)
    p.add_argument("--lpips_net", default="alex", choices=["vgg", "alex"])
    p.add_argument("--cloud_weight", type=float, default=4.0)
    p.add_argument("--val_max", type=int, default=400, help="Max validation patches; 0 uses all")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--amp", default="auto", choices=["auto", "bf16", "fp16", "off"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="./results/sen12mscr/training", help="Local run/checkpoint directory")
    p.add_argument("--backup_dir", default=None, help="Optional Drive/Google Drive sync folder for checkpoints")
    p.add_argument("--backup_every", type=int, default=1, help="Copy checkpoints to backup_dir every N epochs")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume_from", default=None, help="Optional checkpoint path, e.g. last.pt on Google Drive")
    p.add_argument("--init_from", default=None, help="Initialize model weights from an EMA checkpoint")
    return p.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is False")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def choose_amp(device: torch.device, requested: str) -> str:
    if device.type != "cuda" or requested == "off":
        return "off"
    if requested == "auto":
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if requested == "bf16" and not torch.cuda.is_bf16_supported():
        print("WARNING: GPU does not report BF16 support; switching AMP to fp16.")
        return "fp16"
    return requested


def autocast_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "off":
        return nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def atomic_save(obj, path: str):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def backup_files(out_dir: str, backup_dir: str, manifest_path: str):
    os.makedirs(backup_dir, exist_ok=True)
    for filename in ("last.pt", "best_ema.pt", "log.csv"):
        source = os.path.join(out_dir, filename)
        if os.path.isfile(source):
            tmp = os.path.join(backup_dir, filename + ".tmp")
            shutil.copy2(source, tmp)
            os.replace(tmp, os.path.join(backup_dir, filename))
    if os.path.isfile(manifest_path):
        shutil.copy2(manifest_path, os.path.join(backup_dir, "split_manifest.json"))


def main():
    a = parse_args()
    seed_everything(a.seed)
    os.makedirs(a.out, exist_ok=True)
    device = choose_device(a.device)
    amp = choose_amp(device, a.amp)
    use_sar = not a.no_sar
    manifest_path = a.manifest or os.path.join(a.out, "split_manifest.json")

    print(f"Device: {device} | AMP: {amp} | SAR: {use_sar}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")

    manifest = make_splits(
        a.data_root, a.rois, manifest_path, include_sar=use_sar,
        seed=a.seed, rebuild=a.rebuild_manifest, split_mode=a.split_mode,
    )
    splits = manifest["splits"]
    train_items = splits["train"][:a.max_train] if a.max_train else splits["train"]
    val_items = splits["val"]
    if a.val_max and len(val_items) > a.val_max:
        rng = random.Random(a.seed)
        val_items = sorted(rng.sample(val_items, a.val_max), key=lambda x: (x["roi"], x["scene"], x["name"]))
    if not train_items or not val_items:
        raise RuntimeError(f"Need non-empty train and validation sets; got train={len(train_items)}, val={len(val_items)}")

    train_ds = Sen12msCR(
        a.data_root, train_items, train=True, crop=a.crop, use_sar=use_sar,
        s2_scale=a.s2_scale, sar_min_db=a.sar_min_db, sar_max_db=a.sar_max_db,
    )
    val_ds = Sen12msCR(
        a.data_root, val_items, train=False, use_sar=use_sar,
        s2_scale=a.s2_scale, sar_min_db=a.sar_min_db, sar_max_db=a.sar_max_db,
    )
    pin = device.type == "cuda"
    loader_kwargs = dict(num_workers=a.num_workers, pin_memory=pin, persistent_workers=a.num_workers > 0)
    if a.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    train_dl = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, drop_last=False, **loader_kwargs)
    val_dl = DataLoader(val_ds, batch_size=min(8, a.batch_size * 2), shuffle=False, **loader_kwargs)
    if len(train_dl) == 0:
        raise RuntimeError("Training DataLoader is empty. Increase --max_train or reduce --batch_size.")

    net = ResUNet(base=a.base, use_sar=use_sar).to(device)
    ema = copy.deepcopy(net).eval()
    for param in ema.parameters():
        param.requires_grad_(False)
    if a.init_from:
        init_checkpoint = torch.load(
            a.init_from,
            map_location=device,
            weights_only=False,
        )

        if "ema" not in init_checkpoint:
            raise RuntimeError("Initialization checkpoint has no EMA weights")

        checkpoint_args = init_checkpoint.get("args", {})

        if checkpoint_args.get("base", a.base) != a.base:
            raise RuntimeError("Checkpoint base width does not match current model")

        if not a.no_sar and checkpoint_args.get("no_sar", False):
            raise RuntimeError("Checkpoint was trained without SAR")

        net.load_state_dict(init_checkpoint["ema"])
        ema.load_state_dict(init_checkpoint["ema"])

        print(
            f"Initialized from EMA checkpoint, "
            f"saved epoch {init_checkpoint.get('epoch', 'unknown')}"
        )
    print(f"Samples: train={len(train_ds):,} | val={len(val_ds):,} | test={len(splits['test']):,}")
    print(f"Model parameters: {sum(x.numel() for x in net.parameters()) / 1e6:.2f}M")

    criterion = CRLoss(a.lambda_ssim, a.lambda_edge, a.lambda_lpips, a.cloud_weight, a.lpips_net).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=1e-4)
    use_scaler = device.type == "cuda" and amp == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    steps_per_epoch = math.ceil(len(train_dl) / max(1, a.grad_accum))
    total_steps = max(1, a.epochs * steps_per_epoch)

    def lr_at(step: int):
        if step < a.warmup:
            return a.lr * (step + 1) / max(1, a.warmup)
        progress = min(1.0, (step - a.warmup) / max(1, total_steps - a.warmup))
        return a.lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress)))

    start_epoch, global_step, best = 0, 0, -float("inf")
    checkpoint_path = a.resume_from or os.path.join(a.out, "last.pt")
    if a.resume:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
        # When resuming from a Drive checkpoint, bring its companion log/best
        # checkpoint into the local run directory before appending new results.
        if a.resume_from:
            remote_dir = os.path.dirname(os.path.abspath(a.resume_from))
            for filename in ("best_ema.pt", "log.csv"):
                source = os.path.join(remote_dir, filename)
                destination = os.path.join(a.out, filename)
                if os.path.isfile(source) and not os.path.isfile(destination):
                    shutil.copy2(source, destination)
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        previous_args = state.get("args", {})
        for key in ("rois", "split_mode", "seed", "no_sar", "s2_scale", "sar_min_db", "sar_max_db", "base"):
            if key in previous_args and previous_args[key] != getattr(a, key):
                raise RuntimeError(
                    f"Cannot resume: checkpoint {key}={previous_args[key]!r}, "
                    f"current run {key}={getattr(a, key)!r}. Use a new run or restore matching settings."
                )
        net.load_state_dict(state["net"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["opt"])
        if "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        start_epoch, global_step, best = int(state["epoch"]), int(state["step"]), float(state["best"])
        print(f"Resumed from epoch {start_epoch}, optimizer step {global_step}")
        if state.get("args", {}).get("epochs") != a.epochs:
            print("WARNING: --epochs differs from the checkpoint; the cosine LR schedule will be recalculated.")

    @torch.no_grad()
    def validate(model):
        model.eval()
        cloud_psnr, full_psnr = [], []
        for batch in val_dl:
            cloudy = batch["cloudy"].to(device, non_blocking=True)
            clear = batch["clear"].to(device, non_blocking=True)
            sar = batch["s1"].to(device, non_blocking=True) if use_sar else None
            with autocast_context(device, amp):
                prediction = model(cloudy, sar)
            mask = cloud_mask(cloudy, clear)
            metrics = per_image_metrics(prediction.float(), clear, mask)
            keep = metrics["mask_cov"] >= 0.01
            cloud_psnr.extend(metrics["PSNR_cloud"][keep].tolist())
            full_psnr.extend(metrics["PSNR_full"].tolist())
        cloud_score = float(np.mean(cloud_psnr)) if cloud_psnr else float("nan")
        full_score = float(np.mean(full_psnr)) if full_psnr else float("nan")
        return cloud_score, full_score

    log_path = os.path.join(a.out, "log.csv")
    write_header = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
    with open(log_path, "a", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        if write_header:
            writer.writerow(["epoch", "step", "val_psnr_cloud", "val_psnr_full", "loss", "l1", "ssim", "edge", "lpips", "seconds"])
        for epoch in range(start_epoch, a.epochs):
            net.train()
            start_time = time.time()
            sums = {}
            optimizer.zero_grad(set_to_none=True)
            for batch_idx, batch in enumerate(train_dl):
                cloudy = batch["cloudy"].to(device, non_blocking=True)
                clear = batch["clear"].to(device, non_blocking=True)
                sar = batch["s1"].to(device, non_blocking=True) if use_sar else None
                for group in optimizer.param_groups:
                    group["lr"] = lr_at(global_step)

                with autocast_context(device, amp):
                    prediction = net(cloudy, sar)

                # Compute reconstruction losses in FP32, outside autocast.
                loss, parts = criterion(prediction.float(), clear.float(), cloudy.float())

                group_start = (batch_idx // a.grad_accum) * a.grad_accum
                group_size = min(a.grad_accum, len(train_dl) - group_start)
                scaled_loss = loss / group_size

                if use_scaler:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                do_step = ((batch_idx + 1) % a.grad_accum == 0) or (batch_idx + 1 == len(train_dl))
                if do_step:
                    if use_scaler:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    if use_scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    decay = min(a.ema, (1 + global_step) / (10 + global_step))
                    with torch.no_grad():
                        for ema_param, param in zip(ema.parameters(), net.parameters()):
                            ema_param.lerp_(param, 1 - decay)
                        for ema_buffer, buffer in zip(ema.buffers(), net.buffers()):
                            ema_buffer.copy_(buffer)

                sums["total"] = sums.get("total", 0.0) + loss.detach()
                for key, value in parts.items():
                    sums[key] = sums.get(key, 0.0) + value.detach()
                if global_step and global_step % 100 == 0 and do_step:
                    print(f"epoch {epoch + 1} | step {global_step} | loss {loss.detach().item():.4f} | lr {lr_at(global_step):.2e}", flush=True)

            averages = {key: float((value / len(train_dl)).cpu()) for key, value in sums.items()}
            cloud_score, full_score = validate(ema)
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch + 1}/{a.epochs} | {elapsed:.0f}s | loss={averages.get('total', float('nan')):.4f} "
                f"| val cloud PSNR={cloud_score:.3f} | val full PSNR={full_score:.3f} | parts={averages}",
                flush=True,
            )
            writer.writerow([
                epoch + 1, global_step, cloud_score, full_score,
                averages.get("total", ""), averages.get("l1", ""), averages.get("ssim", ""),
                averages.get("edge", ""), averages.get("lpips", ""), round(elapsed, 2),
            ])
            log_file.flush()

            score_for_best = cloud_score if math.isfinite(cloud_score) else full_score
            state = {
                "net": net.state_dict(), "ema": ema.state_dict(), "opt": optimizer.state_dict(),
                "scaler": scaler.state_dict(), "epoch": epoch + 1, "step": global_step,
                "best": max(best, score_for_best), "args": vars(a),
            }
            atomic_save(state, os.path.join(a.out, "last.pt"))
            if math.isfinite(score_for_best) and score_for_best > best:
                best = score_for_best
                atomic_save(
                    {"ema": ema.state_dict(), "args": vars(a), "epoch": epoch + 1, "best_metric": score_for_best},
                    os.path.join(a.out, "best_ema.pt"),
                )

            if (epoch + 1) % 5 == 0:
                try:
                    sample = next(iter(val_dl))
                    c = sample["cloudy"].to(device)
                    y = sample["clear"].to(device)
                    s = sample["s1"].to(device) if use_sar else None
                    with torch.no_grad(), autocast_context(device, amp):
                        pred = ema(c, s).float().clamp(0, 1)
                    grid = torch.cat([(x[:4, RGB] * 2.5).clamp(0, 1) for x in (c, pred, y)], 0)
                    save_image(grid, os.path.join(a.out, f"samples_ep{epoch + 1}.png"), nrow=4)
                except StopIteration:
                    pass

            if a.backup_dir and (epoch + 1) % max(1, a.backup_every) == 0:
                backup_files(a.out, a.backup_dir, manifest_path)
                print(f"Checkpoint backup copied to: {a.backup_dir}")

    print(f"Training complete. Best validation score: {best:.4f}")
    print(f"Local checkpoints: {os.path.abspath(a.out)}")
    if a.backup_dir:
        print(f"Backup directory: {os.path.abspath(a.backup_dir)}")


if __name__ == "__main__":
    main()

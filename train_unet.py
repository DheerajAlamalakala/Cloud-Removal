"""
U-Net baseline for cloud removal (supervised, no GAN, no cycle).

Uses the SAME dataset class, mask files, augmentation, EMA, AMP and paired losses as train.py,
so the only differences from the CycleGAN run are: the generator (U-Net) and the missing
adversarial / cycle terms. That makes the comparison fair.

Loss = lambda_cloud * L1(inside cloud mask)  + lambda_clear * L1(outside mask)
     + lambda_idt_clear * L1(pred vs cloudy input, outside mask)
     + lambda_edge * Sobel edge loss + lambda_ssim * (1 - SSIM)

Checkpoints use the same keys as train.py (G_A2B, G_A2B_ema, args, epoch) and set args["model"]="unet",
so evaluate.py can load them once it has the small patch (see PATCH_evaluate.txt).

Example (matches the CycleGAN budget of about 235 passes over 515 images):
    python train_unet.py --data_root ./data --image_size 256 --channels 3 --batch_size 4 --n_epochs 235 --decay_epoch 135 --output_dir ./checkpoints_unet --sample_dir ./results/unet --checkpoint_interval 20 --num_workers 2 --lambda_edge 20 --lambda_ssim 5 --ema 0.998 --strong_aug 1 --amp 1 --resume ./checkpoints_unet
"""
import argparse
import copy
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from datasets import CloudDataset
from losses import edge_loss, ssim_loss
from unet import UNetGenerator


def get_args():
    p = argparse.ArgumentParser(description="U-Net baseline for cloud removal")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--output_dir", type=str, default="./checkpoints_unet")
    p.add_argument("--sample_dir", type=str, default="./results/unet")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--channels", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--n_epochs", type=int, default=235)
    p.add_argument("--decay_epoch", type=int, default=135)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--base", type=int, default=64, help="U-Net base filters")

    p.add_argument("--lambda_cloud", type=float, default=10.0)
    p.add_argument("--lambda_clear", type=float, default=30.0)
    p.add_argument("--lambda_idt_clear", type=float, default=10.0)
    p.add_argument("--lambda_edge", type=float, default=20.0)
    p.add_argument("--edge_cloud_boost", type=float, default=1.0)
    p.add_argument("--lambda_ssim", type=float, default=5.0)
    p.add_argument("--mask_dir", type=str, default="auto")
    p.add_argument("--mask_thresh", type=float, default=0.1)

    p.add_argument("--ema", type=float, default=0.998)
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--strong_aug", type=int, default=1)
    p.add_argument("--no_strict_pairing", action="store_true")
    p.add_argument("--checkpoint_interval", type=int, default=20)
    p.add_argument("--sample_every", type=int, default=5, help="save a preview grid every N epochs")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--max_batches", type=int, default=0, help="smoke test: stop each epoch after N batches")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def lr_factor(epoch, n_epochs, decay_epoch):
    """Same absolute schedule as train.py: full LR through decay_epoch, then linear decay."""
    if epoch <= decay_epoch:
        return 1.0
    return max(0.0, 1.0 - (epoch - decay_epoch) / float(n_epochs - decay_epoch + 1))


def make_mask(real_A, real_B, thresh):
    diff = (real_A - real_B).abs().mean(1, keepdim=True) / 2.0
    mask = (diff > thresh).float()
    return F.max_pool2d(mask, 5, 1, 2).detach()


def masked_l1(pred, target, mask):
    return (mask * (pred - target).abs()).sum() / (mask.sum() * pred.shape[1] + 1e-6)


def denorm(t):
    return (t * 0.5 + 0.5).clamp(0, 1)


def make_scaler(enabled):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


@torch.no_grad()
def update_ema(ema_model, model, decay):
    for pe, p in zip(ema_model.parameters(), model.parameters()):
        pe.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
    for be, b in zip(ema_model.buffers(), model.buffers()):
        be.copy_(b)


def save_checkpoint(output_dir, epoch, G, ema, opt, args, numbered):
    os.makedirs(output_dir, exist_ok=True)
    cfg = vars(args).copy()
    cfg["model"] = "unet"
    obj = {"epoch": epoch, "G_A2B": G.state_dict(), "opt_G": opt.state_dict(), "args": cfg}
    if ema is not None:
        obj["G_A2B_ema"] = ema.state_dict()

    def _atomic(path):
        tmp = path + ".tmp"
        torch.save(obj, tmp)
        os.replace(tmp, path)

    _atomic(os.path.join(output_dir, "checkpoint_latest.pt"))
    if numbered:
        _atomic(os.path.join(output_dir, f"checkpoint_epoch{epoch}.pt"))


def main():
    args = get_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")
    use_amp = bool(args.amp) and device.type == "cuda"
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)

    # ---------------- data (same class and mask logic as train.py) ----------------
    mask_dir = None
    if args.mask_dir == "auto":
        cand = os.path.join(args.data_root, "trainM")
        if os.path.isdir(cand) and len(os.listdir(cand)) > 0:
            mask_dir = cand
    elif args.mask_dir.lower() not in ("", "none"):
        mask_dir = args.mask_dir
    print(f"Cloud mask source: {'dataset masks from ' + mask_dir if mask_dir else 'computed from |cloudy - clear|'}")

    train_dataset = CloudDataset(
        domain_A_dir=os.path.join(args.data_root, "trainA"),
        domain_B_dir=os.path.join(args.data_root, "trainB"),
        image_size=args.image_size, channels=args.channels, mode="train",
        paired=True, mask_dir=mask_dir, strong_aug=bool(args.strong_aug),
        strict_pairing=not args.no_strict_pairing,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True,
                              pin_memory=(device.type == "cuda"))

    # previews come from the VALIDATION split (never the test split)
    val_A, val_B = os.path.join(args.data_root, "valA"), os.path.join(args.data_root, "valB")
    have_val = os.path.isdir(val_A) and os.path.isdir(val_B) and len(os.listdir(val_A)) > 0
    if have_val:
        val_dataset = CloudDataset(val_A, val_B, image_size=args.image_size, channels=args.channels, mode="test")
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    print(f"Train pairs per epoch: {len(train_dataset)} | batches/epoch: {len(train_loader)}")

    # ---------------- model ----------------
    G = UNetGenerator(args.channels, args.channels, base=args.base).to(device)
    print(f"U-Net parameters: {sum(p.numel() for p in G.parameters()) / 1e6:.1f}M")
    ema = None
    if args.ema > 0:
        ema = copy.deepcopy(G).eval()
        for p_ in ema.parameters():
            p_.requires_grad_(False)

    opt = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    scaler = make_scaler(use_amp)

    start_epoch = 1
    if args.resume:
        path = args.resume if args.resume.endswith(".pt") else os.path.join(args.resume, "checkpoint_latest.pt")
        if os.path.isfile(path):
            print(f"Resuming from {path}")
            ck = torch.load(path, map_location=device)
            G.load_state_dict(ck["G_A2B"])
            opt.load_state_dict(ck["opt_G"])
            if ema is not None:
                ema.load_state_dict(ck["G_A2B_ema"] if "G_A2B_ema" in ck else ck["G_A2B"])
            start_epoch = ck["epoch"] + 1
        else:
            print(f"WARNING: resume path {path} not found, starting fresh.")

    for epoch in range(start_epoch, args.n_epochs + 1):
        t0 = time.time()
        cur_lr = args.lr * lr_factor(epoch, args.n_epochs, args.decay_epoch)
        for g in opt.param_groups:
            g["lr"] = cur_lr

        tot = 0.0
        n_done = 0
        for i, batch in enumerate(train_loader):
            if args.max_batches and i >= args.max_batches:
                break
            real_A = batch["A"].to(device, non_blocking=True)   # cloudy
            real_B = batch["B"].to(device, non_blocking=True)   # cloud-free
            mask = batch["M"].to(device) if "M" in batch else make_mask(real_A, real_B, args.mask_thresh)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                pred = G(real_A)
            pred = pred.float()

            l1_cloud = masked_l1(pred, real_B, mask)
            l1_clear = masked_l1(pred, real_B, 1 - mask)
            idt_clear = masked_l1(pred, real_A, 1 - mask)
            loss = (args.lambda_cloud * l1_cloud + args.lambda_clear * l1_clear
                    + args.lambda_idt_clear * idt_clear)
            l_edge = l_ssim = torch.tensor(0.0, device=device)
            if args.lambda_edge > 0:
                l_edge = edge_loss(pred, real_B, weight_map=1.0 + args.edge_cloud_boost * mask)
                loss = loss + args.lambda_edge * l_edge
            if args.lambda_ssim > 0:
                l_ssim = ssim_loss(pred, real_B)
                loss = loss + args.lambda_ssim * l_ssim

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                update_ema(ema, G, args.ema)

            tot += loss.item()
            n_done += 1
            if (i + 1) % 50 == 0:
                print(f"[Epoch {epoch}/{args.n_epochs}] [Batch {i+1}/{len(train_loader)}] loss: {loss.item():.4f} "
                      f"| cloud: {l1_cloud.item():.3f} clear: {l1_clear.item():.3f} idt_clear: {idt_clear.item():.3f} "
                      f"| edge: {l_edge.item():.4f} ssim_l: {l_ssim.item():.3f}")

        print(f"== Epoch {epoch} done in {time.time()-t0:.1f}s | avg loss: {tot/max(1,n_done):.4f} | lr: {cur_lr:.6f} ==")

        if have_val and (epoch % args.sample_every == 0 or epoch == args.n_epochs):
            Gv = ema if ema is not None else G
            Gv.eval()
            with torch.no_grad():
                b = next(iter(val_loader))
                a, g_ = b["A"].to(device), b["B"].to(device)
                out = Gv(a)
                n = min(4, a.size(0))
                save_image(torch.cat([denorm(a[:n]), denorm(out[:n]), denorm(g_[:n])], 0),
                           os.path.join(args.sample_dir, f"sample_epoch{epoch}.png"), nrow=n)
            G.train()

        numbered = (epoch % args.checkpoint_interval == 0) or epoch == args.n_epochs
        save_checkpoint(args.output_dir, epoch, G, ema, opt, args, numbered)
        if numbered:
            print(f"Saved checkpoint at epoch {epoch}")

    print("Training complete.")


if __name__ == "__main__":
    main()
"""
CycleGAN + paired cloud-aware losses for cloud removal  --  V2

What is new compared with the original train.py (all new options default to OFF or to the
old behaviour, so an old checkpoint resumes unchanged):

  * LR schedule fixed for resuming: --decay_epoch / --n_epochs are now ABSOLUTE epoch numbers,
    even after --resume (the old scheduler restarted its counter at 0 on every resume).
  * --lambda_edge   Sobel edge-consistency loss        (keeps coastlines / roads / piers)
  * --lambda_ssim   1 - SSIM loss                      (optimises the metric you report)
  * --lambda_perc   VGG16 perceptual loss              (texture realism)
  * --ema           exponential moving average of G_A2B (smoother, usually better final model)
  * --amp 1         mixed precision (T4 tensor cores)  (typically much faster epochs)
  * --strong_aug 1  vertical flips + 90 degree rotations on paired data
  * generator upgrades (new model, can NOT load old weights): --upsample resize, --skip 1,
    --se 1, --residual_out 1
  * checkpoint_latest.pt is written EVERY epoch, atomically (a Colab crash can lose at most
    one epoch, and never corrupts the file); numbered checkpoints still follow --checkpoint_interval
  * --max_batches N  quick smoke test (e.g. --max_batches 5 --n_epochs 1)

Example (continue your current model with the new losses, decaying LR from epoch 100 to 120):

    python train.py --data_root ./data --mask_dir ./data/trainM --image_size 256 \
        --batch_size 4 --num_workers 2 --n_epochs 120 --decay_epoch 100 \
        --output_dir /content/drive/MyDrive/cloud_removal_checkpoints_day2 \
        --sample_dir /content/drive/MyDrive/cloud_removal_samples/day5 \
        --resume /content/drive/MyDrive/cloud_removal_checkpoints_day2 \
        --lambda_identity 10 --lambda_edge 20 --lambda_ssim 5 --ema 0.998 --strong_aug 1 --amp 1
"""

import argparse
import copy
import itertools
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models import build_models
from datasets import CloudDataset, ImagePool
from losses import edge_loss, ssim_loss, VGGPerceptual


def get_args():
    p = argparse.ArgumentParser(description="CycleGAN training for cloud removal (V2)")
    p.add_argument("--data_root", type=str, default="./data", help="root folder containing trainA/trainB/testA/testB")
    p.add_argument("--output_dir", type=str, default="./checkpoints", help="where to save model checkpoints")
    p.add_argument("--sample_dir", type=str, default="./results", help="where to save preview images during training")

    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--channels", type=int, default=3, help="3 for RGB, 13 for full Sentinel-2 multispectral")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--n_epochs", type=int, default=200, help="total number of training epochs (absolute)")
    p.add_argument("--decay_epoch", type=int, default=100, help="LR is full through this epoch, then decays linearly; last epoch runs at about 1/(n_epochs-decay_epoch+1) of the base LR (near, not exactly, zero)")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--beta2", type=float, default=0.999)

    p.add_argument("--ngf", type=int, default=64, help="generator base filters")
    p.add_argument("--ndf", type=int, default=64, help="discriminator base filters")
    p.add_argument("--n_blocks", type=int, default=9, help="number of resnet blocks in generator (use 6 if image_size<=128)")

    # generator upgrades (changing any of these makes the model incompatible with old checkpoints)
    p.add_argument("--upsample", type=str, default="deconv", choices=["deconv", "resize"], help="resize = upsample+conv (no checkerboard)")
    p.add_argument("--skip", type=int, default=0, help="1 = U-Net style encoder->decoder skip connections")
    p.add_argument("--se", type=int, default=0, help="1 = squeeze-excitation attention in residual blocks")
    p.add_argument("--residual_out", type=int, default=0, help="1 = generator predicts a correction added to its input")

    p.add_argument("--lambda_cycle", type=float, default=10.0, help="weight for cycle-consistency loss")
    p.add_argument("--lambda_identity", type=float, default=5.0, help="weight for identity loss (as fraction of lambda_cycle; set 0 to disable)")

    p.add_argument("--paired", type=int, default=1, help="1 = trainA[k]/trainB[k] are the same scene (required for cloud-mask losses)")
    p.add_argument("--lambda_cloud", type=float, default=10.0, help="paired L1 weight inside cloud mask")
    p.add_argument("--lambda_clear", type=float, default=30.0, help="paired L1 weight outside cloud mask (protects saturation)")
    p.add_argument("--lambda_idt_clear", type=float, default=10.0, help="fake_B ~ real_A weight outside cloud mask")
    p.add_argument("--gan_clear_floor", type=float, default=0.05, help="min adversarial weight on non-cloud patches (0 = GAN only on cloud)")
    p.add_argument("--mask_dir", type=str, default="auto", help="folder of dataset masks (same filenames as trainA). 'auto' = data_root/trainM if it exists, 'none' = compute mask from |cloudy-clear|")
    p.add_argument("--mask_thresh", type=float, default=0.1, help="|cloudy - clear| threshold (in [0,1] scale) for cloud mask")

    # new structure-preserving losses (need --paired 1)
    p.add_argument("--lambda_edge", type=float, default=0.0, help="Sobel edge loss weight (try 20)")
    p.add_argument("--edge_cloud_boost", type=float, default=1.0, help="edge loss weight map = 1 + boost * cloud_mask")
    p.add_argument("--lambda_ssim", type=float, default=0.0, help="1-SSIM loss weight (try 5)")
    p.add_argument("--lambda_perc", type=float, default=0.0, help="VGG perceptual loss weight (try 0.5; RGB only, slower)")

    p.add_argument("--ema", type=float, default=0.0, help="EMA decay for G_A2B, e.g. 0.998 (0 = off)")
    p.add_argument("--amp", type=int, default=0, help="1 = mixed precision (CUDA only)")
    p.add_argument("--strong_aug", type=int, default=0, help="1 = also random vflip + rot90 on paired data")
    p.add_argument("--no_strict_pairing", action="store_true", help="only warn (do not stop) if cloudy/clear/mask file names do not match")

    p.add_argument("--pool_size", type=int, default=50, help="size of the image buffer used to update discriminators")
    p.add_argument("--sample_interval", type=int, default=200, help="save a preview grid every N iterations")
    p.add_argument("--checkpoint_interval", type=int, default=10, help="also keep a numbered checkpoint every N epochs")
    p.add_argument("--resume", type=str, default="", help="path to a checkpoint dir (or .pt file) to resume from")
    p.add_argument("--max_batches", type=int, default=0, help="stop each epoch after N batches (smoke test); 0 = full epoch")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


class LambdaLRDecay:
    """
    LR factor for a 1-indexed epoch.
      epochs 1 .. decay_epoch          -> 1.0 (full LR)
      epochs decay_epoch+1 .. n_epochs -> falls linearly, last epoch is 1/(n_epochs-decay_epoch+1)
    Example n_epochs=120, decay_epoch=100: epoch 100 = 1.0, 101 = 0.952, 110 = 0.524, 120 = 0.048.
    """

    def __init__(self, n_epochs, decay_epoch):
        self.n_epochs = n_epochs
        self.decay_epoch = decay_epoch

    def step(self, epoch):
        if epoch <= self.decay_epoch:
            return 1.0
        return max(0.0, 1.0 - (epoch - self.decay_epoch) / float(self.n_epochs - self.decay_epoch + 1))


def make_mask(real_A, real_B, thresh):
    """Cloud mask from the pair: 1 where cloudy differs from clear. Inputs in [-1,1]."""
    diff = (real_A - real_B).abs().mean(1, keepdim=True) / 2.0
    mask = (diff > thresh).float()
    mask = F.max_pool2d(mask, 5, 1, 2)  # dilate a little to cover cloud edges
    return mask.detach()


def masked_l1(pred, target, mask):
    return (mask * (pred - target).abs()).sum() / (mask.sum() * pred.shape[1] + 1e-6)


def denorm(tensor):
    """[-1,1] -> [0,1] for saving previews."""
    return (tensor * 0.5 + 0.5).clamp(0, 1)


def make_scaler(enabled):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


@torch.no_grad()
def update_ema(ema_model, model, decay):
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
    for b_ema, b in zip(ema_model.buffers(), model.buffers()):
        b_ema.copy_(b)


def save_checkpoint(output_dir, epoch, G_A2B, G_B2A, D_A, D_B, opt_G, opt_D, ema, args, numbered):
    """Atomic save: write to a temp file, then rename, so a crash never leaves a corrupt checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    obj = {
        "epoch": epoch,
        "G_A2B": G_A2B.state_dict(),
        "G_B2A": G_B2A.state_dict(),
        "D_A": D_A.state_dict(),
        "D_B": D_B.state_dict(),
        "opt_G": opt_G.state_dict(),
        "opt_D": opt_D.state_dict(),
        "args": vars(args),
    }
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
    if args.amp and not use_amp:
        print("WARNING: --amp 1 needs CUDA; running in full precision.")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)

    arch = dict(upsample=args.upsample, skip=bool(args.skip), se=bool(args.se), residual_out=bool(args.residual_out))
    print(f"Generator arch: {arch}")

    # ---------------- Data ----------------
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
        image_size=args.image_size,
        channels=args.channels,
        mode="train",
        paired=bool(args.paired),
        mask_dir=mask_dir,
        strong_aug=bool(args.strong_aug),
        strict_pairing=not args.no_strict_pairing,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, pin_memory=(device.type == "cuda"),
    )

    test_dir_A = os.path.join(args.data_root, "testA")
    test_dir_B = os.path.join(args.data_root, "testB")
    have_test = os.path.isdir(test_dir_A) and os.path.isdir(test_dir_B) \
        and len(os.listdir(test_dir_A)) > 0 and len(os.listdir(test_dir_B)) > 0
    if have_test:
        test_dataset = CloudDataset(test_dir_A, test_dir_B, image_size=args.image_size, channels=args.channels, mode="test")
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=0)

    print(f"Train pairs per epoch: {len(train_dataset)} | batches/epoch: {len(train_loader)}")

    # ---------------- Models ----------------
    G_A2B, G_B2A, D_A, D_B = build_models(
        in_ch_A=args.channels, in_ch_B=args.channels,
        ngf=args.ngf, ndf=args.ndf, n_blocks=args.n_blocks, device=device, arch=arch,
    )
    ema = None
    if args.ema > 0:
        ema = copy.deepcopy(G_A2B).eval()
        for p_ in ema.parameters():
            p_.requires_grad_(False)

    perceptual = None
    if args.lambda_perc > 0:
        if args.channels != 3:
            print("WARNING: --lambda_perc needs 3-channel images; disabling it.")
        else:
            perceptual = VGGPerceptual().to(device)

    # ---------------- Losses ----------------
    criterion_gan = nn.MSELoss()       # LSGAN
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    # ---------------- Optimizers ----------------
    opt_G = torch.optim.Adam(
        itertools.chain(G_A2B.parameters(), G_B2A.parameters()),
        lr=args.lr, betas=(args.beta1, args.beta2),
    )
    opt_D = torch.optim.Adam(
        itertools.chain(D_A.parameters(), D_B.parameters()),
        lr=args.lr, betas=(args.beta1, args.beta2),
    )
    scaler_G, scaler_D = make_scaler(use_amp), make_scaler(use_amp)
    lr_sched = LambdaLRDecay(args.n_epochs, args.decay_epoch)

    fake_A_pool = ImagePool(args.pool_size)
    fake_B_pool = ImagePool(args.pool_size)

    start_epoch = 1
    if args.resume:
        ckpt_path = args.resume if args.resume.endswith(".pt") else os.path.join(args.resume, "checkpoint_latest.pt")
        if os.path.isfile(ckpt_path):
            print(f"Resuming from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            try:
                G_A2B.load_state_dict(ckpt["G_A2B"])
                G_B2A.load_state_dict(ckpt["G_B2A"])
                D_A.load_state_dict(ckpt["D_A"])
                D_B.load_state_dict(ckpt["D_B"])
            except RuntimeError as e:
                raise SystemExit(
                    "Checkpoint does not match the generator you asked for (--upsample/--skip/--se/--residual_out). "
                    "Use the same options the checkpoint was trained with, or train from scratch without --resume.\n" + str(e)[:400])
            opt_G.load_state_dict(ckpt["opt_G"])
            opt_D.load_state_dict(ckpt["opt_D"])
            if ema is not None:
                ema.load_state_dict(ckpt["G_A2B_ema"] if "G_A2B_ema" in ckpt else ckpt["G_A2B"])
            start_epoch = ckpt["epoch"] + 1
        else:
            print(f"WARNING: resume path {ckpt_path} not found, starting fresh.")

    total_iters = 0
    for epoch in range(start_epoch, args.n_epochs + 1):
        epoch_start = time.time()

        # LR for this epoch (absolute schedule: correct even right after a resume)
        cur_lr = args.lr * lr_sched.step(epoch)
        for opt in (opt_G, opt_D):
            for g in opt.param_groups:
                g["lr"] = cur_lr

        running = {"loss_G": 0.0, "loss_D": 0.0}
        n_done = 0

        for i, batch in enumerate(train_loader):
            if args.max_batches and i >= args.max_batches:
                break
            real_A = batch["A"].to(device, non_blocking=True)  # cloudy
            real_B = batch["B"].to(device, non_blocking=True)  # cloud-free
            mask_cov = 0.0

            # ============ Train Generators ============
            opt_G.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                same_B = same_A = None
                if args.lambda_identity > 0:
                    same_B = G_A2B(real_B)
                    same_A = G_B2A(real_A)
                fake_B = G_A2B(real_A)                  # cloudy -> cloud-free
                pred_fake_B = D_B(fake_B)
                fake_A = G_B2A(real_B)                  # cloud-free -> cloudy
                pred_fake_A = D_A(fake_A)
                recov_A = G_B2A(fake_B)
                recov_B = G_A2B(fake_A)
                loss_perc = perceptual(fake_B, real_B) if (perceptual is not None and args.paired) else None

            fake_B_f = fake_B.float()
            pred_fake_B = pred_fake_B.float()
            pred_fake_A = pred_fake_A.float()

            # Identity loss: G_A2B(B) ~ B, G_B2A(A) ~ A
            loss_id = torch.tensor(0.0, device=device)
            if args.lambda_identity > 0:
                loss_id_B = criterion_identity(same_B.float(), real_B)
                loss_id_A = criterion_identity(same_A.float(), real_A)
                loss_id = (loss_id_A + loss_id_B) * 0.5 * args.lambda_cycle * (args.lambda_identity / 10.0)

            # GAN loss
            valid = torch.ones_like(pred_fake_B, device=device)
            mask = None
            if args.paired:
                mask = batch["M"].to(device) if "M" in batch else make_mask(real_A, real_B, args.mask_thresh)
                mask_cov = mask.mean().item()
                # weight the PatchGAN grid by cloud mask: GAN loss mostly comes from cloud patches
                mask_d = F.adaptive_avg_pool2d(mask, pred_fake_B.shape[-2:])
                w = args.gan_clear_floor + mask_d
                loss_gan_A2B = (w * (pred_fake_B - valid) ** 2).sum() / (w.sum() * pred_fake_B.shape[1])
            else:
                loss_gan_A2B = criterion_gan(pred_fake_B, valid)

            valid_A = torch.ones_like(pred_fake_A, device=device)
            loss_gan_B2A = criterion_gan(pred_fake_A, valid_A)
            loss_gan = loss_gan_A2B + loss_gan_B2A

            # Cycle-consistency loss
            loss_cycle = (criterion_cycle(recov_A.float(), real_A) + criterion_cycle(recov_B.float(), real_B)) * args.lambda_cycle

            # Cloud-aware paired losses (needs --paired 1)
            zero = torch.tensor(0.0, device=device)
            loss_paired = l1_cloud = l1_clear = idt_clear = zero
            loss_edge = loss_ssim = zero
            loss_extra = zero
            if args.paired:
                l1_cloud = masked_l1(fake_B_f, real_B, mask)
                l1_clear = masked_l1(fake_B_f, real_B, 1 - mask)
                idt_clear = masked_l1(fake_B_f, real_A, 1 - mask)
                loss_paired = (args.lambda_cloud * l1_cloud
                               + args.lambda_clear * l1_clear
                               + args.lambda_idt_clear * idt_clear)

                if args.lambda_edge > 0:
                    loss_edge = edge_loss(fake_B_f, real_B, weight_map=1.0 + args.edge_cloud_boost * mask)
                    loss_extra = loss_extra + args.lambda_edge * loss_edge
                if args.lambda_ssim > 0:
                    loss_ssim = ssim_loss(fake_B_f, real_B)
                    loss_extra = loss_extra + args.lambda_ssim * loss_ssim
                if loss_perc is not None:
                    loss_extra = loss_extra + args.lambda_perc * loss_perc.float()

            loss_G = loss_gan + loss_cycle + loss_id + loss_paired + loss_extra
            scaler_G.scale(loss_G).backward()
            scaler_G.step(opt_G)
            scaler_G.update()
            if ema is not None:
                update_ema(ema, G_A2B, args.ema)

            # ============ Train Discriminators ============
            opt_D.zero_grad(set_to_none=True)
            fake_A_ = fake_A_pool.query(fake_A.detach().float())
            fake_B_ = fake_B_pool.query(fake_B.detach().float())
            with torch.autocast(device_type=device.type, enabled=use_amp):
                pred_real_A = D_A(real_A)
                pred_fake_A_ = D_A(fake_A_)
                pred_real_B = D_B(real_B)
                pred_fake_B_ = D_B(fake_B_)
            pred_real_A, pred_fake_A_ = pred_real_A.float(), pred_fake_A_.float()
            pred_real_B, pred_fake_B_ = pred_real_B.float(), pred_fake_B_.float()

            loss_D_A = (criterion_gan(pred_real_A, torch.ones_like(pred_real_A))
                        + criterion_gan(pred_fake_A_, torch.zeros_like(pred_fake_A_))) * 0.5
            loss_D_B = (criterion_gan(pred_real_B, torch.ones_like(pred_real_B))
                        + criterion_gan(pred_fake_B_, torch.zeros_like(pred_fake_B_))) * 0.5
            loss_D = loss_D_A + loss_D_B
            scaler_D.scale(loss_D).backward()
            scaler_D.step(opt_D)
            scaler_D.update()

            running["loss_G"] += loss_G.item()
            running["loss_D"] += loss_D.item()
            n_done += 1
            total_iters += 1

            if total_iters % 50 == 0:
                extra_txt = ""
                if args.lambda_edge > 0 or args.lambda_ssim > 0 or loss_perc is not None:
                    perc_v = loss_perc.item() if loss_perc is not None else 0.0
                    extra_txt = f" | edge: {loss_edge.item():.4f} ssim_l: {loss_ssim.item():.3f} perc: {perc_v:.3f}"
                print(
                    f"[Epoch {epoch}/{args.n_epochs}] [Batch {i+1}/{len(train_loader)}] "
                    f"loss_D: {loss_D.item():.4f} | loss_G: {loss_G.item():.4f} "
                    f"(gan: {loss_gan.item():.4f}, cycle: {loss_cycle.item():.4f}, id: {loss_id.item():.4f}) "
                    f"| cloud: {l1_cloud.item():.3f} clear: {l1_clear.item():.3f} idt_clear: {idt_clear.item():.3f} mask_cov: {mask_cov:.2f}"
                    + extra_txt
                )

            if have_test and total_iters % args.sample_interval == 0:
                save_sample_grid(ema if ema is not None else G_A2B, G_B2A, test_loader, device, args, total_iters)

        n_batches = max(1, n_done)
        print(
            f"== Epoch {epoch} done in {time.time()-epoch_start:.1f}s | "
            f"avg loss_D: {running['loss_D']/n_batches:.4f} | avg loss_G: {running['loss_G']/n_batches:.4f} "
            f"| lr: {cur_lr:.6f} =="
        )

        numbered = (epoch % args.checkpoint_interval == 0) or epoch == args.n_epochs
        save_checkpoint(args.output_dir, epoch, G_A2B, G_B2A, D_A, D_B, opt_G, opt_D, ema, args, numbered)
        if numbered:
            print(f"Saved checkpoint at epoch {epoch}")

    print("Training complete.")


def save_sample_grid(G_vis, G_B2A, test_loader, device, args, step):
    """Rows: cloudy input | prediction | ground truth | clear->cloudy (helper generator)."""
    G_vis.eval()
    G_B2A.eval()
    batch = next(iter(test_loader))
    with torch.no_grad():
        real_A = batch["A"].to(device)
        real_B = batch["B"].to(device)
        fake_B = G_vis(real_A)
        fake_A = G_B2A(real_B)
        n = min(4, real_A.size(0))
        grid = torch.cat(
            [denorm(real_A[:n]), denorm(fake_B[:n]), denorm(real_B[:n]), denorm(fake_A[:n])], dim=0
        )
        save_image(grid, os.path.join(args.sample_dir, f"sample_iter{step}.png"), nrow=n)
    G_B2A.train()
    G_vis.train()   # no layer here behaves differently in train/eval (InstanceNorm, no dropout)


if __name__ == "__main__":
    main()

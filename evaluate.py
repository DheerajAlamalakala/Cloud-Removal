"""
Evaluate cloud-removal checkpoints with RMSE, SAM, PSNR, SSIM  --  V2

New compared with the first evaluate.py:
  * BASELINE row: the raw cloudy input scored against the ground truth. If the model's cloud-region
    numbers are not clearly better than this baseline, the model is not helping.
  * cloud-region-only metrics (needs masks) next to whole-image metrics
  * several checkpoints in one run  ->  checkpoint_metrics.csv (one row per checkpoint)
  * --use_ema   evaluate the EMA weights if the checkpoint has them
  * --tta       flip test-time augmentation (average of 3 predictions)
  * per-image CSV sorted worst-first, plus side-by-side images of the worst images
  * the generator architecture is read from the checkpoint, so V1 and V2 models both work

Example:
!python evaluate.py \
    --checkpoint /content/drive/MyDrive/cloud_removal_checkpoints_day2/checkpoint_epoch100.pt \
                 /content/drive/MyDrive/cloud_removal_checkpoints_day2/checkpoint_latest.pt \
    --cloudy_dir ./data/testA --clear_dir ./data/testB --mask_dir ./data/testM \
    --save_dir /content/drive/MyDrive/cloud_removal_eval_v2 --use_ema

--mask_dir: <folder> (non-black pixel = cloud, sorted order like training) | diff | none
"""
import argparse
import csv
import os

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import maximum_filter
from skimage.metrics import structural_similarity

from datasets import list_images
from models import make_generator


# ---------------------------- metrics ([0,1] HxWxC arrays, mask HxW bool) ----------------------------
def rmse(p, g, m=None):
    se = (p - g) ** 2
    return float(np.sqrt((se[m] if m is not None else se).mean()))


def psnr(p, g, m=None):
    se = (p - g) ** 2
    mse = (se[m] if m is not None else se).mean()
    return float(10 * np.log10(1.0 / max(mse, 1e-12)))


def sam(p, g, m=None, eps=1e-8):
    """Spectral Angle Mapper in degrees (mean over pixels)."""
    p, g = p.reshape(-1, p.shape[-1]), g.reshape(-1, g.shape[-1])
    if m is not None:
        mm = m.reshape(-1)
        p, g = p[mm], g[mm]
    cos = (p * g).sum(1) / (np.linalg.norm(p, axis=1) * np.linalg.norm(g, axis=1) + eps)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))).mean())


def ssim(p, g, m=None):
    score, smap = structural_similarity(g, p, data_range=1.0, channel_axis=-1, full=True)
    return float(smap[m].mean()) if m is not None else float(score)


def all_metrics(p, g, m=None):
    return dict(RMSE=rmse(p, g, m), SAM=sam(p, g, m), PSNR=psnr(p, g, m), SSIM=ssim(p, g, m))


# ---------------------------- data helpers (match datasets.py test mode) ----------------------------
def load_rgb(path, size):
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def load_mask(path, size):
    img = Image.open(path).convert("RGB").resize((size, size), Image.NEAREST)
    return np.asarray(img).max(axis=2) > 0          # any non-black pixel = cloud


def mask_from_diff(cloudy, clear, thresh):
    diff = np.abs(cloudy - clear).mean(axis=2)
    return maximum_filter((diff > thresh).astype(np.uint8), size=5) > 0


# ---------------------------- model loading ----------------------------
def load_generator(path, device, use_ema, fallback):
    ckpt = torch.load(path, map_location=device)
    a = ckpt.get("args", {}) or {}
    ch = a.get("channels", fallback.channels)
    arch = dict(
        upsample=a.get("upsample", fallback.upsample),
        skip=bool(a.get("skip", fallback.skip)),
        se=bool(a.get("se", fallback.se)),
        residual_out=bool(a.get("residual_out", fallback.residual_out)),
    )
    G = make_generator(ch, ch, ngf=a.get("ngf", fallback.ngf), n_blocks=a.get("n_blocks", fallback.n_blocks), arch=arch).to(device)
    key = "G_A2B_ema" if (use_ema and "G_A2B_ema" in ckpt) else "G_A2B"
    if use_ema and key != "G_A2B_ema":
        print(f"  (no EMA weights in {os.path.basename(path)}, using raw G_A2B)")
    G.load_state_dict(ckpt[key])
    G.eval()
    return G, ckpt.get("epoch", "?"), key


@torch.no_grad()
def predict(G, cloudy, device, tta):
    x = torch.from_numpy(cloudy).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
    y = G(x)
    if tta:
        yh = torch.flip(G(torch.flip(x, dims=[3])), dims=[3])
        yv = torch.flip(G(torch.flip(x, dims=[2])), dims=[2])
        y = (y + yh + yv) / 3.0
    y = (y.clamp(-1, 1) + 1) / 2
    return y.squeeze(0).permute(1, 2, 0).cpu().numpy()


def summarize(rows, region):
    out = {}
    for m in ("RMSE", "SAM", "PSNR", "SSIM"):
        v = [r[f"{m}_{region}"] for r in rows if f"{m}_{region}" in r]
        out[m] = float(np.mean(v)) if v else float("nan")
        out["n"] = len(v)
    return out


def save_strip(path, cloudy, pred, clear):
    strip = np.concatenate([cloudy, pred, clear], axis=1)       # input | output | ground truth
    Image.fromarray((strip * 255).round().astype(np.uint8)).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", nargs="+", required=True, help="one or more .pt files")
    ap.add_argument("--cloudy_dir", required=True)
    ap.add_argument("--clear_dir", required=True)
    ap.add_argument("--mask_dir", default="none", help="folder | diff | none")
    ap.add_argument("--mask_thresh", type=float, default=0.1)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--use_ema", action="store_true")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--save_dir", default="./eval_out")
    ap.add_argument("--n_save", type=int, default=4, help="first N images saved as comparisons")
    ap.add_argument("--worst_n", type=int, default=6, help="worst N images (lowest PSNR) saved as comparisons")
    ap.add_argument("--max_images", type=int, default=0, help="0 = all")
    ap.add_argument("--no_strict_pairing", action="store_true", help="only warn if cloudy/clear/mask file names do not match")
    # only used if the checkpoint has no saved args (old checkpoints)
    ap.add_argument("--channels", type=int, default=3)
    ap.add_argument("--ngf", type=int, default=64)
    ap.add_argument("--n_blocks", type=int, default=9)
    ap.add_argument("--upsample", default="deconv")
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--se", type=int, default=0)
    ap.add_argument("--residual_out", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    files_A, files_B = list_images(args.cloudy_dir), list_images(args.clear_dir)
    assert len(files_A) == len(files_B) and len(files_A) > 0, \
        f"cloudy ({len(files_A)}) and clear ({len(files_B)}) counts must match"
    files_M = []
    if args.mask_dir not in ("none", "diff", ""):
        files_M = list_images(args.mask_dir)
        assert len(files_M) == len(files_A), f"mask count {len(files_M)} != {len(files_A)}"
    stem = lambda p: os.path.splitext(os.path.basename(p))[0]
    problems = [f"{len([1 for a, b in zip(files_A, files_B) if stem(a) != stem(b)])} cloudy/clear filename mismatches"
                ] if any(stem(a) != stem(b) for a, b in zip(files_A, files_B)) else []
    if files_M and any(stem(a) != stem(m) for a, m in zip(files_A, files_M)):
        problems.append(f"{len([1 for a, m in zip(files_A, files_M) if stem(a) != stem(m)])} cloudy/mask filename mismatches")
    if problems:
        msg = "Pairing check failed: " + "; ".join(problems) + " (metrics would compare different scenes)"
        if not args.no_strict_pairing:
            raise SystemExit(msg + ". Fix the file names or pass --no_strict_pairing to only warn.")
        print("WARNING: " + msg)
    n = len(files_A) if not args.max_images else min(args.max_images, len(files_A))

    # load everything once (test sets are small)
    cloudy_l, clear_l, mask_l = [], [], []
    for i in range(n):
        c, g = load_rgb(files_A[i], args.image_size), load_rgb(files_B[i], args.image_size)
        m = None
        if files_M:
            m = load_mask(files_M[i], args.image_size)
        elif args.mask_dir == "diff":
            m = mask_from_diff(c, g, args.mask_thresh)
        if m is not None and m.sum() == 0:
            m = None
        cloudy_l.append(c); clear_l.append(g); mask_l.append(m)
    print(f"Loaded {n} pairs (paired by sorted order, as in training); masks: {args.mask_dir}")

    table = []   # rows for checkpoint_metrics.csv

    def add_table_row(name, rows):
        row = {"checkpoint": name}
        for region in ("full", "cloud"):
            s = summarize(rows, region)
            if s["n"]:
                for m in ("PSNR", "RMSE", "SAM", "SSIM"):
                    row[f"{m}_{region}"] = round(s[m], 4)
        table.append(row)

    # ---------------- baseline: cloudy input vs ground truth ----------------
    base_rows = []
    for c, g, m in zip(cloudy_l, clear_l, mask_l):
        r = {f"{k}_full": v for k, v in all_metrics(c, g).items()}
        if m is not None:
            r.update({f"{k}_cloud": v for k, v in all_metrics(c, g, m).items()})
        base_rows.append(r)
    add_table_row("BASELINE_cloudy_input", base_rows)

    # ---------------- each checkpoint ----------------
    for path in args.checkpoint:
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"\nEvaluating {name} ...")
        G, epoch, key = load_generator(path, device, args.use_ema, args)
        tag = name + ("_ema" if key == "G_A2B_ema" else "") + ("_tta" if args.tta else "")
        out_dir = os.path.join(args.save_dir, tag)
        os.makedirs(out_dir, exist_ok=True)

        rows, preds = [], []
        for i in range(n):
            pred = predict(G, cloudy_l[i], device, args.tta)
            preds.append(pred)
            row = {"image": os.path.basename(files_A[i])}
            row.update({f"{k}_full": v for k, v in all_metrics(pred, clear_l[i]).items()})
            if mask_l[i] is not None:
                row.update({f"{k}_cloud": v for k, v in all_metrics(pred, clear_l[i], mask_l[i]).items()})
                row["mask_cov"] = float(mask_l[i].mean())
            rows.append(row)

        add_table_row(tag, rows)

        sort_key = "PSNR_cloud" if any("PSNR_cloud" in r for r in rows) else "PSNR_full"
        order = sorted(range(n), key=lambda i: rows[i].get(sort_key, rows[i]["PSNR_full"]))
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "image", k))
        with open(os.path.join(out_dir, "metrics_per_image.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows([rows[i] for i in order])          # worst first

        for rank, i in enumerate(order[: args.worst_n]):
            save_strip(os.path.join(out_dir, f"worst{rank+1}_{os.path.splitext(rows[i]['image'])[0]}.png"),
                       cloudy_l[i], preds[i], clear_l[i])
        for i in range(min(args.n_save, n)):
            save_strip(os.path.join(out_dir, f"first{i+1}_{os.path.splitext(rows[i]['image'])[0]}.png"),
                       cloudy_l[i], preds[i], clear_l[i])

    # ---------------- summary ----------------
    cols = ["checkpoint"] + sorted({k for r in table for k in r if k != "checkpoint"})
    with open(os.path.join(args.save_dir, "checkpoint_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(table)

    print("\n=========== SUMMARY (mean over test images) ===========")
    for region, label in (("full", "WHOLE IMAGE"), ("cloud", "CLOUD REGION ONLY")):
        if not any(f"PSNR_{region}" in r for r in table):
            continue
        print(f"\n{label}")
        print(f"{'checkpoint':<34}{'PSNR':>8}{'RMSE':>9}{'SAM':>8}{'SSIM':>8}")
        for r in table:
            if f"PSNR_{region}" in r:
                print(f"{r['checkpoint']:<34}{r[f'PSNR_{region}']:>8.2f}{r[f'RMSE_{region}']:>9.4f}"
                      f"{r[f'SAM_{region}']:>8.2f}{r[f'SSIM_{region}']:>8.4f}")
    print("\nRead it like this: a checkpoint must beat the BASELINE row (raw cloudy input) to be helping,")
    print("and the CLOUD REGION table is the honest one (clear land inflates whole-image scores).")
    print(f"\nSaved checkpoint_metrics.csv, per-image CSVs (worst first) and comparison images to {args.save_dir}")


if __name__ == "__main__":
    main()

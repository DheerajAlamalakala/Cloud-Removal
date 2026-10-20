"""
Run cloud removal on new images using a trained G_A2B generator.

Usage:
    python inference.py \
        --checkpoint ./checkpoints/checkpoint_latest.pt \
        --input_dir ./data/testA \
        --output_dir ./results/derained \
        --image_size 256 --channels 3
"""

import argparse
import os

import torch
from PIL import Image
from torchvision.utils import save_image
import torchvision.transforms as T

from models import make_generator
from datasets import list_images


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input_dir", required=True, help="folder of cloudy images")
    p.add_argument("--output_dir", default="./results/cloud_removed")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--channels", type=int, default=3)
    p.add_argument("--ngf", type=int, default=64)
    p.add_argument("--n_blocks", type=int, default=9)
    p.add_argument("--use_ema", action="store_true", help="use EMA generator weights if the checkpoint has them")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device)
    a = ckpt.get("args", {}) or {}          # architecture the checkpoint was trained with
    ch = a.get("channels", args.channels)
    arch = dict(upsample=a.get("upsample", "deconv"), skip=bool(a.get("skip", 0)),
                se=bool(a.get("se", 0)), residual_out=bool(a.get("residual_out", 0)))
    G_A2B = make_generator(ch, ch, ngf=a.get("ngf", args.ngf), n_blocks=a.get("n_blocks", args.n_blocks), arch=arch).to(device)
    key = "G_A2B_ema" if (args.use_ema and "G_A2B_ema" in ckpt) else "G_A2B"
    G_A2B.load_state_dict(ckpt[key])
    G_A2B.eval()
    print(f"Loaded {key} (epoch {ckpt.get('epoch', '?')}), arch={arch}")

    transform = T.Compose([
        T.Resize((args.image_size, args.image_size), Image.BICUBIC),
        T.ToTensor(),
        T.Normalize([0.5] * args.channels, [0.5] * args.channels),
    ])

    files = list_images(args.input_dir)
    print(f"Found {len(files)} images in {args.input_dir}")

    with torch.no_grad():
        for path in files:
            img = Image.open(path).convert("RGB") if args.channels == 3 else Image.open(path)
            x = transform(img).unsqueeze(0).to(device)
            fake_clear = G_A2B(x)
            out = (fake_clear * 0.5 + 0.5).clamp(0, 1)
            name = os.path.splitext(os.path.basename(path))[0]
            save_image(out, os.path.join(args.output_dir, f"{name}_declouded.png"))

    print(f"Saved cloud-removed outputs to {args.output_dir}")


if __name__ == "__main__":
    main()

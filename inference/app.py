"""SEN12MS-CR cloud-removal demo backend (FastAPI).

Start it with inference\\run_demo.ps1 (which sets DEMO_DATA, DEMO_MANIFEST, DEMO_CKPT, DEMO_CSV).
It never edits your project files: it imports the SEN12MS-CR package under src/.
"""
from __future__ import annotations

import csv
import io
import math
import os
import random
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sen12mscr.dataset import Sen12msCR, read_manifest  # noqa: E402
from sen12mscr.losses import cloud_mask, per_image_metrics  # noqa: E402
from sen12mscr.model import ResUNet  # noqa: E402

RGB = [3, 2, 1]  # Sentinel-2 B4, B3, B2


def need(var: str) -> str:
    value = os.environ.get(var, "")
    if not value:
        raise RuntimeError(f"{var} is not set. Start the demo with inference\\run_demo.ps1 so paths.ps1 sets it.")
    return value


DATA_ROOT = need("DEMO_DATA")
MANIFEST = need("DEMO_MANIFEST")
CKPT = need("DEMO_CKPT")
CSV_PATH = os.environ.get("DEMO_CSV", "")
SPLIT = os.environ.get("DEMO_SPLIT", "test")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(CKPT, map_location=device, weights_only=False)
saved_args = ckpt.get("args", {}) or {}
USE_SAR = not saved_args.get("no_sar", False)
model = ResUNet(base=int(saved_args.get("base", 48)), use_sar=USE_SAR).to(device).eval()
weights = ckpt.get("ema", ckpt.get("net"))
if weights is None:
    raise RuntimeError("Checkpoint has neither 'ema' nor 'net' weights")
model.load_state_dict(weights)
CKPT_EPOCH = int(ckpt.get("epoch", -1))

items = read_manifest(MANIFEST)["splits"][SPLIT]
by_name = {it["name"]: i for i, it in enumerate(items)}
dataset = Sen12msCR(DATA_ROOT, items, train=False, use_sar=USE_SAR)

# ---- sample list: stratified-random by pseudo-cloud coverage (no cherry-picking) ----
def build_samples(per_bin: int = 8):
    rng = random.Random(7)
    coverage = {}
    if CSV_PATH and os.path.isfile(CSV_PATH):
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    coverage[row["image"]] = float(row["pseudo_cloud_coverage"])
                except (KeyError, ValueError):
                    pass
    names = [it["name"] for it in items]
    if not coverage:
        return [{"name": n, "coverage": None} for n in rng.sample(names, min(24, len(names)))]
    bins = {"low": [], "mid": [], "high": []}
    for n in names:
        c = coverage.get(n)
        if c is None or c < 0.01:
            continue
        bins["low" if c < 0.10 else "mid" if c < 0.50 else "high"].append(n)
    out = []
    for key in ("low", "mid", "high"):
        pool = bins[key]
        for n in rng.sample(pool, min(per_bin, len(pool))):
            out.append({"name": n, "coverage": coverage[n]})
    return out


SAMPLES = build_samples()

# ---- inference with a small cache ----
_cache: "OrderedDict[str, dict]" = OrderedDict()


def _clean(x):
    x = float(x)
    return None if math.isnan(x) or math.isinf(x) else x


@torch.no_grad()
def run(name: str) -> dict:
    if name in _cache:
        _cache.move_to_end(name)
        return _cache[name]
    if name not in by_name:
        raise HTTPException(404, f"Unknown sample {name}")
    s = dataset[by_name[name]]
    cloudy = s["cloudy"][None].to(device).clamp(0, 1)
    clear = s["clear"][None].to(device).clamp(0, 1)
    sar = s["s1"][None].to(device) if USE_SAR else None
    pred = model(cloudy, sar).float().clamp(0, 1)
    mask = cloud_mask(cloudy, clear)
    m_model = per_image_metrics(pred, clear, mask)
    m_input = per_image_metrics(cloudy, clear, mask)
    keys = ["PSNR_full", "PSNR_cloud", "SSIM_full", "SSIM_cloud", "SAM_full", "SAM_cloud", "RMSE_full", "RMSE_cloud"]
    result = {
        "arrays": {
            "cloudy": cloudy[0].cpu().numpy(),
            "pred": pred[0].cpu().numpy(),
            "clear": clear[0].cpu().numpy(),
            "mask": mask[0, 0].cpu().numpy(),
        },
        "metrics": {
            "input": {k: _clean(m_input[k][0]) for k in keys},
            "model": {k: _clean(m_model[k][0]) for k in keys},
        },
        "coverage": _clean(mask.mean()),
        "item": items[by_name[name]],
    }
    _cache[name] = result
    if len(_cache) > 24:
        _cache.popitem(last=False)
    return result


def to_png(arr: np.ndarray, kind: str) -> bytes:
    if kind == "mask":
        img = (arr > 0.5).astype(np.uint8) * 255
        image = Image.fromarray(img, mode="L")
    else:
        linear = np.clip(arr[RGB].transpose(1, 2, 0) * 2.5, 0, 1)
        # Apply the standard sRGB transfer curve after a fixed exposure boost.
        rgb = np.where(linear <= 0.0031308, 12.92 * linear,
                       1.055 * np.power(linear, 1 / 2.4) - 0.055)
        image = Image.fromarray((rgb * 255).round().astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


app = FastAPI(title="Cloud removal demo")


@app.get("/api/info")
def info():
    return {
        "device": str(device),
        "checkpoint": Path(CKPT).parent.name + "/" + Path(CKPT).name,
        "epoch": CKPT_EPOCH,
        "split": SPLIT,
        "n_samples": len(items),
        "uses_sar": USE_SAR,
    }


@app.get("/api/samples")
def samples():
    return SAMPLES


@app.get("/api/predict/{name}")
def predict(name: str):
    r = run(name)
    return {"name": name, "coverage": r["coverage"], "metrics": r["metrics"],
            "roi": r["item"]["roi"], "scene": r["item"]["scene"]}


@app.get("/api/image/{name}/{kind}.png")
def image(name: str, kind: str):
    if kind not in ("cloudy", "pred", "clear", "mask"):
        raise HTTPException(404, "kind must be cloudy, pred, clear or mask")
    return Response(to_png(run(name)["arrays"][kind], kind), media_type="image/png")


@app.get("/api/geotiff/{name}")
def geotiff(name: str):
    import rasterio
    r = run(name)
    with rasterio.open(os.path.join(DATA_ROOT, r["item"]["cloudy"])) as src:
        profile = src.profile.copy()
    for key in ("photometric", "blockxsize", "blockysize", "tiled", "interleave"):
        profile.pop(key, None)
    profile.update(driver="GTiff", count=13, dtype="uint16", nodata=None, compress="deflate")
    data = (r["arrays"]["pred"] * 10000.0).round().clip(0, 10000).astype(np.uint16)
    with rasterio.MemoryFile() as mem:
        with mem.open(**profile) as dst:
            dst.write(data)
        payload = mem.read()
    fname = Path(name).stem + "_cloud_removed.tif"
    return Response(payload, media_type="image/tiff",
                    headers={"Content-Disposition": f'attachment; filename="{fname}"'})


STATIC = Path(__file__).resolve().parent / "static"


@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

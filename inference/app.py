"""SEN12MS-CR cloud-removal demo backend (FastAPI).

Start it with inference\\run_demo.ps1 (which sets the data, manifest, checkpoint and format paths).
It supports both the GeoTIFF reader and the lazy Parquet reader.
"""
from __future__ import annotations

import csv
import io
import math
import os
import random
import sys
from collections import OrderedDict
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sen12mscr.dataset import Sen12msCR, read_manifest  # noqa: E402
from sen12mscr.losses import cloud_mask, per_image_metrics  # noqa: E402
from sen12mscr.model import ResUNet  # noqa: E402
from sen12mscr.parquet_dataset import Sen12msCRParquet  # noqa: E402

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
DATA_FORMAT = os.environ.get("DEMO_DATA_FORMAT", "geotiff").lower()
if DATA_FORMAT not in ("geotiff", "parquet"):
    raise RuntimeError("DEMO_DATA_FORMAT must be 'geotiff' or 'parquet'")

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
dataset = None
if DATA_FORMAT == "parquet":
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Parquet demo requires pyarrow in the project Python environment") from exc

    # Build a small name -> row-group/row index using only the patch-ID column.
    # Pixel arrays are decoded lazily only for the selected demo sample.
    by_name = {}
    parquet_files = {}
    for group in items:
        rel_path = Path(DATA_ROOT).joinpath(*PurePosixPath(group["file"]).parts)
        key = str(rel_path)
        parquet_file = parquet_files.get(key)
        if parquet_file is None:
            parquet_file = pq.ParquetFile(key)
            parquet_files[key] = parquet_file
        row_index = 0
        for record_batch in parquet_file.iter_batches(
            batch_size=1024,
            row_groups=[int(group["row_group"])],
            columns=["patch"],
            use_threads=False,
        ):
            for patch in record_batch.column(0).to_pylist():
                name = f"{group['season']}_scene_{group['scene']}_{patch}"
                by_name[name] = (group, row_index)
                row_index += 1
        if row_index != int(group["rows"]):
            raise RuntimeError(
                f"Could not index {group['file']} row group {group['row_group']}: "
                f"found {row_index} patches, expected {group['rows']}"
            )
else:
    by_name = {it["name"]: i for i, it in enumerate(items)}
    dataset = Sen12msCR(DATA_ROOT, items, train=False, use_sar=USE_SAR)
season_by_name = {
    name: (by_name[name][0]["season"] if DATA_FORMAT == "parquet" else items[by_name[name]]["roi"])
    for name in by_name
}

# ---- sample list: stratified-random by pseudo-cloud coverage (no cherry-picking) ----
def build_samples(per_cell: int = 2):
    rng = random.Random(7)
    coverage = {}
    if CSV_PATH and os.path.isfile(CSV_PATH):
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    coverage[row["image"]] = float(row["pseudo_cloud_coverage"])
                except (KeyError, ValueError):
                    pass
    names = list(by_name) if DATA_FORMAT == "parquet" else [it["name"] for it in items]
    if not coverage:
        names_by_season = {}
        for name in names:
            names_by_season.setdefault(season_by_name.get(name, "unknown"), []).append(name)
        out = []
        quota = max(1, math.ceil(24 / max(1, len(names_by_season))))
        for season, season_names_list in sorted(names_by_season.items()):
            for name in rng.sample(season_names_list, min(quota, len(season_names_list))):
                out.append({"name": name, "coverage": None, "season": season})
        return out[:24]
    bins = {}
    for n in names:
        c = coverage.get(n)
        if c is None:
            continue
        season = season_by_name.get(n, "unknown")
        cloud_bin = "low" if c < 0.10 else "mid" if c < 0.50 else "high"
        bins.setdefault((season, cloud_bin), []).append(n)
    out = []
    for season in sorted({key[0] for key in bins}):
        for cloud_bin in ("low", "mid", "high"):
            pool = bins.get((season, cloud_bin), [])
            for name in rng.sample(pool, min(per_cell, len(pool))):
                out.append({"name": name, "coverage": coverage[name], "season": season})
    if len(out) < 24:
        chosen = {sample["name"] for sample in out}
        remaining = [name for name in names if name not in chosen and name in coverage]
        for name in rng.sample(remaining, min(24 - len(out), len(remaining))):
            out.append({"name": name, "coverage": coverage[name], "season": season_by_name.get(name, "unknown")})
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
    if DATA_FORMAT == "parquet":
        group, target_row = by_name[name]
        stream = Sen12msCRParquet(
            DATA_ROOT, [group], train=False, use_sar=USE_SAR, batch_size=8,
        )
        s = None
        rows_seen = 0
        for batch in stream:
            batch_names = batch["name"]
            if name in batch_names:
                row = batch_names.index(name)
                s = {
                    "cloudy": batch["cloudy"][row],
                    "clear": batch["clear"][row],
                    "name": name,
                    "roi": batch["roi"][row],
                    "scene": batch["scene"][row],
                    "season": batch["season"][row],
                    "patch": batch["patch"][row],
                }
                if USE_SAR:
                    s["s1"] = batch["s1"][row]
                break
            rows_seen += len(batch_names)
        if s is None:
            raise HTTPException(404, f"Could not decode patch row {target_row} for {name}")
    else:
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
        # GeoTIFF manifests are per-sample lists; Parquet manifests are
        # row-group lists, so use the decoded sample metadata for Parquet.
        "item": s if DATA_FORMAT == "parquet" else items[by_name[name]],
    }
    _cache[name] = result
    if len(_cache) > 24:
        _cache.popitem(last=False)
    return result


def to_png(arr: np.ndarray, kind: str, display: str = "raw") -> bytes:
    if display not in ("raw", "enhanced"):
        raise HTTPException(400, "display must be 'raw' or 'enhanced'")
    if kind == "mask":
        img = (arr > 0.5).astype(np.uint8) * 255
        image = Image.fromarray(img, mode="L")
    else:
        linear = np.clip(arr[RGB].transpose(1, 2, 0) * 2.5, 0, 1)
        # Apply the standard sRGB transfer curve after a fixed exposure boost.
        rgb = np.where(linear <= 0.0031308, 12.92 * linear,
                       1.055 * np.power(linear, 1 / 2.4) - 0.055)
        if display == "enhanced":
            # One fixed transform is applied to cloudy, prediction and target.
            # This is only for viewing; the model arrays/metrics stay untouched.
            rgb = np.power(np.clip((rgb - 0.5) * 1.10 + 0.5, 0, 1), 0.94)
        image = Image.fromarray((rgb * 255).round().astype(np.uint8), mode="RGB")
        if display == "enhanced":
            image = image.filter(ImageFilter.UnsharpMask(radius=0.8, percent=45, threshold=3))
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
        "n_samples": len(by_name),
        "uses_sar": USE_SAR,
        "data_format": DATA_FORMAT,
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
def image(name: str, kind: str, display: str = "raw"):
    if kind not in ("cloudy", "pred", "clear", "mask"):
        raise HTTPException(404, "kind must be cloudy, pred, clear or mask")
    return Response(to_png(run(name)["arrays"][kind], kind, display), media_type="image/png")


@app.get("/api/geotiff/{name}")
def geotiff(name: str):
    if DATA_FORMAT == "parquet":
        raise HTTPException(404, "Parquet patches have no georeferencing; download the reflectance NPZ instead")
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


@app.get("/api/prediction/{name}.npz")
def prediction_npz(name: str):
    if DATA_FORMAT != "parquet":
        raise HTTPException(404, "NPZ prediction export is enabled for Parquet patches")
    prediction = run(name)["arrays"]["pred"]
    payload = io.BytesIO()
    np.savez_compressed(
        payload,
        prediction=(prediction * 10000.0).round().clip(0, 10000).astype(np.uint16),
        band_names=np.asarray(["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]),
        reflectance_scale=np.uint16(10000),
    )
    fname = Path(name).stem + "_cloud_removed.npz"
    return Response(payload.getvalue(), media_type="application/octet-stream",
                    headers={"Content-Disposition": f'attachment; filename="{fname}"'})


STATIC = Path(__file__).resolve().parent / "static"


@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

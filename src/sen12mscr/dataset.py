"""SEN12MS-CR loader for paired cloudy S2 + clear S2 + optional S1 SAR GeoTIFFs.

Expected TUM layout for a collection/season name such as ROIs1158_spring:
  ROOT/ROIs1158_spring_s1/s1_1/<matching S1 tif>
  ROOT/ROIs1158_spring_s2/s2_1/<matching clear S2 tif>
  ROOT/ROIs1158_spring_s2_cloudy/s2_cloudy_1/<matching cloudy S2 tif>

The manifest stores paths relative to ROOT, so it can be reused on a laptop and
in Colab as long as the same dataset contents and ROI selection are present.
"""
from __future__ import annotations

import glob
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .official_splits import official_scene_split

RGB = [3, 2, 1]  # Sentinel-2 B4, B3, B2 in the official B1..B12/B8A order.
MANIFEST_VERSION = 2


def _relative(path: str, root: str) -> str:
    return os.path.relpath(os.path.abspath(path), os.path.abspath(root)).replace("\\", "/")


def scan(root: str, rois: list[str], include_sar: bool = True) -> list[dict[str, Any]]:
    """Find triplets by matching ROI, scene folder and patch filename."""
    root = os.path.abspath(root)
    items: list[dict[str, Any]] = []
    missing_clear = 0
    missing_sar = 0

    for roi in rois:
        pattern = os.path.join(root, f"{roi}_s2_cloudy", "s2_cloudy_*", "*.tif")
        for cloudy_path in sorted(glob.glob(pattern)):
            name = os.path.basename(cloudy_path)
            scene_dir = os.path.basename(os.path.dirname(cloudy_path))
            scene = scene_dir.rsplit("_", 1)[-1]
            clear_name = name.replace("s2_cloudy", "s2", 1)
            clear_path = os.path.join(root, f"{roi}_s2", f"s2_{scene}", clear_name)
            sar_name = name.replace("s2_cloudy", "s1", 1)
            sar_path = os.path.join(root, f"{roi}_s1", f"s1_{scene}", sar_name)

            if not os.path.isfile(clear_path):
                missing_clear += 1
                continue
            if include_sar and not os.path.isfile(sar_path):
                missing_sar += 1
                continue

            items.append({
                "roi": roi,
                "scene": scene,
                "name": name,
                "cloudy": _relative(cloudy_path, root),
                "clear": _relative(clear_path, root),
                "s1": _relative(sar_path, root) if include_sar else None,
            })

    if not items:
        expected = "cloudy + clear + SAR" if include_sar else "cloudy + clear"
        raise RuntimeError(
            f"No complete {expected} samples found under {root}. "
            f"Check the TUM folder layout and ROI names. Missing clear={missing_clear}, "
            f"missing SAR={missing_sar}."
        )
    print(
        f"Scanned {len(items):,} paired samples from {len(rois)} ROI collection(s). "
        f"Skipped missing clear={missing_clear:,}, missing SAR={missing_sar:,}."
    )
    return items


def _split_groups(items: list[dict[str, Any]], seed: int, fracs: tuple[float, float, float]):
    groups = sorted({(x["roi"], str(x["scene"])) for x in items})
    if len(groups) < 3:
        raise RuntimeError(
            f"Only {len(groups)} scene groups were found. At least 3 are required "
            "for non-empty train/validation/test splits."
        )
    rng = random.Random(seed)
    rng.shuffle(groups)
    n = len(groups)
    n_train = max(1, int(fracs[0] * n))
    n_val = max(1, int(fracs[1] * n))
    if n_train + n_val >= n:
        n_train = n - 2
        n_val = 1
    assignment = {}
    for idx, group in enumerate(groups):
        assignment[group] = "train" if idx < n_train else "val" if idx < n_train + n_val else "test"
    splits = {"train": [], "val": [], "test": []}
    for item in items:
        splits[assignment[(item["roi"], str(item["scene"]))]].append(item)
    return splits, len(groups)


def make_splits(
    root: str,
    rois: list[str],
    out_json: str,
    include_sar: bool = True,
    seed: int = 42,
    fracs: tuple[float, float, float] = (0.7, 0.1, 0.2),
    rebuild: bool = False,
    split_mode: str = "official",
) -> dict[str, Any]:
    """Create/reuse a versioned scene-level manifest; reject stale ROI manifests."""
    out_json = os.path.abspath(out_json)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    expected_meta = {
        "version": MANIFEST_VERSION,
        "rois": list(rois),
        "include_sar": bool(include_sar),
        "seed": int(seed),
        "fractions": list(fracs),
        "grouping": "roi+scene",
        "split_mode": split_mode,
    }

    if os.path.isfile(out_json) and not rebuild:
        with open(out_json, "r", encoding="utf-8") as f:
            saved = json.load(f)
        if "metadata" not in saved or "splits" not in saved:
            raise RuntimeError(
                f"{out_json} is an older manifest format. Rename/delete it or pass --rebuild_manifest."
            )
        for key, value in expected_meta.items():
            if saved["metadata"].get(key) != value:
                raise RuntimeError(
                    f"Existing manifest {out_json} has {key}={saved['metadata'].get(key)!r}, "
                    f"but this run requests {value!r}. Use a new --manifest path or --rebuild_manifest."
                )
        # Verify that the manifest's files still exist at this machine's data_root.
        missing = []
        for split_name in ("train", "val", "test"):
            for item in saved["splits"][split_name]:
                for key in ("cloudy", "clear", "s1"):
                    rel = item.get(key)
                    if rel and not os.path.isfile(os.path.join(root, rel)):
                        missing.append(rel)
                        if len(missing) >= 5:
                            break
                if len(missing) >= 5:
                    break
        if missing:
            raise RuntimeError(
                "Manifest paths are missing under this data_root. Keep the same relative folder "
                f"layout or rebuild the manifest. Examples: {missing}"
            )
        return saved

    items = scan(root, rois, include_sar=include_sar)
    if split_mode == "official":
        splits = {"train": [], "val": [], "test": []}
        unknown = []
        for item in items:
            split_name = official_scene_split(item["roi"], item["scene"])
            if split_name is None:
                unknown.append(f"{item['roi']}/scene_{item['scene']}")
            else:
                splits[split_name].append(item)
        if unknown:
            raise RuntimeError(
                "Some scene IDs are not present in the authors' official split mapping. "
                f"Examples: {unknown[:10]}. Use --split_mode scene only if you intentionally want a custom split."
            )
        empty = [name for name, values in splits.items() if not values]
        if empty:
            raise RuntimeError(
                f"Official split has empty {empty} partition(s) for ROI selection {rois}. "
                "Download the corresponding official split ROIs or use a custom scene split."
            )
        group_count = len({(x["roi"], str(x["scene"])) for x in items})
    elif split_mode == "scene":
        splits, group_count = _split_groups(items, seed, fracs)
    else:
        raise ValueError("split_mode must be 'official' or 'scene'")
    manifest = {
        "metadata": {**expected_meta, "scene_groups": group_count, "sample_count": len(items)},
        "splits": splits,
    }
    tmp = out_json + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, out_json)
    return manifest


def read_manifest(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    # Support the new wrapped format only; silently reading legacy manifests can
    # hide incompatible absolute paths or stale dataset selections.
    if "splits" not in manifest:
        raise RuntimeError("Unsupported manifest format. Regenerate it with the adapted train.py.")
    return manifest


def _read_tif(path: str) -> np.ndarray:
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError("rasterio is required. Install it with: pip install rasterio") from exc
    with rasterio.open(path) as src:
        # Masked read prevents declared nodata pixels from becoming extreme values.
        arr = src.read(masked=True).astype(np.float32).filled(np.nan)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


class Sen12msCR(Dataset):
    def __init__(
        self,
        root: str,
        items: list[dict[str, Any]],
        train: bool = False,
        crop: int | None = None,
        use_sar: bool = True,
        s2_scale: float = 10000.0,
        sar_min_db: float = -25.0,
        sar_max_db: float = 0.0,
    ):
        if s2_scale <= 0:
            raise ValueError("s2_scale must be positive")
        if sar_max_db <= sar_min_db:
            raise ValueError("sar_max_db must be greater than sar_min_db")
        self.root = os.path.abspath(root)
        self.items = items
        self.train = train
        self.crop = crop
        self.use_sar = use_sar
        self.s2_scale = float(s2_scale)
        self.sar_min_db = float(sar_min_db)
        self.sar_max_db = float(sar_max_db)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        cloudy = np.clip(_read_tif(os.path.join(self.root, item["cloudy"])), 0, self.s2_scale) / self.s2_scale
        clear = np.clip(_read_tif(os.path.join(self.root, item["clear"])), 0, self.s2_scale) / self.s2_scale

        if cloudy.shape[0] != 13 or clear.shape[0] != 13:
            raise RuntimeError(
                f"Expected 13 Sentinel-2 bands, got cloudy={cloudy.shape}, clear={clear.shape}. "
                f"File: {item['name']}"
            )
        if cloudy.shape != clear.shape:
            raise RuntimeError(f"Cloudy/clear shape mismatch for {item['name']}: {cloudy.shape} vs {clear.shape}")

        tensors = [torch.from_numpy(np.ascontiguousarray(x)) for x in (cloudy, clear)]
        cloudy_t, clear_t = tensors
        result = {"cloudy": cloudy_t, "clear": clear_t, "name": item["name"], "roi": item["roi"], "scene": str(item["scene"])}

        if self.use_sar:
            if not item.get("s1"):
                raise RuntimeError(f"SAR path missing in manifest for {item['name']}; rebuild with SAR enabled.")
            sar = _read_tif(os.path.join(self.root, item["s1"]))
            if sar.shape[0] != 2 or sar.shape[1:] != cloudy.shape[1:]:
                raise RuntimeError(
                    f"Expected co-registered 2-band SAR matching S2, got SAR={sar.shape}, S2={cloudy.shape} "
                    f"for {item['name']}"
                )
            sar = np.clip(sar, self.sar_min_db, self.sar_max_db)
            sar = (sar - self.sar_min_db) / (self.sar_max_db - self.sar_min_db)
            result["s1"] = torch.from_numpy(np.ascontiguousarray(sar.astype(np.float32)))

        if self.train:
            height, width = cloudy_t.shape[-2:]
            if self.crop:
                if self.crop > min(height, width):
                    raise ValueError(f"Crop {self.crop} exceeds image size {height}x{width} ({item['name']})")
                if self.crop < min(height, width):
                    y = random.randint(0, height - self.crop)
                    x = random.randint(0, width - self.crop)
                    for key in ("cloudy", "clear", "s1"):
                        if key in result:
                            result[key] = result[key][:, y:y + self.crop, x:x + self.crop]
            if random.random() < 0.5:
                for key in ("cloudy", "clear", "s1"):
                    if key in result:
                        result[key] = result[key].flip(-1)
            if random.random() < 0.5:
                for key in ("cloudy", "clear", "s1"):
                    if key in result:
                        result[key] = result[key].flip(-2)
            k = random.randint(0, 3)
            if k:
                for key in ("cloudy", "clear", "s1"):
                    if key in result:
                        result[key] = torch.rot90(result[key], k, (-2, -1))
        return result

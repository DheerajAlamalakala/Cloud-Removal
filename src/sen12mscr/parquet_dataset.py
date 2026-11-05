"""Lazy SEN12MS-CR Parquet reader for the Hugging Face mirror.

The Hub card defines the train/validation/test scene-file assignments. This
module reads that local card, builds a compact row-group manifest, then loads
only the requested Parquet row group in each worker process.
"""
from __future__ import annotations

import json
import math
import os
import random
import warnings
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info


MANIFEST_VERSION = 1
PARQUET_COLUMNS = {"sar", "cloudy", "target", "sar_shape", "opt_shape", "patch"}


def _parquet_modules():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Parquet support requires pyarrow. Activate the project Python 3.11 "
            "environment and run: python -m pip install -r requirements.txt"
        ) from exc
    return pq


def _read_card_splits(root: Path) -> dict[str, list[str]]:
    card_path = root / "README.md"
    if not card_path.is_file():
        raise FileNotFoundError(f"Hugging Face dataset card not found: {card_path}")
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "Reading the Hub split assignments requires PyYAML. Install project "
            "requirements with the Python 3.11 environment."
        ) from exc

    card = card_path.read_text(encoding="utf-8")
    if not card.startswith("---"):
        raise RuntimeError(f"Dataset card has no YAML front matter: {card_path}")
    parts = card.split("---", 2)
    if len(parts) != 3:
        raise RuntimeError(f"Could not parse dataset card front matter: {card_path}")
    metadata = yaml.safe_load(parts[1]) or {}
    configs = metadata.get("configs") or []
    config = next((x for x in configs if x.get("config_name") == "default"), None)
    if config is None and configs:
        config = configs[0]
    if config is None:
        raise RuntimeError("Dataset card does not declare a Parquet split configuration")

    result: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for entry in config.get("data_files", []):
        source_split = entry.get("split")
        split = {"train": "train", "validation": "val", "test": "test"}.get(source_split)
        if split is None:
            continue
        paths = entry.get("path", [])
        if isinstance(paths, str):
            paths = [paths]
        for raw_path in paths:
            relative = PurePosixPath(str(raw_path))
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"Unsafe path in dataset card: {raw_path!r}")
            result[split].append(relative.as_posix())

    if any(not result[name] for name in result):
        raise RuntimeError("Dataset card must provide non-empty train, validation, and test splits")
    all_paths = [path for values in result.values() for path in values]
    if len(all_paths) != len(set(all_paths)):
        raise RuntimeError("A Parquet scene file is assigned to more than one Hub split")
    return result


def _build_groups(root: Path) -> tuple[dict[str, list[dict[str, Any]]], str]:
    pq = _parquet_modules()
    card_splits = _read_card_splits(root)
    groups: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    seen_files: set[str] = set()

    for split, relative_paths in card_splits.items():
        for relative in relative_paths:
            path = root.joinpath(*PurePosixPath(relative).parts)
            if not path.is_file():
                raise FileNotFoundError(f"Parquet file listed in README.md is missing: {path}")
            parquet_file = pq.ParquetFile(str(path))
            missing_columns = PARQUET_COLUMNS.difference(parquet_file.schema_arrow.names)
            if missing_columns:
                raise RuntimeError(f"{relative} is missing required columns: {sorted(missing_columns)}")
            season = PurePosixPath(relative).parts[0]
            scene = Path(relative).stem.removeprefix("scene_")
            if season not in {"spring", "summer", "fall", "winter"}:
                raise RuntimeError(f"Unexpected season in Parquet path: {relative}")
            for row_group in range(parquet_file.metadata.num_row_groups):
                rows = int(parquet_file.metadata.row_group(row_group).num_rows)
                if rows:
                    groups[split].append({
                        "file": relative,
                        "row_group": row_group,
                        "rows": rows,
                        "season": season,
                        "scene": scene,
                        "size": path.stat().st_size,
                    })
            seen_files.add(relative)

    discovered = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*.parquet")
        if ".cache" not in path.relative_to(root).parts
    }
    if discovered != seen_files:
        missing = sorted(discovered - seen_files)[:5]
        absent = sorted(seen_files - discovered)[:5]
        raise RuntimeError(
            "The local Parquet files do not match the split list in README.md. "
            f"Unlisted files: {missing}; missing files: {absent}."
        )
    if any(not groups[name] for name in groups):
        raise RuntimeError("The dataset card produced an empty split")

    import hashlib
    card_digest = hashlib.sha256((root / "README.md").read_bytes()).hexdigest()
    return groups, card_digest


def make_parquet_manifest(root: str, out_json: str, rebuild: bool = False) -> dict[str, Any]:
    """Build or validate a compact manifest using the Hub-provided file splits."""
    root_path = Path(root).expanduser().resolve()
    out_path = Path(out_json).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"Parquet data root does not exist: {root_path}")
    groups, card_digest = _build_groups(root_path)
    expected_meta = {"version": MANIFEST_VERSION, "format": "hf_parquet", "split_source": "local_dataset_card",
                     "readme_sha256": card_digest}

    if out_path.is_file() and not rebuild:
        saved = json.loads(out_path.read_text(encoding="utf-8"))
        saved_meta = saved.get("metadata", {})
        if any(saved_meta.get(key) != value for key, value in expected_meta.items()) or "splits" not in saved:
            raise RuntimeError(
                f"Existing manifest {out_path} does not match this dataset card. "
                "Choose a new manifest path or pass --rebuild_manifest."
            )
        for split in ("train", "val", "test"):
            if saved["splits"].get(split) != groups[split]:
                raise RuntimeError(
                    f"Existing manifest {out_path} does not match local {split} Parquet files. "
                    "Choose a new manifest path or pass --rebuild_manifest."
                )
        return saved

    counts = {split: sum(group["rows"] for group in values) for split, values in groups.items()}
    manifest = {"metadata": {**expected_meta, "group_counts": {k: len(v) for k, v in groups.items()},
                              "sample_counts": counts},
                "splits": groups}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, out_path)
    return manifest


def _as_chw(array: np.ndarray, channels: int, label: str) -> np.ndarray:
    if array.ndim != 3:
        raise RuntimeError(f"Expected 3D {label} array, got shape {array.shape}")
    if array.shape[0] == channels:
        result = array
    elif array.shape[-1] == channels:
        result = np.transpose(array, (2, 0, 1))
    else:
        raise RuntimeError(f"Expected {channels} {label} channels, got shape {array.shape}")
    return np.ascontiguousarray(result, dtype=np.float32)


def _decode(raw: Any, shape: Any, dtype: np.dtype, channels: int, label: str) -> np.ndarray:
    if raw is None or shape is None:
        raise RuntimeError(f"Parquet sample is missing {label} bytes or shape metadata")
    try:
        array = np.frombuffer(raw, dtype=dtype).reshape(tuple(int(x) for x in shape))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Could not decode {label} with declared shape {shape}") from exc
    return _as_chw(array, channels, label)


class Sen12msCRParquet(IterableDataset):
    """Stream bounded Parquet batches; never materialize a complete row group."""

    def __init__(self, root: str, groups: list[dict[str, Any]], train: bool = False,
                 crop: int | None = None, use_sar: bool = True, s2_scale: float = 10000.0,
                 sar_min_db: float = -25.0, sar_max_db: float = 0.0,
                 batch_size: int = 8, seed: int = 42):
        super().__init__()
        if s2_scale <= 0 or sar_max_db <= sar_min_db or batch_size < 1:
            raise ValueError("Invalid optical scale, SAR interval, or batch size")
        self.root = os.path.abspath(root)
        self.groups = groups
        self.train = train
        self.crop = crop
        self.use_sar = use_sar
        self.s2_scale = float(s2_scale)
        self.sar_min_db = float(sar_min_db)
        self.sar_max_db = float(sar_max_db)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.sample_count = sum(int(group["rows"]) for group in groups)
        self._warned_sar_files: set[str] = set()

    def __len__(self) -> int:
        # This dataset yields already-batched dictionaries to DataLoader.
        return sum(math.ceil(int(group["rows"]) / self.batch_size) for group in self.groups)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _decode_row(self, record_batch, row: int, group: dict[str, Any]) -> dict[str, Any]:
        optical_shape = record_batch.column("opt_shape")[row].as_py()
        cloudy = _decode(record_batch.column("cloudy")[row].as_py(), optical_shape, np.dtype(np.int16), 13, "cloudy S2")
        clear = _decode(record_batch.column("target")[row].as_py(), optical_shape, np.dtype(np.int16), 13, "clear S2")
        if cloudy.shape != clear.shape:
            raise RuntimeError(f"Cloudy/clear shape mismatch in {group['file']}")
        cloudy = np.nan_to_num(np.clip(cloudy, 0, self.s2_scale) / self.s2_scale).astype(np.float32, copy=False)
        clear = np.nan_to_num(np.clip(clear, 0, self.s2_scale) / self.s2_scale).astype(np.float32, copy=False)
        patch = str(record_batch.column("patch")[row].as_py())
        name = f"{group['season']}_scene_{group['scene']}_{patch}"
        result: dict[str, Any] = {
            "cloudy": torch.from_numpy(cloudy),
            "clear": torch.from_numpy(clear),
            "name": name,
            "roi": group["season"],
            "scene": str(group["scene"]),
            "season": group["season"],
            "patch": patch,
        }
        if self.use_sar:
            sar_shape = record_batch.column("sar_shape")[row].as_py()
            sar = _decode(record_batch.column("sar")[row].as_py(), sar_shape, np.dtype(np.float32), 2, "SAR")
            if sar.shape[1:] != cloudy.shape[1:]:
                raise RuntimeError(f"SAR/S2 spatial shape mismatch in {group['file']}: {sar.shape} vs {cloudy.shape}")
            finite = np.isfinite(sar)
            if not finite.any():
                raise RuntimeError(f"SAR contains no finite values in {name} ({group['file']})")
            plausible_db = np.logical_and(sar[finite] >= -80.0, sar[finite] <= 20.0).mean()
            finite_median = float(np.median(sar[finite]))
            if plausible_db < 0.99 or finite_median >= 0.0:
                raise RuntimeError(
                    f"SAR values in {name} do not look like Sentinel-1 dB values "
                    f"(plausible-range fraction={plausible_db:.2%}, median={finite_median:.3f}). "
                    "Check the Parquet SAR units before training."
                )
            if not finite.all():
                # Parquet contains occasional missing SAR pixels. Fill each
                # polarization's invalid pixels with that channel's median so
                # NaNs cannot poison the network or loss for the whole batch.
                for channel in range(sar.shape[0]):
                    channel_finite = np.isfinite(sar[channel])
                    if not channel_finite.any():
                        raise RuntimeError(f"SAR channel {channel} contains no finite values in {name}")
                    sar[channel, ~channel_finite] = np.median(sar[channel, channel_finite])
                if group["file"] not in self._warned_sar_files:
                    fraction = 1.0 - float(finite.mean())
                    warnings.warn(
                        f"Filled {fraction:.3%} nonfinite SAR pixels with per-channel medians "
                        f"in {group['file']} (example patch {name}).",
                        RuntimeWarning,
                    )
                    self._warned_sar_files.add(group["file"])
            sar = np.clip(sar, self.sar_min_db, self.sar_max_db)
            sar = (sar - self.sar_min_db) / (self.sar_max_db - self.sar_min_db)
            result["s1"] = torch.from_numpy(np.ascontiguousarray(sar, dtype=np.float32))

        if self.train:
            height, width = result["cloudy"].shape[-2:]
            if self.crop:
                if self.crop > min(height, width):
                    raise ValueError(f"Crop {self.crop} exceeds image size {height}x{width} ({name})")
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
            rotations = random.randint(0, 3)
            if rotations:
                for key in ("cloudy", "clear", "s1"):
                    if key in result:
                        result[key] = torch.rot90(result[key], rotations, (-2, -1))
        return result

    @staticmethod
    def _collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
        batch: dict[str, Any] = {
            "cloudy": torch.stack([sample["cloudy"] for sample in samples]),
            "clear": torch.stack([sample["clear"] for sample in samples]),
        }
        if "s1" in samples[0]:
            batch["s1"] = torch.stack([sample["s1"] for sample in samples])
        for key in ("name", "roi", "scene", "season", "patch"):
            batch[key] = [sample[key] for sample in samples]
        return batch

    def __iter__(self) -> Iterator[dict[str, Any]]:
        pq = _parquet_modules()
        worker = get_worker_info()
        worker_id, worker_count = (worker.id, worker.num_workers) if worker else (0, 1)
        group_order = list(range(len(self.groups)))
        rng = random.Random(self.seed + self.epoch)
        if self.train:
            rng.shuffle(group_order)
        group_order = group_order[worker_id::worker_count]
        columns = ["cloudy", "target", "opt_shape", "patch"]
        if self.use_sar:
            columns.extend(["sar", "sar_shape"])

        for group_index in group_order:
            group = self.groups[group_index]
            path = os.path.join(self.root, *PurePosixPath(group["file"]).parts)
            if not os.path.isfile(path) or os.path.getsize(path) != int(group["size"]):
                raise RuntimeError(f"Parquet file changed or disappeared after manifest creation: {path}")
            parquet_file = pq.ParquetFile(path)
            rows_seen = 0
            for record_batch in parquet_file.iter_batches(
                batch_size=self.batch_size,
                row_groups=[int(group["row_group"])],
                columns=columns,
                use_threads=False,
            ):
                remaining = int(group["rows"]) - rows_seen
                if remaining <= 0:
                    break
                if record_batch.num_rows > remaining:
                    record_batch = record_batch.slice(0, remaining)
                order = list(range(record_batch.num_rows))
                if self.train:
                    rng.shuffle(order)
                samples = [self._decode_row(record_batch, index, group) for index in order]
                rows_seen += len(samples)
                yield self._collate(samples)
            if rows_seen != int(group["rows"]):
                raise RuntimeError(
                    f"Read {rows_seen} rows from {group['file']} row group {group['row_group']}, "
                    f"expected {group['rows']}"
                )

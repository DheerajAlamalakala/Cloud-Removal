"""Official SEN12MS-CR scene/ROI split IDs transcribed from the authors' loader.

Source: PatrickTUM/SEN12MS-CR-TS/data/dataLoader.py, SEN12MSCR class,
which labels these ROI assignments as conforming to the SEN12MS-CR split.
"""

OFFICIAL_SPLITS = {
    "ROIs1158_spring": {
        "train": [9, 1, 124, 40, 101, 21, 134, 145, 141, 66, 8, 26, 77, 113, 100, 117, 119, 6, 58, 120, 110, 126, 115, 121, 39, 109, 63, 75, 132, 128, 142, 15, 45, 97, 147],
        "val": [17],
        "test": [106, 123, 140, 31, 44],
    },
    "ROIs1868_summer": {
        "train": [90, 87, 25, 124, 114, 135, 40, 101, 42, 31, 36, 139, 56, 133, 55, 43, 113, 100, 76, 123, 143, 93, 125, 89, 120, 126, 72, 115, 121, 146, 140, 95, 102, 7, 11, 132, 15, 137, 4, 27, 147, 86, 47],
        "val": [19, 80, 127, 17],
        "test": [119, 73],
    },
    "ROIs1970_fall": {
        "train": [3, 22, 148, 107, 1, 114, 135, 40, 42, 31, 149, 64, 28, 144, 57, 35, 133, 30, 134, 141, 112, 116, 37, 26, 77, 100, 83, 71, 93, 119, 104, 136, 6, 41, 125, 91, 131, 120, 110, 19, 14, 81, 39, 109, 33, 88, 11, 128, 142, 122, 4, 27, 147, 85, 82, 105],
        "val": [65],
        "test": [139],
    },
    "ROIs2017_winter": {
        "train": [68, 25, 62, 135, 42, 64, 21, 55, 112, 116, 8, 59, 49, 104, 81, 146, 75, 94, 102, 61, 47],
        "val": [22, 107, 130, 84],
        "test": [108, 63],
    },
}


def official_scene_split(roi: str, scene: str) -> str | None:
    """Return train/val/test for one collection + scene ID, or None if unknown."""
    try:
        scene_id = int(scene)
    except (TypeError, ValueError):
        return None
    for split_name in ("train", "val", "test"):
        if scene_id in OFFICIAL_SPLITS.get(roi, {}).get(split_name, []):
            return split_name
    return None

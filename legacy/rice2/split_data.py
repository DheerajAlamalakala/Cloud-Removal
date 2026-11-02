import argparse
import json
import random
import shutil
from pathlib import Path

parser = argparse.ArgumentParser(description="Create a reproducible RICE2 train/val/test split.")
parser.add_argument("--data_root", default="data", help="RICE2 folder containing trainA/trainB/trainM")
parser.add_argument("--manifest", default="split_manifest.json", help="where to write the split manifest")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

ROOT = Path(args.data_root)
FRAC = {"train": 0.7, "val": 0.1, "test": 0.2}

names = sorted(p.name for p in (ROOT / "trainA").iterdir())
# every name must exist in A, B and M
for n in names:
    assert (ROOT/"trainB"/n).exists() and (ROOT/"trainM"/n).exists(), n

random.Random(args.seed).shuffle(names)
n = len(names)
n_train = int(n * FRAC["train"])
n_val = int(n * FRAC["val"])
split = {
    "train": names[:n_train],
    "val": names[n_train:n_train + n_val],
    "test": names[n_train + n_val:],
}
manifest_path = Path(args.manifest)
manifest_path.parent.mkdir(parents=True, exist_ok=True)
with manifest_path.open("w", encoding="utf-8") as f:
    json.dump(split, f, indent=1)

for name, files in split.items():
    if name == "train":
        continue
    for sub in "ABM":
        out = ROOT / f"{name}{sub}"
        out.mkdir(exist_ok=True)
        for f in files:
            shutil.move(str(ROOT / f"train{sub}" / f), str(out / f))
    print(name, len(files))
print("train", len(split["train"]))

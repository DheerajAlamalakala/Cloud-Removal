import random, json, shutil
from pathlib import Path

ROOT = Path("data")
SEED = 42
FRAC = {"train": 0.7, "val": 0.1, "test": 0.2}

names = sorted(p.name for p in (ROOT / "trainA").iterdir())
# every name must exist in A, B and M
for n in names:
    assert (ROOT/"trainB"/n).exists() and (ROOT/"trainM"/n).exists(), n

random.Random(SEED).shuffle(names)
n = len(names)
n_train = int(n * FRAC["train"])
n_val = int(n * FRAC["val"])
split = {
    "train": names[:n_train],
    "val": names[n_train:n_train + n_val],
    "test": names[n_train + n_val:],
}
json.dump(split, open("split_manifest.json", "w"), indent=1)

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
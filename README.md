# Cloud Removal: RICE2 history + SEN12MS-CR project

This repository keeps two cloud-removal workflows side by side and makes the dataset choice explicit:

- **RICE2 (historical workflow):** RGB optical cloud removal with CycleGAN and U-Net. Its runnable source is in `legacy/rice2/`; its saved metric tables and preview images are in `results/rice2/`.
- **SEN12MS-CR (current workflow):** a SAR-assisted residual U-Net that takes cloudy Sentinel-2 plus Sentinel-1 VV/VH and predicts all 13 Sentinel-2 bands. Its source is in `src/sen12mscr/`.

The RICE2 and SEN12MS-CR scores are reported separately. They use different sensors, data, masks, and evaluation protocols, so the numbers should not be compared as if they came from one benchmark. The previous RICE2 checkpoint is not included. No model checkpoint weights are tracked in this repository.

## Repository map

```text
legacy/rice2/                 Historical CycleGAN and RGB U-Net code, including its datasets.py, models.py, and losses.py
src/sen12mscr/                Current SEN12MS-CR dataset.py, model.py, losses.py, training and evaluation
inference/                    SEN12MS-CR browser demo
benchmarks/                   SEN12MS-CR comparisons, profiling and inspection tools
configs/training/             Training and reproducibility notes
splits/                       Clean SEN12MS-CR split manifests
results/rice2/                Preserved RICE2 metrics and images
results/sen12mscr/            SEN12MS-CR metrics, logs, and visual records
```

Root commands remain available for the historical workflow. `--dataset` switches between workflows without moving files or editing imports. Dataset data and model weights stay local and are ignored by Git.

For SEN12MS-CR, use modules under `src/sen12mscr/` and utilities under `benchmarks/`. Root `train.py` and `evaluate.py` are the shared command dispatchers; they preserve the old RICE2 default and accept `--dataset sen12mscr`. The RICE2 implementation stays together under `legacy/rice2/`, keeping similarly named dataset and model modules separated by workflow.

## Clone and install

```powershell
git clone https://github.com/DheerajAlamalakala/Cloud-Removal.git
cd Cloud-Removal
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

For an NVIDIA GPU, install the PyTorch build that matches your CUDA setup before installing the remaining requirements. On CPU, the requirements file installs the standard PyTorch build.

## Switch workflows

### Historical RICE2 workflow

Keep the RICE2 data in the layout described in [`docs/legacy_rice2_guide.md`](docs/legacy_rice2_guide.md). For example:

```powershell
python train.py --dataset rice2 --data_root .\data\rice2 --image_size 256 --channels 3 --batch_size 4 --n_epochs 200 --decay_epoch 100
```

The `rice2` selector is the default, so earlier commands such as `python train.py --data_root ...` continue to use RICE2. To train its U-Net, run `python train_unet.py --data_root .\data\rice2 ...`. Checkpoints stay under `results/rice2/checkpoints/` locally and are excluded from Git. A single PowerShell launcher can select either workflow: `.\run_training.ps1 -Dataset sen12mscr -ExtraArgs @('--manifest', '.\splits\sen12mscr_spring_clean.json', '--epochs', '30')`; use `-Dataset rice2` for the earlier model.

### Current SEN12MS-CR workflow

Download and extract the desired SEN12MS-CR season locally. For the included spring manifest, `--data_root` must point to the directory containing the three TUM collections below (not to one of the inner `s2_*` folders). Keep this relative layout so manifest paths resolve:

```text
data/sen12mscr/
├── ROIs1158_spring_s1/s1_1/*.tif
├── ROIs1158_spring_s2/s2_1/*.tif
└── ROIs1158_spring_s2_cloudy/s2_cloudy_1/*.tif
```

The clean, scene-level manifest is included under `splits/`. It contains paths and sample identifiers, not imagery; the matching season data must be downloaded separately. If the files live elsewhere, set `$env:SEN12MSCR_DATA_ROOT` in PowerShell or pass `--data_root` on each command. Keep dataset archives and extracted imagery out of Git.

```powershell
python train.py --dataset sen12mscr --data_root .\data\sen12mscr --manifest .\splits\sen12mscr_spring_clean.json --out .\results\sen12mscr\training --epochs 30 --batch_size 4 --crop 128
```

Use `python train.py --dataset sen12mscr --help` for all options. Evaluate a saved local checkpoint with:

```powershell
python evaluate.py --dataset sen12mscr --data_root .\data\sen12mscr --manifest .\splits\sen12mscr_spring_clean.json --ckpt .\results\sen12mscr\training\best_ema.pt --split test --out .\results\sen12mscr\evaluation
```

After training, `best_ema.pt` is saved under `results/sen12mscr/training/` and is excluded from Git. Evaluation writes metrics and visualizations under `results/sen12mscr/evaluation/`; model comparison defaults to `results/sen12mscr/final_comparison/`. The Windows demo auto-detects the trained checkpoint and the comparison CSV. To use another checkpoint or data location, set `$env:SEN12MSCR_CHECKPOINT` and/or `$env:SEN12MSCR_DATA_ROOT`, then run `inference/run_demo.ps1`. The previous path `demo/run_demo.ps1` remains as a compatibility launcher.

For a fresh clone, checkpoints are intentionally absent: train the selected workflow locally or supply a compatible checkpoint path before evaluation or inference. RICE2 checkpoints and SEN12MS-CR checkpoints are separate models and cannot be substituted for one another.

To compare two locally available SEN12MS-CR checkpoints on the same clean test split, pass both paths to the Windows launcher:

```powershell
.\run_final_comparison.ps1 `
  -BaselineCheckpoint '.\results\sen12mscr\baseline\best_ema.pt' `
  -CandidateCheckpoint '.\results\sen12mscr\candidate\best_ema.pt'
```

The launcher writes CSV metrics and visual comparisons to `results/sen12mscr/final_comparison/`. It keeps compatibility with the historical local handoff paths when those checkpoint files are present. On other platforms, pass the same checkpoint paths to `benchmarks/compare_models.py` with `--data_root`, `--manifest`, and `--out`.

## Results and presentation

- [Evaluation overview](docs/evaluation.md) summarizes both dataset tracks without mixing their scores.
- [Experiment details](docs/sen12mscr/EXPERIMENT_RESULTS.md) records the validation-based SEN12MS-CR model selection and its limitations.
- [Demo and supervisor wording](docs/sen12mscr/EXPO_GUIDE.md) gives a short walkthrough.
- [Supervisor brief](docs/supervisor_brief.md) explains the project transition and suggested presentation flow.

All historical RICE2 metrics and images remain under `results/rice2/`. The non-checkpoint SEN12MS-CR run records from the laptop are stored under `results/sen12mscr/experiments/`; split manifests are stored once under `splits/`. Training datasets and checkpoint weights remain local.

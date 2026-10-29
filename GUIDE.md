# Cloud Removal for Satellite Imagery (CycleGAN + U-Net)

Train your own model that turns **cloudy** optical satellite images into **cloud-free** ones. This repo includes two models trained and evaluated on the same data split with the same metrics:

| Model | Idea | Supervision |
|-------|------|-------------|
| **CycleGAN** | Two generators + two discriminators with cycle-consistency, plus paired L1 / edge / SSIM losses | Paired + adversarial + cycle |
| **U-Net** | Single encoder-decoder with skip connections (~30M params) | Paired (L1 + SSIM), no GAN |

Both use optical-only input (cloudy → cloud-free), EMA weights, mixed precision (AMP) and strong augmentation.

---

## Results (RICE2, held-out test set of 148 images)

| Metric | Cloudy input (baseline) | CycleGAN | U-Net |
|--------|-------------------------|----------|-------|
| Cloud-region PSNR (dB) | 18.06 | 28.70 | **31.96** |
| Cloud-region SSIM | 0.483 | 0.771 | **0.828** |
| Cloud-region RMSE | 0.162 | 0.0525 | **0.0378** |
| Cloud-region SAM | 7.56 | 3.54 | **2.60** |
| Whole-image PSNR (dB) | 23.02 | 30.51 | **33.18** |
| Whole-image SSIM | 0.730 | 0.833 | **0.882** |
| Whole-image SAM | 4.33 | 2.70 | **1.94** |

> With paired data, a plain U-Net can beat CycleGAN. CycleGAN is most useful when you have **unpaired** data, so try both on your dataset.

---

## 1. Requirements

- Python 3.10+
- An NVIDIA GPU is strongly recommended (about 2 min/epoch for CycleGAN on an RTX 5060 Laptop GPU; about 16 s/epoch for U-Net)
- Google Colab (free T4) also works, but sessions disconnect, so save checkpoints to Google Drive

## 2. Setup

```bash
git clone https://github.com/DheerajAlamalakala/Cloud-Removal.git
cd Cloud-Removal

python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows
venv\Scripts\activate
```

Install PyTorch for your CUDA version from <https://pytorch.org/get-started/locally/>. Newer GPUs (e.g. the RTX 50 series) need a recent CUDA build such as `cu128`. Then install the rest:

```bash
pip install -r requirements.txt
```

Check the GPU is visible:

```python
import torch
print(torch.__version__, torch.cuda.is_available())
```

## 3. Get a dataset

You need **paired** cloudy / cloud-free images (and, optionally, cloud masks). Supported options:

| Dataset | Source | Notes |
|---------|--------|-------|
| **RICE2** (Landsat 8) | Kaggle | Used for the results above; includes masks |
| **RICE** | Kaggle: `shubhank001/rice-remote-sensing-images-for-cloud-removal` | Google Earth imagery pairs |
| **SEN12MS-CR** | <https://mediatum.ub.tum.de/1554803> and <https://patricktum.github.io/cloud_removal/sen12mscr/> | Sentinel-2 pairs, very large |

Code supports two dataset modes: `PairedFolderDataset` (folders of images, used for RICE / RICE2) and `Sen12mscrDataset`.

## 4. Arrange the data

Put the files in this layout. Matching filenames in `A`, `B` and `M` must correspond to the same scene:

```
data/
├── trainA/   # cloudy images
├── trainB/   # cloud-free images (ground truth)
├── trainM/   # cloud masks (optional, enables cloud-area losses)
├── valA/  valB/  valM/
└── testA/ testB/ testM/
```

### Split the data

Create a reproducible train / val / test split (about 70 / 10 / 20):

```bash
python split_data.py --seed 42
```

This moves files into the folders above and writes `split_manifest.json`, so you can reproduce or share the exact split. For a rigorous comparison, **always evaluate every model on the same split**.

> Tip: if several crops come from the same satellite scene, split by scene rather than by file, so near-duplicate crops don't leak between train and test.

## 5. Train CycleGAN

Smoke test first (1 epoch) to confirm paths and GPU memory:

```bash
python train.py --data_root ./data --image_size 256 --channels 3 \
  --batch_size 4 --n_epochs 1 --decay_epoch 1 \
  --output_dir ./checkpoints_test --sample_dir ./results/test
```

Full run (200 epochs, the configuration behind the results above):

```bash
python train.py --data_root ./data --image_size 256 --channels 3 \
  --batch_size 4 --num_workers 2 \
  --n_epochs 200 --decay_epoch 100 \
  --mask_dir ./data/trainM \
  --lambda_identity 10 --lambda_edge 20 --lambda_ssim 5 \
  --ema 0.998 --strong_aug --amp \
  --output_dir ./checkpoints --sample_dir ./results/run1 \
  --checkpoint_interval 20 --sample_interval 1500
```

Key flags:

| Flag | Meaning |
|------|---------|
| `--n_epochs` / `--decay_epoch` | Total epochs; the learning rate decays linearly after `decay_epoch` |
| `--mask_dir` | Cloud masks for cloud-region / clear-region losses |
| `--lambda_identity`, `--lambda_edge`, `--lambda_ssim` | Loss weights |
| `--ema` | Exponential moving average of generator weights (use the `*_ema` checkpoint for evaluation) |
| `--amp` | Mixed precision for speed and lower VRAM |
| `--strong_aug` | Stronger augmentation (helps with small datasets) |
| `--checkpoint_interval` | Save a numbered checkpoint every N epochs |
| `--sample_interval` | Save preview images every N iterations |

### Resume after a crash or disconnect

Add `--resume ./checkpoints` to the same command. It loads `checkpoint_latest`.

### Using Google Colab

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then point `--output_dir`, `--sample_dir` and `--resume` at folders inside `/content/drive/MyDrive/...`. Colab's local disk is wiped on disconnect, so anything not on Drive is lost.

## 6. Train the U-Net baseline

```bash
python train_unet.py --data_root ./data --image_size 256 --channels 3 \
  --batch_size 4 --n_epochs 235 --decay_epoch 135 \
  --output_dir ./checkpoints_unet --sample_dir ./results/unet
```

It uses the same paired losses, EMA and augmentation as CycleGAN, minus the GAN and cycle terms. Run `python train_unet.py --help` to see every option.

## 7. Evaluate

`evaluate.py` reports PSNR, SSIM, RMSE and SAM for the **cloudy-input baseline** and for your model, both on the whole image and on the **cloud region only**:

```bash
# CycleGAN
python evaluate.py --data_root ./data --split test \
  --checkpoint ./checkpoints/checkpoint_latest_ema.pth

# U-Net
python evaluate.py --data_root ./data --split test --model unet \
  --checkpoint ./checkpoints_unet/checkpoint_latest_ema.pth
```

Use `python evaluate.py --help` for the exact flags in your version. Look at the **baseline** numbers first: a model only counts if it clearly beats "do nothing".

Use the `val` split to choose checkpoints and settings, and report the `test` split **once** at the end.

## 8. Project structure

```
Cloud-Removal/
├── config.py          # default hyperparameters
├── datasets.py        # PairedFolderDataset, Sen12mscrDataset
├── split_data.py      # reproducible train/val/test split
├── train.py           # CycleGAN training
├── unet.py            # U-Net model
├── train_unet.py      # U-Net training
├── evaluate.py        # metrics for any model
└── requirements.txt
```

## 9. Troubleshooting

| Problem | Fix |
|---------|-----|
| Out of memory | Lower `--batch_size` (try 2 or 1), keep `--amp` on, or use `--image_size 128` |
| Epochs seem to run twice as long on Windows | Windows file systems are case-insensitive, so `*.png` and `*.PNG` can match the same files twice. In `datasets.py`, make the file list `sorted(set(files))` |
| Green or neon-looking preview images in early epochs | Common early in CycleGAN training. Check again after 20+ epochs; the cycle and L1 losses should keep falling. Raising `--lambda_identity` can help |
| Discriminator loss stuck near 0.2 | The discriminator is winning. Rely on the paired and cycle losses, and judge by the preview images and validation metrics rather than `loss_D` |
| Lost progress when Colab disconnected | Save checkpoints to Google Drive and use `--resume` |
| `CUDA available: False` | Install the PyTorch build matching your CUDA version (see Setup) |

## 10. Ideas to extend

- Add pix2pix as a third model on the same split
- Train on SEN12MS-CR (add SAR as extra input channels)
- Add perceptual loss (`--lambda_perc`)
- Test on your own Landsat / Sentinel crops

## Acknowledgements

- CycleGAN: Zhu et al., *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*
- U-Net: Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*
- Datasets: RICE / RICE2, SEN12MS-CR (TUM)


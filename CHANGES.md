# Cloud removal V2 - what changed and how to run it

Files changed: `train.py`, `models.py`, `datasets.py` (small), `evaluate.py`. New: `losses.py`.
`inference.py` and `preprocess_sen12mscr.py` are unchanged.

Status: the code parses and was reviewed line by line, but it has NOT been run end-to-end
(no GPU/PyTorch in my sandbox). Do the smoke test below before a long run.

## 1. Fixes
- LR decay now works after --resume. --decay_epoch / --n_epochs are absolute epoch numbers.
  (No manual 4-line patch needed any more.)
- checkpoint_latest.pt is saved every epoch, atomically (temp file + rename). Numbered checkpoints
  still follow --checkpoint_interval.
- Checkpoints now also store the run's arguments (so evaluate.py rebuilds the right generator).

## 2. New options (all default OFF, so old behaviour and old checkpoints still work)
| Flag | What it does | Starting value |
|---|---|---|
| --lambda_edge | Sobel edge loss: keeps coastlines, roads, piers | 20 |
| --edge_cloud_boost | edge loss counts (1 + boost) x more inside the cloud mask | 1.0 |
| --lambda_ssim | 1 - SSIM loss, optimises your SSIM metric | 5 |
| --lambda_perc | VGG16 perceptual loss (texture); slower; downloads VGG weights | 0.5 |
| --ema | moving average of generator weights, smoother final model | 0.998 |
| --amp 1 | mixed precision, usually much faster on a T4 | 1 |
| --strong_aug 1 | vertical flips + 90 degree rotations | 1 |

These weights are starting guesses, not tuned. The log prints `edge`, `ssim_l`, `perc` so you can
see their size next to the other losses.

## 3. New generator (needs a NEW training run, cannot load old weights)
`--upsample resize --skip 1 --se 1 --residual_out 1`
- resize: no checkerboard artifacts
- skip: high-resolution detail bypasses the bottleneck (targets the lost harbor structure)
- se: channel attention in residual blocks
- residual_out: generator predicts a correction added to the input, so clear land is copied through

## 4. evaluate.py
- BASELINE row (raw cloudy vs ground truth); whole-image AND cloud-region tables
- several checkpoints in one run -> checkpoint_metrics.csv
- --use_ema, --tta, worst-first per-image CSV, worst-image comparison pictures

## 5. Commands (Colab)

Smoke test first (1 epoch of 5 batches, writes to /content not Drive; assumes latest checkpoint is epoch 100):
```
!python train.py --data_root ./data --mask_dir ./data/trainM --image_size 256 --batch_size 4 --num_workers 2 \
  --n_epochs 101 --decay_epoch 100 --output_dir /content/smoke --sample_dir /content/smoke_samples \
  --resume /content/drive/MyDrive/cloud_removal_checkpoints_day2 --lambda_identity 10 \
  --lambda_edge 20 --lambda_ssim 5 --ema 0.998 --strong_aug 1 --amp 1 --max_batches 5
```

Path A - continue your current model (try this first; compatible with the epoch-100 weights):
```
!python train.py --data_root ./data --mask_dir ./data/trainM --image_size 256 --channels 3 \
  --batch_size 4 --num_workers 2 --n_epochs 120 --decay_epoch 100 \
  --output_dir /content/drive/MyDrive/cloud_removal_checkpoints_day2 \
  --sample_dir /content/drive/MyDrive/cloud_removal_samples/day5 \
  --checkpoint_interval 5 --sample_interval 500 \
  --resume /content/drive/MyDrive/cloud_removal_checkpoints_day2 \
  --lambda_identity 10 --lambda_edge 20 --lambda_ssim 5 --ema 0.998 --strong_aug 1 --amp 1
```

Path B - new advanced generator, from scratch (long run):
```
!python train.py --data_root ./data --mask_dir ./data/trainM --image_size 256 --channels 3 \
  --batch_size 4 --num_workers 2 --n_epochs 150 --decay_epoch 75 \
  --output_dir /content/drive/MyDrive/cloud_removal_v2_scratch \
  --sample_dir /content/drive/MyDrive/cloud_removal_samples/v2_scratch \
  --checkpoint_interval 5 --sample_interval 500 \
  --upsample resize --skip 1 --se 1 --residual_out 1 \
  --lambda_identity 10 --lambda_edge 20 --lambda_ssim 5 --ema 0.998 --strong_aug 1 --amp 1
```

Evaluate (compare before and after):
```
!python evaluate.py --checkpoint <epoch100.pt> <new_latest.pt> \
  --cloudy_dir ./data/testA --clear_dir ./data/testB --mask_dir ./data/testM \
  --save_dir /content/drive/MyDrive/cloud_removal_eval_v2 --use_ema
```

## 6. Honest expectations
- Path A should help structure and SSIM somewhat. It cannot fully undo an over-smoothing habit
  the model learned over 100 epochs.
- Path B is the real attempt at fixing structure, but it is unproven on your data and costs a full run.
- Judge both by the CLOUD REGION table versus the BASELINE row, not by the loss numbers.

## 7. Review fixes (added after an outside code review)
- datasets.py: paired mode now stops if cloudy/clear/mask counts or file-name stems differ
  (`--no_strict_pairing` only warns). Prevents silently training against the wrong target.
- inference.py: rebuilds the generator architecture stored in the checkpoint, supports `--use_ema`
  (before, a model trained with the V2 options could not be loaded).
- train.py: LR mapping is now exactly: full LR through `decay_epoch`, then linear decay.
  n_epochs=120, decay_epoch=100 -> epoch 100: 2.0e-4, 101: 1.9e-4, 110: 1.0e-4, 120: 9.5e-6.
- Not changed (out of scope for your RGB/RICE2 project): 13-band support and SEN12MS-CR preprocessing.

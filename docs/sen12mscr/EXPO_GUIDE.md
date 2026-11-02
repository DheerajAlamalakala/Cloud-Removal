# Expo demo guide

## What to present

This is a **SAR-assisted 13-band Sentinel-2 cloud-removal prototype**. It combines cloudy Sentinel-2 with Sentinel-1 VV/VH, predicts all 13 Sentinel-2 bands, and exports a georeferenced 13-band GeoTIFF. The browser preview shows RGB only.

## Demo flow

1. Start `inference/run_demo.ps1` from the project folder. It loads the fine-tuned EMA checkpoint by default; set `SEN12MSCR_CHECKPOINT` to the baseline file to compare. RGB previews use the same fixed exposure and sRGB display gamma for every panel; exported reflectance values are unchanged. The old `demo/run_demo.ps1` path remains available.
2. Select a high-cloud sample and show the cloudy-to-model slider.
3. Toggle the clear reference and pseudo-cloud mask. Explain that the clear reference is the paired target and that the mask is estimated from RGB differences.
4. Show the 13-band GeoTIFF download to demonstrate that the model output is multispectral, not just an RGB image.

## Claims and limits

- Use the validation comparison in `EXPERIMENT_RESULTS.md` for model selection. The fine-tune's validation gains over baseline are modest: +0.042 dB full-image PSNR and lower SAM, with a small SSIM reduction.
- If citing test results, use the post-selection report in `EXPERIMENT_RESULTS.md` and say the test split was previously examined for other checkpoints.
- The dataset has no ground-truth cloud masks in this pipeline. Call these *pseudo-cloud-region metrics*.
- The current local dataset is spring-only; do not claim all-season generalization.
- The held-out test set has already been used in earlier model comparisons. Avoid tuning or selecting models from test metrics.
- SAR helps inform reconstruction but cannot recover the exact optical observation hidden by cloud.

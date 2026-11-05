# Expo demo guide

## What to present

This is a **SAR-assisted 13-band Sentinel-2 cloud-removal prototype**. It combines cloudy Sentinel-2 with Sentinel-1 VV/VH and predicts all 13 Sentinel-2 bands. The browser preview shows RGB only. GeoTIFF inputs can be exported as georeferenced GeoTIFF predictions; Parquet inputs have no georeferencing and export a 13-band reflectance NPZ instead.

## Demo flow

1. Start `inference/run_demo.ps1` from the project folder. GeoTIFF is the default input; for Parquet set `SEN12MSCR_DATA_FORMAT=parquet` and provide the local Parquet data root, generated manifest, and checkpoint. RGB previews use the same exposure and sRGB conversion for each panel. The optional enhanced preview is for display only; predictions and exported reflectance values are unchanged.
2. Select a high-cloud sample and show the cloudy-to-model slider.
3. Toggle the clear reference and pseudo-cloud mask. Explain that the clear reference is the paired target and that the mask is estimated from RGB differences.
4. Show the 13-band GeoTIFF or NPZ download to demonstrate that the model output is multispectral, not just an RGB image.

## Claims and limits

- Use the validation comparison in `EXPERIMENT_RESULTS.md` for model selection. The fine-tune's validation gains over baseline are modest: +0.042 dB full-image PSNR and lower SAM, with a small SSIM reduction.
- If citing test results, use the post-selection report in `EXPERIMENT_RESULTS.md` and say the test split was previously examined for other checkpoints.
- The dataset has no ground-truth cloud masks in this pipeline. Call these *pseudo-cloud-region metrics*.
- The documented GeoTIFF run is spring-only; do not use it to claim all-season generalization.
- The separate [Run 2 report](PARQUET_RUN2.md) covers an all-season Parquet test split; present it separately from the GeoTIFF official-ROI results and disclose that the Hub test split has already been evaluated.
- The held-out test set has already been used in earlier model comparisons. Avoid tuning or selecting models from test metrics.
- SAR helps inform reconstruction but cannot recover the exact optical observation hidden by cloud.

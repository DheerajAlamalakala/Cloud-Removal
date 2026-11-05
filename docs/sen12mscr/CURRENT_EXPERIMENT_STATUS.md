# Current experiment status

Updated 2026-11-05. This file tracks the Spring GeoTIFF experiment and the later all-season Parquet fine-tune. Their data splits and scores are separate.

## Spring GeoTIFF experiment

- Spring-only SAR-assisted ResUNet, official scene split, 30 epochs completed.
- Training used batch size 8, 128-pixel crops, base width 48, BF16, eight workers, S2 scale 10,000, SAR clipped to -25..0 dB, and a 400-patch validation subset.
- The epoch-20 EMA checkpoint was selected for the follow-up comparison. It scored 29.907 dB cloud PSNR on the training validation subset; full-image PSNR peaked at epoch 14 (30.117 dB).
- The full Spring validation split contains 756 patches from scene 17 only, so this estimate is scene-specific.
- On the held-out Spring test set, one of 3,983 SAR rasters was unreadable. The epoch-20 report uses the remaining 3,982 patches and is not a complete 3,983-patch benchmark evaluation.

## All-season Parquet fine-tune

Run 2 fine-tuned from the Spring epoch-20 EMA using Parquet data from spring, summer, fall, and winter. Its 20-epoch run selected the epoch-16 EMA checkpoint. The paired Hub-provided test comparison includes 7,899 patches and is summarized in [PARQUET_RUN2.md](PARQUET_RUN2.md).

This is a separate data protocol from the Spring official-ROI evaluation. The Parquet test split has already been evaluated; use validation data for future model selection.

## Evaluation cautions

- Cloud regions are pseudo-labels estimated from cloudy/clear RGB differences, not supplied cloud annotations.
- Report the split, data format, and seasons for each result. Do not mix RICE2, GeoTIFF Spring, and Hub Parquet scores.
- The Run 2 comparison is a fine-tune evaluation, not evidence of a from-scratch all-season model.
- The Spring test result excludes one unreadable SAR patch; disclose the 3,982-sample count if reporting it.

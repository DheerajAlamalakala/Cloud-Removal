# Run 2: all-season Parquet fine-tune

Run 2 fine-tuned the SAR-assisted 13-band residual U-Net using the Hugging Face Parquet mirror of SEN12MS-CR. Its data loader streams bounded row-group batches and reads the train, validation, and test scene assignments from the downloaded dataset `README.md`. The test split was not used for training or checkpoint selection.

## Reproduction settings

- Input: cloudy Sentinel-2 (13 bands) and Sentinel-1 VV/VH (2 bands).
- Seasons: spring, summer, fall, and winter.
- Initialization: the Spring epoch-20 EMA checkpoint.
- Fine-tune: 20 epochs, batch size 8, crop 128, base width 48, learning rate `2e-5`, and curvature weight `0.05`.
- Best checkpoint: Run 2 epoch 16 EMA (`best_ema.pt`). Weights are intentionally excluded from Git; obtain the checkpoint through the project's local handoff process and verify it with the [published SHA-256 checksum](../../release/run2_epoch16_best_ema.sha256).
- Test comparison: 7,899 patches from the Parquet mirror's test split, comparing the Spring checkpoint with Run 2 on the same samples.

The code and commands are in the [repository README](../../README.md#optional-all-season-hugging-face-parquet-workflow). Data, generated manifests, checkpoints, and detailed per-image output remain local under ignored paths.

## Paired test summary

These values are means of per-image metrics from the same Parquet test split. Cloud-region metrics use the RGB cloudy/clear difference pseudo-mask; SEN12MS-CR does not provide pixel cloud labels in this pipeline.

| Metric | Spring checkpoint | Run 2 epoch 16 |
|---|---:|---:|
| Full-image PSNR (dB) | 28.7309 | 29.1922 |
| Pseudo-cloud PSNR (dB) | 27.5154 | 28.0031 |
| Full-image SSIM | 0.8768 | 0.8860 |
| Pseudo-cloud SSIM | 0.8472 | 0.8589 |
| Full-image SAM (degrees) | 8.8485 | 8.4633 |
| Pseudo-cloud SAM (degrees) | 9.8619 | 9.2775 |
| Full-image LPIPS | 0.1928 | 0.1772 |
| Pseudo-cloud LPIPS | 0.2261 | 0.2079 |

The comparison favored Run 2 on these aggregate metrics. This is a final report on the Hub-provided test split, not an untouched test set for future model selection. The Hub split is a different evaluation protocol from the GeoTIFF official-ROI manifests; keep their scores separate.

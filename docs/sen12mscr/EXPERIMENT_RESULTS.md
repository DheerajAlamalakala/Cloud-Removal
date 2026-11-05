# Experiment results

## Scope

These are spring-only SEN12MS-CR results from the official scene-level split. The held-out test evaluation contains 3,982 patches; one unreadable SAR patch was excluded in the clean evaluation manifest. Metrics below are averages of per-image metrics, not pooled pixel metrics.

## Expo checkpoint selection (validation only)

The baseline's recorded best epoch was chosen on a 400-patch validation subset; the current fine-tune used all 756 validation patches. To compare fairly, both best EMA checkpoints were re-evaluated on the same 756 patches. The fine-tune checkpoint is epoch 1; later epochs in that run did not improve its validation score.

| Metric | Baseline epoch 6 | Fine-tuned epoch 1 | Difference (fine-tune minus baseline) |
|---|---:|---:|---:|
| Full-image PSNR (dB) | 30.1224 | 30.1647 | +0.0424 |
| Pseudo-cloud PSNR (dB) | 29.8911 | 29.9056 | +0.0145 |
| Full-image SSIM | 0.9147 | 0.9141 | -0.0006 |
| Pseudo-cloud SSIM | 0.9073 | 0.9071 | -0.0002 |
| Full-image SAM (degrees) | 4.8889 | 4.7583 | -0.1307 |
| Pseudo-cloud SAM (degrees) | 5.0600 | 4.9463 | -0.1136 |

Paired bootstrap intervals over validation patches show a positive full-image PSNR difference (95% CI +0.0086 to +0.0764 dB); the pseudo-cloud PSNR difference is uncertain (95% CI -0.0228 to +0.0529 dB). The fine-tune has a small SSIM regression and lower SAM. The expo demo uses the fine-tune because its full-image PSNR and SAM are better on the full validation split; show the baseline as a comparison if space allows. These are small gains, not a dramatic improvement.

## Paired test comparison

Both checkpoints were evaluated on the same clean test manifest. The baseline is the best EMA checkpoint from `runs/fp32_loss_test`; the spectral model is the best EMA checkpoint from `runs/spectral_weighted_v1`.

| Metric | Cloudy input | Baseline | Spectral-weighted |
|---|---:|---:|---:|
| Full-image PSNR (dB) | 18.7199 | 28.6632 | 28.8426 |
| Pseudo-cloud PSNR (dB) | 15.3503 | 26.9333 | 27.1651 |
| Full-image SSIM | 0.6809 | 0.8818 | 0.8858 |
| Pseudo-cloud SSIM | 0.5564 | 0.8453 | 0.8506 |
| Full-image SAM (degrees) | 13.7119 | 8.1432 | 7.8760 |
| Pseudo-cloud SAM (degrees) | 16.3610 | 9.3496 | 9.1007 |

The spectral-weighted checkpoint improves the aggregate image-level metrics in this paired run. Per-band evaluation shows a different view: it improves 8 of 13 bands in both global and pseudo-cloud MAE, with most gains in B06–B12 and small regressions in B01–B05. Keep both checkpoints when visible-band performance and multispectral error are separate goals.

## Expo candidate test report

After selecting the fine-tuned epoch 1 checkpoint on validation, it was evaluated once on the same 3,982-patch test split used by the earlier comparisons. The test results did not change checkpoint selection. This split has already been examined for previous variants, so present these as a final project report, not as an untouched test set.

| Metric | Baseline epoch 6 | Expo fine-tune epoch 1 |
|---|---:|---:|
| Full-image PSNR (dB) | 28.6632 | 28.6464 |
| Pseudo-cloud PSNR (dB) | 26.9333 | 26.9869 |
| Full-image RMSE | 0.04165 | 0.04135 |
| Pseudo-cloud RMSE | 0.05007 | 0.04947 |
| Full-image SSIM | 0.8818 | 0.8820 |
| Pseudo-cloud SSIM | 0.8453 | 0.8461 |
| Full-image SAM (degrees) | 8.1432 | 7.9968 |
| Pseudo-cloud SAM (degrees) | 9.3496 | 9.2001 |

The expo candidate has slightly lower full-image PSNR, but better pseudo-cloud PSNR, RMSE, SSIM, and SAM in this report. Keep the validation-based selection rationale and disclose the small trade-off.

The detailed per-image metrics, grouped summaries, and four-panel visual comparisons are preserved under `results/sen12mscr/experiments/final_comparison/`. Recreate them with `run_final_comparison.ps1` on Windows or `compare_models.py` elsewhere.

## Training and selection

- Baseline: 11 epochs; best validation checkpoint at epoch 6, with 29.9749 dB pseudo-cloud PSNR and 30.2636 dB full-image PSNR.
- Spectral-weighted fine-tuning: initialized from baseline EMA; best validation checkpoint at fine-tuning epoch 3, with 29.8511 dB pseudo-cloud PSNR and 30.1167 dB full-image PSNR.
- Both runs showed validation scores declining while training loss continued to fall. The recorded runs therefore show overfitting after the best epoch; extending them with the same settings is not recommended.
- The test set has been used for this comparison. Select future variants on validation data and reserve test data for final reporting.

## Metric limitation

The dataset does not provide the pixel cloud labels used here. The pseudo-cloud region is estimated from RGB cloudy/clear differences, so it can also include haze, shadow, or acquisition-date changes. Label these as *pseudo-cloud-region* metrics and do not compare them directly with metrics based on supplied ground-truth masks.

## All-season Parquet Run 2

The later all-season fine-tune and its paired Hub Parquet test summary are documented in [PARQUET_RUN2.md](PARQUET_RUN2.md). Those results use a separate split and protocol from the GeoTIFF evaluations above and should be reported separately.

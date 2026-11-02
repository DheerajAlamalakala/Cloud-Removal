# Evaluation overview

## RICE2 historical experiment

The RICE2 project evaluated RGB CycleGAN and U-Net models on 148 held-out images. The repository preserves its reported metrics and visual records in `results/rice2/`; the original workflow is documented in `legacy_rice2_guide.md`. These results use RICE2's supplied cloud masks.

## SEN12MS-CR current experiment

The current work uses the spring subset of SEN12MS-CR and a SAR-assisted 13-band residual U-Net. Validation data informed model selection. The recorded test split has already been examined for earlier variants, so its metrics are final project reporting rather than a fresh, untouched test estimate.

The selected fine-tune's validation gain over the baseline was modest: full-image PSNR increased by 0.042 dB and SAM decreased, with a small SSIM reduction. On the recorded test report, the fine-tune improved pseudo-cloud PSNR, RMSE, SSIM, and SAM, while full-image PSNR was slightly lower. The full tables and caveats are in `sen12mscr/EXPERIMENT_RESULTS.md`.

SEN12MS-CR does not provide pixel cloud labels in this pipeline. Its cloud-region figures use a pseudo-mask derived from cloudy/clear RGB differences. They can include haze, shadow, and acquisition-date changes, and must not be compared directly with RICE2 metrics based on supplied masks.

## How to present the two tracks

Present RICE2 as the earlier RGB optical experiment and SEN12MS-CR as the current multispectral, SAR-assisted extension. Compare the model variants only within the same dataset and split. Keep the datasets, masks, metrics, checkpoints, and results in their named folders.

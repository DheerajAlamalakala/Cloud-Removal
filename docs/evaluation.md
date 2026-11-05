# Evaluation overview

## RICE2 historical experiment

The RICE2 project evaluated RGB CycleGAN and U-Net models on 148 held-out images. The repository preserves its reported metrics and visual records in `results/rice2/`; the original workflow is documented in `legacy_rice2_guide.md`. These results use RICE2's supplied cloud masks.

## SEN12MS-CR current experiment

The SEN12MS-CR work includes a Spring GeoTIFF experiment using the official scene-level ROI split and a later all-season Parquet fine-tune using the Hugging Face mirror's split. They use different data assignments and evaluation protocols; report them separately. Both test splits have already been evaluated and are final project reporting rather than fresh, untouched test estimates.

The Spring selected fine-tune's validation gain over its baseline was modest: full-image PSNR increased by 0.042 dB and SAM decreased, with a small SSIM reduction. The all-season Run 2 paired test summary favored the candidate on the reported aggregate metrics. The full tables and caveats are in `sen12mscr/EXPERIMENT_RESULTS.md` and `sen12mscr/PARQUET_RUN2.md`.

SEN12MS-CR does not provide pixel cloud labels in this pipeline. Its cloud-region figures use a pseudo-mask derived from cloudy/clear RGB differences. They can include haze, shadow, and acquisition-date changes, and must not be compared directly with RICE2 metrics based on supplied masks.

## How to present the two tracks

Present RICE2 as the earlier RGB optical experiment and SEN12MS-CR as the multispectral, SAR-assisted extension. Compare model variants only within the same dataset and split. Identify the SEN12MS-CR data format and split protocol when reporting results.

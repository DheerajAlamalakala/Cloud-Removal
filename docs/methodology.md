# Methodology

## RICE2 (historical)

The earlier project trained optical RGB CycleGAN and paired U-Net models with RICE2 image pairs and supplied masks. Its historical code and workflow are retained under `legacy/rice2/` and `docs/legacy_rice2_guide.md`. The reported held-out set contains 148 images. Results are preserved under `results/rice2/`.

## SEN12MS-CR (current)

The project uses paired SEN12MS-CR Sentinel-1/Sentinel-2 data. The model receives 13 cloudy Sentinel-2 bands and, by default, two Sentinel-1 VV/VH bands, then predicts all 13 clear Sentinel-2 bands. The repository supports a Spring GeoTIFF workflow with the official scene-level ROI split and a separate all-season Parquet workflow using the Hub-provided split.

Model selection uses validation metrics. Test results are reported after selection and disclosed as previously examined. The detailed protocols and exact numbers are in `sen12mscr/EXPERIMENT_RESULTS.md` and `sen12mscr/PARQUET_RUN2.md`.

The current training objective combines band-weighted, cloud-weighted L1 with square-root-stretched SSIM, edge, and LPIPS terms plus a spectral-angle penalty (`lambda_sam=0.5`). Parquet runs also default to a target-matched curvature term (`lambda_curvature=0.05`); the GeoTIFF default is `0`. The historical experiment results use the settings recorded for those runs, so a new run should be reported as a separate experiment.

## Reporting rules

- Report each dataset's results separately; the sensors, scales, masks, and protocols differ.
- Label SEN12MS-CR cloud-region metrics as pseudo-cloud-region metrics.
- State the seasons and split protocol for each run. Run 2 is an all-season fine-tune, not a from-scratch all-season result.
- Keep datasets, checkpoints, and machine-specific paths out of Git; keep compact split manifests, metrics, and selected images in the repository.

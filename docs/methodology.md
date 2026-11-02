# Methodology

## RICE2 (historical)

The earlier project trained optical RGB CycleGAN and paired U-Net models with RICE2 image pairs and supplied masks. Its historical code and workflow are retained under `legacy/rice2/` and `docs/legacy_rice2_guide.md`. The reported held-out set contains 148 images. Results are preserved under `results/rice2/`.

## SEN12MS-CR (current)

The current project uses the SEN12MS-CR paired Sentinel-1/Sentinel-2 data. The model receives 13 cloudy Sentinel-2 bands and, by default, two Sentinel-1 VV/VH bands, then predicts all 13 clear Sentinel-2 bands. The current published experiment uses Spring and the official scene-level ROI split.

Model selection uses validation metrics. Test results are reported after selection and are disclosed as previously examined because earlier variants were also evaluated on that split. The detailed protocol and exact numbers are in `sen12mscr/EXPERIMENT_RESULTS.md`.

## Reporting rules

- Report each dataset's results separately; the sensors, scales, masks, and protocols differ.
- Label SEN12MS-CR cloud-region metrics as pseudo-cloud-region metrics.
- State the spring-only scope and avoid claims of all-season generalization.
- Keep datasets, checkpoints, and machine-specific paths out of Git; keep compact split manifests, metrics, and selected images in the repository.

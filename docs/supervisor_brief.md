# Supervisor brief

## Project story

The project began with RGB cloud removal on RICE2 using CycleGAN and U-Net. It then moved to SEN12MS-CR, a different and more demanding satellite dataset. The current model combines cloudy 13-band Sentinel-2 with Sentinel-1 VV/VH and predicts all 13 cloud-free Sentinel-2 bands.

## Suggested explanation

1. Show the two dataset tracks separately and explain why the transition adds multispectral output and SAR input.
2. Demonstrate the current browser demo with a held-out SEN12MS-CR patch. Show cloudy input, model output, clear paired reference, and the pseudo-cloud-region mask.
3. Show that the model output is a 13-band GeoTIFF; the on-screen RGB is only a preview.
4. Use validation results to explain why the fine-tune was selected. Describe the gains as modest and mention the small SSIM trade-off.
5. If presenting test results, disclose that this test split was examined for earlier variants. Do not describe it as untouched.
6. Show RICE2 results as the previous experiment. Do not compare RICE2 and SEN12MS-CR scores directly because their data and mask protocols differ.

## Reproducibility

The repository retains code, split manifests, metrics, and selected visual records for both tracks. Dataset archives and model checkpoints are not included. The README gives clone, install, and workflow-selection commands.

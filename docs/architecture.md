# Architecture

## Current SEN12MS-CR model

`src/sen12mscr/model.py` defines the residual U-Net used for the current work. `dataset.py` loads paired SEN12MS-CR patches and optional SAR; `losses.py` contains the training losses and metrics; `train.py`, `evaluate.py`, and `evaluate_bands.py` are the primary workflows. The default model uses SAR and predicts all 13 Sentinel-2 bands.

The browser demo is in `inference/`. It renders an RGB preview from the 13-band prediction and can export the georeferenced multispectral output. The preview transfer curve does not alter the exported reflectance values.

## Historical RICE2 models

`legacy/rice2/` retains the RGB CycleGAN and U-Net implementations. Those models use three-channel optical imagery and their original training/evaluation interface. Root launchers route to them for backward compatibility.

The two code paths stay separate because their inputs, outputs, masks, and checkpoints are not interchangeable.

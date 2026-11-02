# Architecture

## Current SEN12MS-CR model

`src/sen12mscr/model.py` defines the residual U-Net used for the current work. `src/sen12mscr/dataset.py` loads paired SEN12MS-CR patches and optional SAR; `src/sen12mscr/losses.py` contains the training losses and metrics; `src/sen12mscr/train.py`, `src/sen12mscr/evaluate.py`, and `src/sen12mscr/evaluate_bands.py` are the primary workflows. SEN12MS-CR profiling and inspection scripts live in `benchmarks/`. The default model uses SAR and predicts all 13 Sentinel-2 bands.

The older RGB workflow lives in `legacy/rice2/`, including its own `datasets.py`, `models.py`, and `losses.py`. Root `train.py` and `evaluate.py` dispatch to the selected workflow and keep RICE2 as the default for older commands. There are no root-level dataset or model implementation modules; use the workflow-specific paths above.

The browser demo is in `inference/`. It renders an RGB preview from the 13-band prediction and can export the georeferenced multispectral output. The preview transfer curve does not alter the exported reflectance values.

## Historical RICE2 models

`legacy/rice2/` retains the RGB CycleGAN and U-Net implementations. Those models use three-channel optical imagery and their original training/evaluation interface. Root launchers route to them for backward compatibility.

The two code paths stay separate because their inputs, outputs, masks, and checkpoints are not interchangeable.

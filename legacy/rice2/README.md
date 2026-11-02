# Legacy RICE2 workflow

This folder contains the earlier CycleGAN and RGB U-Net implementation. It remains runnable so the previous workflow can be reproduced; no RICE2 checkpoint weights are included.

From the repository root, the old commands still select RICE2 by default:

```powershell
python train.py --data_root .\data\rice2 --help
python train_unet.py --data_root .\data\rice2 --help
python evaluate.py --help
```

Or select it explicitly with `--dataset rice2`. The prior evaluation tables and images are preserved in `results/rice2/`. New local checkpoints are saved under `results/rice2/checkpoints/`, which is ignored by Git.

See [`docs/legacy_rice2_guide.md`](../../docs/legacy_rice2_guide.md) for the full historical guide.

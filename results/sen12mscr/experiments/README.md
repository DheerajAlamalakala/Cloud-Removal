# Recent SEN12MS-CR validation experiments

These are validation-only comparisons on the 7,176-patch all-season Parquet validation split. Cloud-region metrics use the RGB-difference pseudo-mask described in the project methodology.

## Eight-view test-time augmentation

`tta_8view_both_val/` evaluates the Both-attention checkpoint with the eight rotations/reflections and averages inverse-transformed predictions. PSNR_full increased from 29.307 to 29.345 dB, SSIM_full from 0.90373 to 0.90549, and SAM_full decreased from 6.220° to 6.185°. Model inference took about 7.97 times as long.

`tta_8view_val/` contains the corresponding self-attention result: PSNR_full increased from 29.208 to 29.264 dB, SSIM_full from 0.90133 to 0.90330, and SAM_full decreased from 6.298° to 6.261°. Inference took about 7.94 times as long.

Each TTA folder includes aggregate and per-image metrics plus timing. The evaluation script is `benchmarks/evaluate_tta.py`.

## Edge-loss fine-tune

`attention_both_edge25_val/` records a five-epoch fine-tune from the Both-attention EMA checkpoint using edge weight 2.5. Against the original checkpoint, RGB edge error decreased from 0.012297 to 0.012111 and line-edge peak decreased from 0.019771 to 0.019328. Cloud PSNR changed from 28.348 to 28.299 dB, and cloud SAM from 6.749° to 6.909°. The small edge improvement came with a spectral-quality tradeoff, so the original Both checkpoint remains the recommended default.

Checkpoint weights are intentionally kept outside Git; the logs, metrics, and visual comparisons are included here.

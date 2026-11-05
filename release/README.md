# Run 2 checkpoint checksum

The Run 2 epoch-16 EMA checkpoint is intentionally not stored in Git. If the checkpoint is available locally as `run2_epoch16_best_ema.pt`, verify it against `run2_epoch16_best_ema.sha256` before using it.

On Windows PowerShell:

```powershell
(Get-FileHash .\run2_epoch16_best_ema.pt -Algorithm SHA256).Hash
```

The hash file records the expected SHA-256 digest for that checkpoint; it does not contain the model weights.

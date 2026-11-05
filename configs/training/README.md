# Training configuration notes

The supported command-line options are defined in `src/sen12mscr/train.py`; use `python train.py --dataset sen12mscr --help` to inspect them. The repository's experiment reports record the selected run settings and validation rationale. Keep machine-specific dataset paths and checkpoint paths in environment variables or local command lines, not in shared configuration files.

The current loss defaults include the spectral-angle term (`--lambda_sam 0.5`). Parquet training also enables curvature loss by default (`--lambda_curvature 0.05`); GeoTIFF training defaults it to `0`. For exact comparisons, record the loss flags alongside the data split and format.

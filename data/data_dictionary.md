# Data Dictionary

## `RawData/gpkg/`

Geospatial packages for grid, land use, road, transit, built-environment, urban-form, and energy layers used by the released workflow. See `RawData/DATA_SOURCES.md` for source notes.

## `surrogate/data/`

- `dataset_20x3.csv`: surrogate tabular dataset used by the public surrogate pipeline.
- `feature_scalers.json`: feature scaling metadata.
- `xgb_metrics.json`: surrogate performance metrics.
- `xgb_feature_importance_*.csv`: feature-importance tables for surrogate targets.

## `CoOpt/results/`

Released JSON outputs from optimization and representative-solution analysis.

## `RealDis/dataset.csv.example`

Header-only example for the real-block matching input expected by `RealDis/match_reals.py`.

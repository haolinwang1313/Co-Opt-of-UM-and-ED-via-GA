# GeneGenera

Gene-layer definitions and decoder/evaluator for 16 x 250 m grids (1 km block). Encodes interpretable genes, decodes to 20 surrogate features, and applies constraints.

## Files

- `decoder.py`: `CellGene` / `GlobalGene` dataclasses, `GeneDecoder` to generate 20 features per grid and apply caps.
- `evaluator.py`: wraps decoder and surrogate prediction; returns per-grid features, energy estimates, block-level summary, and penalties.
- `calibrate.py` / `calibration.json`: optional calibration of weights and caps from baseline data (`surrogate/data/dataset_20x3.csv`).

## Data expectations

- Baseline features: align to `surrogate/data/dataset_20x3.csv`.
- Optional GIS sources for further calibration: 250 m grids, urban form, built-environment, and transport GPKG layers.

## Usage

- Decode genes to features: instantiate `GeneDecoder`, call `decode(cell_genes, global_gene)`.
- Evaluate: use `GeneEvaluator` to get features, energy estimates, summary values, and penalties. Surrogate models are loaded from `surrogate/models/`.

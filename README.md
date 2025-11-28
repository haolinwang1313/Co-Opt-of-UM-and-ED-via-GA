# Co-Opt-of-UM-and-ED-via-GA

This open-source snapshot keeps core code and lightweight placeholders only. Original data, visual outputs, and work logs are excluded.

## What’s included
- `CoOpt/`: NSGA-II optimization pipeline, configs, decoding/evaluation calls, frontier analysis.
- `GeneGenera/`: gene definitions, decoder, evaluator.
- `surrogate/`: surrogate training/inference code (place weights in `surrogate/models/` or link for download).
- `RealDis/`: matching representatives to real 1 km blocks; `dataset.csv.example` shows required columns.
- `RawData/`: placeholder README only; no raw data.

Not included:
- `Visualization/` scripts or figures.
- `work_logs/` (experiment logs).
- Any raw vector/raster data under `RawData/`.

## Environment
- Python 3.9+ (conda/venv recommended).
- Key deps: numpy, pandas, matplotlib, seaborn, scikit-learn, geopandas (if using GPKG).
- Add `requirements.txt` / `environment.yml` as needed.

## How to run
1. Surrogate (optional)  
   - Train or load models under `surrogate/` (place weights in `surrogate/models/` or use a download link).
2. NSGA-II optimization  
   - Configure scenarios/ranges in `CoOpt/config.py` (S1/S2).  
   - Quick test: `python CoOpt/nsga_small.py`.  
   - Full run (multi-gen/pop): `python CoOpt/run_nsga.py` or `run_multi_seeds.py`.  
   - Frontier analysis: `python CoOpt/analyze_frontier.py`.
3. Real block matching  
   - Prepare `RealDis/dataset.csv` (schema in `dataset.csv.example`), assuming each 16 consecutive `grid_id_main` form a 1 km block.  
   - Run `python RealDis/match_reals.py` to get `match_results.csv` (Top-N similar real blocks per representative).

## Data notes
- Original data and visual outputs are not public.  
- `RealDis/dataset.csv.example` is header-only; build your own 250 m dataset to match the schema.  
- If retraining surrogates, preprocess with your licensed data.

## License
- MIT (see LICENSE).  
- Follow data providers’ terms for any external datasets.

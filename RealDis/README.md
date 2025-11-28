# RealDis

Matching representative 1 km solutions to real 1 km blocks.

## Data prep
- `dataset.csv` (not public): 250 m grid features, schema shown in `dataset.csv.example`. Assumes consecutive `grid_id_main` and `block_id = grid_id_main // 16` (16 cells per 1 km block).
- Required columns (example): `grid_id_main`, `cooling_kwh_per_m2`, `heating_kwh_per_m2`, `other_electricity_kwh_per_m2`, `multi_family_ha`, `facility_industrial_ha`, `road_area_ha`, `parks_green_ha`, etc.

## Matching script
- `match_reals.py`: reads representatives (`CoOpt/results/final_outputs/representatives_grids/*.json`) and real blocks, z-score normalizes selected features, computes Euclidean distance, and outputs top-N similar blocks.
- Run: `python RealDis/match_reals.py` (requires a prepared `dataset.csv`).
- Output: `match_results.csv` with top 5 similar real blocks per representative and their features.

## Note
- No real data are included here; only the placeholder example. Prepare your own licensed 250 m data to run the matcher.

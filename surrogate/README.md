# surrogate

Surrogate model training/inference code with lightweight sample artifacts. Replace data/models with your own as needed.

## Contents
- `data/`: sample scalers/metrics/feature-importance files and a demo `dataset_20x3.csv` (20 features + 3 energy targets, kWh/m²).
- `models/`: sample xgboost boosters (`xgb_*_kwh_per_m2.json`).
- `feature_pipeline.py`: builds the 20 input features and applies normalization (uses `feature_scalers.json`).
- `predict_energy.py`: loads models + scalers to predict cooling/heating/other electricity.

## Usage
```bash
# Predict with existing models
python surrogate/predict_energy.py --input path/to/features.csv --output pred.csv

# (Optional) train your own models
# add your training script (e.g., train_surrogate.py) or retrain boosters as needed
```

### Feature pipeline
- `compute_surrogate_features()` accepts urban form, built env, and transport tables (`grid_id_main` required). It can take raw or normalized columns and returns `['grid_id_main', *20 features]` aligned with `data/dataset_20x3.csv`.
- If processing a single candidate, pass a one-row DataFrame; ordering is preserved.

Notes:
- The bundled data/models are placeholders for reproducibility. For production, recalibrate scalers and retrain models on your licensed datasets.

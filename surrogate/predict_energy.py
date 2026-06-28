


from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import importlib

if TYPE_CHECKING:  
    import xgboost as xgb

_XGB_MODULE = None

FEATURE_COLUMNS = [
    "ci_norm",
    "vci_norm",
    "lum_norm",
    "lum_adjacency_norm",
    "lum_intensity_norm",
    "lum_proximity_norm",
    "gi_norm",
    "li_norm",
    "single_family_ha",
    "multi_family_ha",
    "facility_neighborhood_ha",
    "facility_sales_ha",
    "facility_office_ha",
    "facility_education_ha",
    "facility_industrial_ha",
    "parks_green_ha",
    "water_area_ha",
    "road_area_ha",
    "subway_influence_ha",
    "bus_routes_cnt",
]

TARGET_COLUMNS = [
    "cooling_kwh_per_m2",
    "heating_kwh_per_m2",
    "other_electricity_kwh_per_m2",
]

MODEL_FILENAMES = {target: f"xgb_{target}.json" for target in TARGET_COLUMNS}


def require_xgboost():
    
    global _XGB_MODULE
    if _XGB_MODULE is None:
        try:
            _XGB_MODULE = importlib.import_module("xgboost")
        except ImportError as exc:  
            raise SystemExit(
                "xgboost is required for prediction (pip install xgboost)"
            ) from exc
    return _XGB_MODULE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="CSV file containing at least the 20 feature columns.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write predictions CSV with *_pred columns appended.",
    )
    parser.add_argument(
        "--models-dir",
        default=Path(__file__).resolve().parent / "models",
        type=Path,
        help="Directory holding the exported XGBoost *.json files.",
    )
    parser.add_argument(
        "--report-metrics",
        action="store_true",
        help="If ground-truth target columns exist, print RMSE/MAE/R2.",
    )
    return parser.parse_args()


def load_models(models_dir: Path) -> dict[str, "xgb.Booster"]:
    xgb = require_xgboost()
    boosters: dict[str, "xgb.Booster"] = {}
    for target, filename in MODEL_FILENAMES.items():
        model_path = models_dir / filename
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model file: {model_path}")
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        boosters[target] = booster
    return boosters


def ensure_features(data: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in FEATURE_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(f"Dataset missing required features: {missing}")
    return data[FEATURE_COLUMNS]


def predict(
    feature_frame: pd.DataFrame, boosters: dict[str, "xgb.Booster"]
) -> pd.DataFrame:
    xgb = require_xgboost()
    dmatrix = xgb.DMatrix(feature_frame, feature_names=list(feature_frame.columns))
    pred_cols: dict[str, np.ndarray] = {}
    for target, booster in boosters.items():
        pred_cols[f"{target}_pred"] = booster.predict(dmatrix)
    return pd.DataFrame(pred_cols)


def compute_metrics(
    df: pd.DataFrame, pred_df: pd.DataFrame
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for target in TARGET_COLUMNS:
        truth_col = target
        pred_col = f"{target}_pred"
        if truth_col not in df.columns or pred_col not in pred_df.columns:
            continue
        truth = df[truth_col].to_numpy()
        preds = pred_df[pred_col].to_numpy()
        error = preds - truth
        rmse = float(np.sqrt(np.mean(error**2)))
        mae = float(np.mean(np.abs(error)))
        denom = np.sum((truth - truth.mean()) ** 2)
        r2 = float(1 - np.sum(error**2) / denom) if denom > 0 else float("nan")
        metrics[target] = {"rmse": rmse, "mae": mae, "r2": r2}
    return metrics


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    df = pd.read_csv(input_path)
    feature_df = ensure_features(df)

    boosters = load_models(args.models_dir)
    preds = predict(feature_df, boosters)
    result = pd.concat([df.reset_index(drop=True), preds], axis=1)
    result.to_csv(output_path, index=False)

    if args.report_metrics:
        metrics = compute_metrics(result, preds)
        if not metrics:
            print("No ground-truth targets found; skipping metric report.")
        else:
            print("Evaluation metrics (dataset order):")
            for target, vals in metrics.items():
                rmse = vals["rmse"]
                mae = vals["mae"]
                r2 = vals["r2"]
                print(
                    f"{target}: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}"
                )


if __name__ == "__main__":
    main()

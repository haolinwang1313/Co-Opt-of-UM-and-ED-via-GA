

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.cluster import KMeans
except ImportError as exc:  
    raise SystemExit("scikit-learn is required for calibration (pip install scikit-learn)") from exc

SURROGATE_DATASET = Path(__file__).resolve().parents[1] / "surrogate" / "data" / "dataset_20x3.csv"
CALIB_PATH = Path(__file__).resolve().parent / "calibration.json"
CELL_AREA_HA = (250.0 ** 2) / 10_000.0


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(SURROGATE_DATASET)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def calibrate_shape_effects(df: pd.DataFrame) -> list[dict]:
    features = ["ci_norm", "vci_norm", "lum_adjacency_norm", "lum_intensity_norm"]
    data = df[features].values
    kmeans = KMeans(n_clusters=4, random_state=0, n_init=10)
    labels = kmeans.fit_predict(data)
    df_temp = df.copy()
    df_temp["shape_cluster"] = labels
    global_mean = df_temp[features].replace(0, np.nan).mean().fillna(1.0)
    cluster_stats = []
    for cluster in range(4):
        subset = df_temp[df_temp["shape_cluster"] == cluster]
        if subset.empty:
            continue
        mean_vals = subset[features].mean()
        rel = (mean_vals / global_mean).to_dict()
        cluster_stats.append({
            : cluster,
            : float(rel["ci_norm"]),
            : float(rel["vci_norm"]),
            : float(rel["lum_adjacency_norm"]),
            : float(rel["lum_intensity_norm"]),
            : float(mean_vals["ci_norm"]),
            : float(mean_vals["lum_adjacency_norm"]),
        })

    cluster_stats.sort(key=lambda x: x["mean_ci"], reverse=True)
    names = ["rectangle", "L", "U", "cross"]
    shape_effects = []
    for info, name in zip(cluster_stats, names):
        shape_effects.append({
            : name,
            : round(info["ci"], 4),
            : round(info["vci"], 4),
            : round(info["lum_adj"], 4),
            : round(info["lum_cluster"], 4),
        })
    return shape_effects


def calibrate_service_weights(df: pd.DataFrame) -> dict:
    service_cols = [
        ,
        ,
        ,
        ,
    ]
    service_sum = df[service_cols].sum(axis=1)
    service_share = df[service_cols].div(service_sum.replace(0, np.nan), axis=0).fillna(0.0)
    dominant = service_share.idxmax(axis=1)
    focus_mapping = {
        : "retail",
        : "office",
        : "education",
        : "retail",
    }
    weights = {"retail": [], "office": [], "education": []}
    for focus_name in weights:
        mask = dominant.map(focus_mapping).eq(focus_name)
        subset = service_share[mask]
        if subset.empty:
            continue
        weights[focus_name] = subset.mean().to_dict()

    def fmt(focus: str, key: str) -> float:
        if focus in weights and key in weights[focus]:
            return float(weights[focus][key])
        return float(service_share[key].mean())

    result = {
        : {
            : round(fmt("retail", "facility_neighborhood_ha"), 4),
            : round(fmt("retail", "facility_sales_ha"), 4),
            : round(fmt("retail", "facility_office_ha"), 4),
            : round(fmt("retail", "facility_education_ha"), 4),
        },
        : {
            : round(fmt("office", "facility_neighborhood_ha"), 4),
            : round(fmt("office", "facility_sales_ha"), 4),
            : round(fmt("office", "facility_office_ha"), 4),
            : round(fmt("office", "facility_education_ha"), 4),
        },
        : {
            : round(fmt("education", "facility_neighborhood_ha"), 4),
            : round(fmt("education", "facility_sales_ha"), 4),
            : round(fmt("education", "facility_office_ha"), 4),
            : round(fmt("education", "facility_education_ha"), 4),
        },
    }
    return result


def calibrate_green_weights(df: pd.DataFrame) -> dict:
    parks_ratio = (df["parks_green_ha"] / CELL_AREA_HA).clip(lower=0.0)
    lum_adj = df["lum_adjacency_norm"].clip(lower=0.0)
    q1, q2 = parks_ratio.quantile([0.33, 0.66]).to_list()
    if np.isclose(q1, q2):
        q2 = min(1.0, q1 + 0.05)
    labels = ["linear", "ring", "central"]
    categories = pd.cut(
        parks_ratio,
        bins=[-np.inf, q1, q2, np.inf],
        labels=labels,
        include_lowest=True,
    )
    weights = {}
    global_mean = lum_adj.mean() or 1.0
    for label in labels:
        subset = lum_adj[categories == label]
        if subset.empty:
            weights[label] = 1.0
        else:
            weights[label] = float((subset.mean() / global_mean))
    return {k: round(v, 4) for k, v in weights.items()}


def main() -> None:
    df = load_dataset()
    calibration = {
        : calibrate_shape_effects(df),
        : calibrate_service_weights(df),
        : calibrate_green_weights(df),
    }
    CALIB_PATH.write_text(json.dumps(calibration, indent=2, ensure_ascii=False))
    print(f"Calibration written to {CALIB_PATH}")


if __name__ == "__main__":
    main()

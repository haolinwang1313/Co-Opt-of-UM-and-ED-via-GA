

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
DATA_CELLS = BASE / "RealDis" / "dataset.csv"
REPS_DIR = BASE / "CoOpt" / "results" / "final_outputs" / "representatives_grids"
OUT_CSV = BASE / "RealDis" / "match_results.csv"


AREA_FEATURES = [
    ,
    ,
    ,
    ,
]
ENERGY_FEATURES = [
    ,
    ,
    ,
]


def load_reps() -> dict[str, dict]:
    reps = {}
    for path in REPS_DIR.glob("*.json"):
        data = json.loads(path.read_text())
        key = f"{data['scenario']}-{data['label']}"
        reps[key] = data
    return reps


def rep_block_features(rep: dict) -> pd.Series:
    cells = pd.DataFrame(rep["cells"])
    agg = {}
    
    for col in ENERGY_FEATURES:
        agg[col] = cells[col].mean()
    
    for col in AREA_FEATURES:
        agg[col] = cells[col].sum()
    agg["developed_area_ratio"] = rep["summary"].get("developed_area_ratio", np.nan)
    agg["ind_id"] = rep.get("ind_id")
    agg["label"] = rep["label"]
    agg["scenario"] = rep["scenario"]
    return pd.Series(agg)


def load_real_blocks() -> pd.DataFrame:
    df = pd.read_csv(DATA_CELLS)
    
    df["block_id"] = df["grid_id_main"] // 16
    agg_rows = []
    for block_id, g in df.groupby("block_id"):
        row = {"block_id": block_id}
        for col in ENERGY_FEATURES:
            row[col] = g[col].mean()
        for col in AREA_FEATURES:
            row[col] = g[col].sum()
        row["cell_count"] = len(g)
        agg_rows.append(row)
    blocks = pd.DataFrame(agg_rows)
    return blocks


def zscore(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    zdf = df.copy()
    for col in cols:
        mu = df[col].mean()
        sigma = df[col].std(ddof=0)
        zdf[col] = (df[col] - mu) / sigma if sigma > 0 else 0.0
    return zdf


def main():
    reps = load_reps()
    rep_df = pd.DataFrame([rep_block_features(r) for r in reps.values()])
    real_blocks = load_real_blocks()

    use_cols = ENERGY_FEATURES + AREA_FEATURES
    
    real_z = zscore(real_blocks, use_cols)

    results = []
    for _, rep_row in rep_df.iterrows():
        
        rep_z = rep_row.copy()
        for col in use_cols:
            mu = real_blocks[col].mean()
            sigma = real_blocks[col].std(ddof=0)
            rep_z[col] = (rep_row[col] - mu) / sigma if sigma > 0 else 0.0
        
        diff = real_z[use_cols] - rep_z[use_cols].values
        dists = np.sqrt((diff ** 2).sum(axis=1))
        top_idx = np.argsort(dists)[:5]
        for rank, idx in enumerate(top_idx, start=1):
            block = real_blocks.iloc[idx]
            results.append(
                {
                    : f"{rep_row['scenario']}-{rep_row['label']}",
                    : rank,
                    : int(block["block_id"]),
                    : float(dists[idx]),
                    **{f"block_{c}": block[c] for c in use_cols},
                }
            )

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Saved matches to {OUT_CSV}")


if __name__ == "__main__":
    main()



from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

if __package__ is None:  
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from GeneGenera.decoder import (
    CELL_AREA_M2,
    CELL_AREA_HA,
    CellGene,
    GeneDecoder,
    GlobalGene,
)
from surrogate.predict_energy import (
    FEATURE_COLUMNS,
    MODEL_FILENAMES,
    TARGET_COLUMNS,
    ensure_features,
    load_models,
    predict,
    require_xgboost,
)

SURROGATE_MODELS_DIR = Path(__file__).resolve().parents[1] / "surrogate" / "models"
BASELINE_DATASET = Path(__file__).resolve().parents[1] / "surrogate" / "data" / "dataset_20x3.csv"

BUILT_COLUMNS = [
    ,
    ,
    ,
    ,
    ,
    ,
    ,
]


@dataclass
class GeneEvaluator:
    decoder: GeneDecoder = field(default_factory=GeneDecoder)
    models_dir: Path = field(default=SURROGATE_MODELS_DIR)

    def __post_init__(self) -> None:
        require_xgboost()
        self.boosters = load_models(self.models_dir)
        df = pd.read_csv(BASELINE_DATASET)
        self.feature_bounds = {
            col: (
                float(df[col].quantile(0.01)),
                float(df[col].quantile(0.99)),
            )
            for col in FEATURE_COLUMNS
        }
        self.dev_ratio_bounds = (1.3, 1.6)
        self.block_caps = {
            : 45.0,
            : 45.0,
            : 1.8,
        }
        caps_path = Path("CoOpt/block_stats.json")
        if caps_path.exists():
            data = json.loads(caps_path.read_text())
            try:
                self.block_caps["multi_family_ha"] = data["built"]["multi_family_ha"]["max"]
                self.block_caps["facility_industrial_ha"] = data["built"]["facility_industrial_ha"]["max"]
                self.block_caps["road_area_ha"] = data["transport"]["road_area_ha"]["max"]
            except KeyError:
                pass

    def evaluate(
        self, cell_genes: Iterable[CellGene], global_gene: GlobalGene
    ) -> dict[str, pd.DataFrame | dict | list]:
        features = self.decoder.decode(cell_genes, global_gene)
        pred_df = predict(ensure_features(features), self.boosters)
        result = pd.concat([features.reset_index(drop=True), pred_df], axis=1)
        outliers = self._detect_outliers(features)
        summary = self._aggregate(result)
        penalty, breakdown = self._compute_penalty(features, summary, outliers)
        summary["penalty"] = penalty
        summary["penalty_breakdown"] = breakdown
        return {
            : features,
            : result,
            : summary,
            : outliers,
        }

    def _detect_outliers(self, features: pd.DataFrame) -> list[str]:
        flags: list[str] = []
        check_cols = [col for col in self.feature_bounds if col not in {"lum_norm", "lum_proximity_norm"}]
        for col in check_cols:
            low, high = self.feature_bounds[col]
            below = features[col] < low
            above = features[col] > high
            if below.any():
                ids = features.loc[below, "grid_id_main"].tolist()
                flags.append(f"{col} below p1 for grids {ids}")
            if above.any():
                ids = features.loc[above, "grid_id_main"].tolist()
                flags.append(f"{col} above p99 for grids {ids}")
        return flags

    def _compute_penalty(
        self,
        features: pd.DataFrame,
        summary: dict,
        outliers: list[str],
    ) -> tuple[float, dict[str, float]]:
        breakdown: dict[str, float] = {}
        for col in ["multi_family_ha", "facility_industrial_ha", "road_area_ha"]:
            cap = self.block_caps[col]
            excess = (features[col] - cap).clip(lower=0)
            breakdown[f"cap_{col}"] = float(excess.sum())
        lower, upper = self.dev_ratio_bounds
        dev_ratio = summary.get("developed_area_ratio", 0.0)
        dev_penalty = 0.0
        if dev_ratio < lower:
            dev_penalty = (lower - dev_ratio) * 100
        elif dev_ratio > upper:
            dev_penalty = (dev_ratio - upper) * 100
        breakdown["dev_ratio"] = dev_penalty
        breakdown["outliers"] = float(len(outliers) * 500)
        total = sum(breakdown.values())
        return total, breakdown

    def _aggregate(self, result: pd.DataFrame) -> dict[str, float]:
        developed_ha = result[BUILT_COLUMNS].sum(axis=1)
        developed_m2 = developed_ha * 10_000.0
        agg = {}
        total_area_m2 = len(result) * CELL_AREA_M2
        for target in TARGET_COLUMNS:
            pred_col = f"{target}_pred"
            cell_energy_mwh = (result[pred_col] * developed_m2) / 1_000.0
            agg[f"{target}_total_mwh"] = float(cell_energy_mwh.sum())
            agg[f"{target}_average_kwh_per_m2"] = float(
                cell_energy_mwh.sum() * 1_000.0 / max(developed_m2.sum(), 1e-6)
            )
        agg["developed_area_ratio"] = float(developed_ha.sum() / (len(result) * CELL_AREA_HA))
        agg["energy_balance_index"] = float(result[[f"{t}_pred" for t in TARGET_COLUMNS]].std().mean())
        agg["total_area_m2"] = float(total_area_m2)
        return agg


if __name__ == "__main__":
    evaluator = GeneEvaluator()
    sample_genes = [
        CellGene(grid_id=0, shape_id=0, coverage_factor=1.0, floor_base=15, mix_seed=(0.4, 0.3, 0.2)),
        CellGene(grid_id=1, shape_id=1, coverage_factor=0.9, floor_base=10, mix_seed=(0.3, 0.4, 0.2)),
    ]
    global_gene = GlobalGene(target_far=2.6, industrial_quota=0.15)
    report = evaluator.evaluate(sample_genes, global_gene)
    print(report["predictions"].head())
    print(report["summary"])
    if report["outlier_flags"]:
        print("Outliers:")
        for flag in report["outlier_flags"]:
            print(" -", flag)

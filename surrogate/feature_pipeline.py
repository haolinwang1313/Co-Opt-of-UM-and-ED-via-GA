


from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

SURROGATE_ROOT = Path(__file__).resolve().parent
SCALER_PATH = SURROGATE_ROOT / "data" / "feature_scalers.json"

FORM_FEATURES = [
    ,
    ,
    ,
    ,
    ,
    ,
    ,
    ,
]

BUILT_FEATURES = [
    ,
    ,
    ,
    ,
    ,
    ,
    ,
    ,
    ,
]

TRANSPORT_FEATURES = [
    ,
    ,
    ,
]

FEATURE_COLUMNS = FORM_FEATURES + BUILT_FEATURES + TRANSPORT_FEATURES


@dataclass(frozen=True)
class Normalizer:
    

    name: str
    log_transform: bool
    clip_low: float | None
    clip_high: float | None
    min_val: float
    max_val: float

    def transform(self, series: pd.Series) -> pd.Series:
        values = series.astype(float).replace([np.inf, -np.inf], np.nan)
        if self.log_transform:
            values = values.clip(lower=0.0)
            values = pd.Series(np.log1p(values), index=series.index)
        values = values.clip(lower=self.clip_low, upper=self.clip_high)
        denom = self.max_val - self.min_val
        if denom == 0:
            return pd.Series(np.nan, index=series.index)
        return (values - self.min_val) / denom


def _load_normalizers(path: Path = SCALER_PATH) -> Mapping[str, Normalizer]:
    data = json.loads(path.read_text())
    return {
        name: Normalizer(
            name=name,
            log_transform=meta["log_transform"],
            clip_low=meta["clip_low"],
            clip_high=meta["clip_high"],
            min_val=meta["min"],
            max_val=meta["max"],
        )
        for name, meta in data.items()
    }


NORMALIZERS = _load_normalizers()


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Input dataframe missing columns: {missing}")


def _extract_form_column(
    df: pd.DataFrame, norm_col: str, raw_col: str
) -> pd.Series:
    if norm_col in df.columns:
        return df[norm_col].astype(float)
    if raw_col not in df.columns:
        raise KeyError(f"Input dataframe must include '{norm_col}' or raw '{raw_col}'")
    return NORMALIZERS[raw_col].transform(df[raw_col])


def _normalize_form_features(urban_form_df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(urban_form_df, ["grid_id_main"])
    df = pd.DataFrame({"grid_id_main": urban_form_df["grid_id_main"]})
    df["ci_norm"] = _extract_form_column(urban_form_df, "ci_norm", "ci")
    df["vci_norm"] = _extract_form_column(urban_form_df, "vci_norm", "vci")
    df["lum_norm"] = _extract_form_column(urban_form_df, "lum_norm", "lum")
    df["lum_adjacency_norm"] = _extract_form_column(
        urban_form_df, "lum_adjacency_norm", "lum_adjacency"
    )
    df["lum_intensity_norm"] = _extract_form_column(
        urban_form_df, "lum_intensity_norm", "lum_intensity"
    )
    df["lum_proximity_norm"] = _extract_form_column(
        urban_form_df, "lum_proximity_norm", "lum_proximity"
    )
    df["gi_norm"] = _extract_form_column(urban_form_df, "gi_norm", "gi_mean")
    df["li_norm"] = _extract_form_column(urban_form_df, "li_norm", "li_mean")
    return df


def _to_hectares(df: pd.DataFrame, base_name: str) -> pd.Series:
    ha_col = f"{base_name}_ha"
    m2_col = f"{base_name}_m2"
    if ha_col in df.columns:
        return df[ha_col].astype(float)
    if m2_col in df.columns:
        return df[m2_col].astype(float) / 10_000.0
    raise KeyError(f"Expected '{ha_col}' or '{m2_col}' in input dataframe")


def _compute_built_features(built_df: pd.DataFrame) -> pd.DataFrame:
    required = ["grid_id_main"]
    _require_columns(built_df, required)
    df = pd.DataFrame({"grid_id_main": built_df["grid_id_main"]})
    df["single_family_ha"] = _to_hectares(built_df, "single_family")
    df["multi_family_ha"] = _to_hectares(built_df, "multi_family")
    df["facility_neighborhood_ha"] = _to_hectares(built_df, "facility_neighborhood")
    df["facility_sales_ha"] = _to_hectares(built_df, "facility_sales")
    df["facility_office_ha"] = _to_hectares(built_df, "facility_office")
    df["facility_education_ha"] = _to_hectares(built_df, "facility_education")
    df["facility_industrial_ha"] = _to_hectares(built_df, "facility_industrial")
    df["parks_green_ha"] = _to_hectares(built_df, "parks_green")
    df["water_area_ha"] = _to_hectares(built_df, "water_area")
    df[BUILT_FEATURES] = df[BUILT_FEATURES].fillna(0.0).clip(lower=0.0)
    return df


def _compute_transport_features(transport_df: pd.DataFrame) -> pd.DataFrame:
    required = ["grid_id_main", "bus_routes_cnt"]
    _require_columns(transport_df, required)
    df = pd.DataFrame({"grid_id_main": transport_df["grid_id_main"]})
    df["road_area_ha"] = _to_hectares(transport_df, "road_area")
    df["subway_influence_ha"] = _to_hectares(transport_df, "subway_influence")
    df["bus_routes_cnt"] = transport_df["bus_routes_cnt"].fillna(0).astype(int)
    df[["road_area_ha", "subway_influence_ha"]] = df[["road_area_ha", "subway_influence_ha"]].fillna(0.0).clip(lower=0.0)
    df["bus_routes_cnt"] = df["bus_routes_cnt"].clip(lower=0)
    return df


def compute_surrogate_features(
    urban_form_df: pd.DataFrame,
    built_env_df: pd.DataFrame,
    transport_df: pd.DataFrame,
) -> pd.DataFrame:
    

    form = _normalize_form_features(urban_form_df)
    built = _compute_built_features(built_env_df)
    transport = _compute_transport_features(transport_df)

    merged = form.merge(built, on="grid_id_main", how="inner")
    merged = merged.merge(transport, on="grid_id_main", how="inner")
    ordered_cols = ["grid_id_main", *FEATURE_COLUMNS]
    return merged[ordered_cols]

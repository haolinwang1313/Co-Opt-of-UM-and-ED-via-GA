

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

SURROGATE_DATASET = Path(__file__).resolve().parents[1] / "surrogate" / "data" / "dataset_20x3.csv"
CALIB_PATH = Path(__file__).resolve().parent / "calibration.json"
CELL_LENGTH_M = 250.0
CELL_AREA_M2 = CELL_LENGTH_M ** 2
CELL_AREA_HA = CELL_AREA_M2 / 10_000.0
EPS = 1e-6



DEFAULT_SHAPE_EFFECTS = [
    {"name": "rectangle", "ci": 1.00, "vci": 0.98, "lum_adj": 0.95, "lum_cluster": 0.90},
    {"name": "L", "ci": 0.93, "vci": 1.02, "lum_adj": 1.05, "lum_cluster": 0.85},
    {"name": "U", "ci": 0.90, "vci": 1.05, "lum_adj": 1.08, "lum_cluster": 0.80},
    {"name": "cross", "ci": 0.97, "vci": 1.00, "lum_adj": 0.92, "lum_cluster": 0.88},
]

DEFAULT_GREEN_WEIGHTS = {"central": 1.05, "ring": 1.00, "linear": 0.95}
DEFAULT_SERVICE_WEIGHTS = {
    : {"neighborhood": 0.35, "sales": 0.35, "office": 0.20, "education": 0.10},
    : {"neighborhood": 0.25, "sales": 0.15, "office": 0.40, "education": 0.20},
    : {"neighborhood": 0.20, "sales": 0.10, "office": 0.20, "education": 0.50},
}
DEFAULT_SERVICE_FOCUS = "retail"


def _load_calibration() -> tuple[list[dict], dict, dict]:
    if not CALIB_PATH.exists():
        return DEFAULT_SHAPE_EFFECTS, DEFAULT_SERVICE_WEIGHTS, DEFAULT_GREEN_WEIGHTS
    data = json.loads(CALIB_PATH.read_text())
    shape_effects = data.get("shape_effects", DEFAULT_SHAPE_EFFECTS)
    service_weights = data.get("service_focus_weights", DEFAULT_SERVICE_WEIGHTS)
    green_weights = data.get("green_pattern_weights", DEFAULT_GREEN_WEIGHTS)
    return shape_effects, service_weights, green_weights


SHAPE_EFFECTS, SERVICE_FOCUS_WEIGHTS, GREEN_PATTERN_WEIGHTS = _load_calibration()
BLOCK_STATS_PATH = Path(__file__).resolve().parents[1] / "CoOpt" / "block_stats.json"
if BLOCK_STATS_PATH.exists():
    block_stats = json.loads(BLOCK_STATS_PATH.read_text())
    BUILT_CAPS = {
                : block_stats["built"].get("multi_family_ha", {}).get("max", 40.0) * 1.2,
        : block_stats["built"].get("facility_industrial_ha", {}).get("max", 40.0) * 1.2,
    }
    ROAD_CAP = block_stats["transport"].get("road_area_ha", {}).get("max", 1.5) * 1.2
else:
    BUILT_CAPS = {"multi_family_ha": 45.0, "facility_industrial_ha": 45.0}
    ROAD_CAP = 1.8


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _entropy_norm(shares: Sequence[float]) -> float:
    shares = np.array(shares, dtype=float)
    shares = shares / shares.sum() if shares.sum() > 0 else np.ones_like(shares) / len(shares)
    entropy = -(shares * np.log(shares + EPS)).sum()
    max_entropy = np.log(len(shares))
    return entropy / max_entropy if max_entropy > 0 else 0.0


def _dirichlet_shares(alpha: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(alpha, dtype=float)
    arr = np.clip(arr, 0.01, None)
    arr = np.append(arr, 1.0)  
    arr = arr / arr.sum()
    return {"R": arr[0], "M": arr[1], "B": arr[2], "P": arr[3]}


def _apply_industrial_target(shares: dict[str, float], quota: float) -> dict[str, float]:
    quota = _clamp(quota, 0.02, 0.60)
    delta = quota - shares["B"]
    updated = shares.copy()
    updated["B"] = _clamp(shares["B"] + delta, 0.02, 0.60)
    residual = 1.0 - updated["B"]
    other_keys = [k for k in shares if k != "B"]
    re_norm = sum(shares[k] for k in other_keys)
    if re_norm <= 0:
        for k in other_keys:
            updated[k] = residual / len(other_keys)
    else:
        for k in other_keys:
            updated[k] = residual * shares[k] / re_norm
    return updated


@dataclass
class CellGene:
    grid_id: int
    shape_id: int = 0
    orientation_deg: float = 0.0
    coverage_factor: float = 1.0
    floor_base: int = 12
    floor_gradient: float = 0.0
    podium_ratio: float = 0.3
    mix_seed: tuple[float, float, float] = (0.4, 0.3, 0.2)
    res_type_split: float = 0.5
    service_focus: str = DEFAULT_SERVICE_FOCUS
    green_ratio: float = 0.12
    blue_ratio: float = 0.02
    green_pattern: str = "central"
    road_width_factor: float = 1.0
    road_grid_density: float = 1.0
    bus_route_toggle: int = 0
    transit_hub_flag: bool = False

    def normalized_service_focus(self) -> str:
        if self.service_focus not in SERVICE_FOCUS_WEIGHTS:
            return DEFAULT_SERVICE_FOCUS
        return self.service_focus


@dataclass
class GlobalGene:
    target_far: float = 2.6
    industrial_quota: float = 0.15
    green_network_continuity: float = 0.8
    bus_ring_strength: float = 0.5


@dataclass
class GeneDecoder:
    baseline_path: Path = field(default=SURROGATE_DATASET)

    def __post_init__(self) -> None:
        self.baseline = pd.read_csv(self.baseline_path).set_index("grid_id_main")
        self.feature_columns = [
            ,
            ,
            ,
            ,
            ,
            ,
            ,
            ,
            ,
                ,
            ,
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

    def decode(self, cell_genes: Iterable[CellGene], global_gene: GlobalGene) -> pd.DataFrame:
        genes = list(cell_genes)
        if not genes:
            raise ValueError("cell_genes cannot be empty")
        far_values = np.array([g.coverage_factor * max(g.floor_base, 1) for g in genes], dtype=float)
        mean_far = far_values.mean() if far_values.size else 1.0
        far_scale = global_gene.target_far / max(mean_far, EPS)

        rows = []
        for gene, far_raw in zip(genes, far_values):
            if gene.grid_id not in self.baseline.index:
                raise KeyError(f"grid_id {gene.grid_id} missing in baseline dataset")
            baseline = self.baseline.loc[gene.grid_id]
            shape_effect = SHAPE_EFFECTS[gene.shape_id % len(SHAPE_EFFECTS)]
            shares = _dirichlet_shares(gene.mix_seed)
            shares = _apply_industrial_target(shares, global_gene.industrial_quota)
            gross_floor_area_ha = CELL_AREA_HA * gene.coverage_factor * max(gene.floor_base, 1) * far_scale

            
            res_area = gross_floor_area_ha * shares["R"]
            single_family_ha = res_area * _clamp(gene.res_type_split, 0.0, 1.0)
            multi_family_ha = max(res_area - single_family_ha, 0.0)

            
            service_weights = SERVICE_FOCUS_WEIGHTS[gene.normalized_service_focus()]
            mixed_area = gross_floor_area_ha * shares["M"]
            public_area = gross_floor_area_ha * shares["P"]
            fac_neigh = mixed_area * service_weights["neighborhood"]
            fac_sales = mixed_area * service_weights["sales"]
            fac_office = public_area * service_weights["office"]
            fac_education = public_area * service_weights["education"]
            fac_industrial = gross_floor_area_ha * shares["B"]

            
            continuity = _clamp(global_gene.green_network_continuity, 0.3, 1.2)
            open_space_ha = max(CELL_AREA_HA - (gene.coverage_factor * CELL_AREA_HA * 0.8), 0.0)
            parks_green_ha = min(open_space_ha, CELL_AREA_HA * gene.green_ratio * continuity)
            water_area_ha = min(open_space_ha - parks_green_ha, CELL_AREA_HA * gene.blue_ratio)
            parks_green_ha = max(parks_green_ha, 0.0)
            water_area_ha = max(water_area_ha, 0.0)

            
            road_area = baseline["road_area_ha"] * gene.road_width_factor * gene.road_grid_density
            road_area = _clamp(road_area, 0.0, CELL_AREA_HA * 0.65)
            subway_influence = baseline["subway_influence_ha"]
            if gene.transit_hub_flag:
                subway_influence = min(CELL_AREA_HA, subway_influence + 0.5 * CELL_AREA_HA)
            bus_routes = baseline["bus_routes_cnt"] + gene.bus_route_toggle + int(round(global_gene.bus_ring_strength * 2))
            if gene.transit_hub_flag:
                bus_routes += 1
            bus_routes = max(0, int(round(bus_routes)))

            
            coverage_term = 0.9 + 0.2 * (gene.coverage_factor - 1.0)
            podium_term = 1.0 + 0.1 * (gene.podium_ratio - 0.3)
            gradient_term = 1.0 + 0.05 * gene.floor_gradient
            ci_norm = baseline["ci_norm"] * shape_effect["ci"] * coverage_term * gradient_term
            vci_norm = baseline["vci_norm"] * shape_effect["vci"] * (1.0 + 0.03 * (gene.floor_base - 10))
            lum_shares = [shares["R"], shares["M"], shares["B"], shares["P"]]
            lum_norm = 0.0  
            green_weight = GREEN_PATTERN_WEIGHTS.get(gene.green_pattern, 1.0)
            lum_adjacency = baseline["lum_adjacency_norm"] * shape_effect["lum_adj"] * green_weight
            lum_intensity = max(lum_shares)
            lum_proximity = 0.0

            gi_norm = (
                baseline["gi_norm"]
                * (0.9 + 0.25 * gene.road_grid_density)
                * (1.0 + 0.05 * global_gene.bus_ring_strength)
            )
            li_norm = (
                baseline["li_norm"]
                * (0.9 + 0.2 * gene.road_width_factor)
                * (1.0 + 0.05 * gene.bus_route_toggle)
            )

            row = {
                : gene.grid_id,
                : _clamp(ci_norm, 0.0, 1.0),
                : _clamp(vci_norm, 0.0, 1.5),
                : _clamp(lum_norm, 0.0, 1.0),
                : _clamp(lum_adjacency, 0.0, 1.5),
                : _clamp(lum_intensity, 0.0, 1.0),
                : _clamp(lum_proximity, 0.0, 1.0),
                : _clamp(gi_norm, 0.0, 1.5),
                : _clamp(li_norm, 0.0, 1.5),
                : single_family_ha,
                : min(multi_family_ha, BUILT_CAPS["multi_family_ha"]),
                : fac_neigh,
                : fac_sales,
                : fac_office,
                : fac_education,
                : min(
                    fac_industrial, BUILT_CAPS["facility_industrial_ha"]
                ),
                : parks_green_ha,
                : water_area_ha,
                : min(road_area, ROAD_CAP),
                : _clamp(subway_influence, 0.0, CELL_AREA_HA),
                : bus_routes,
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        ordered_cols = ["grid_id_main", *self.feature_columns]
        return df[ordered_cols]


if __name__ == "__main__":
    
    decoder = GeneDecoder()
    genes = [
        CellGene(grid_id=0, shape_id=1, coverage_factor=1.05, floor_base=18, mix_seed=(0.5, 0.2, 0.2), green_ratio=0.15),
        CellGene(grid_id=1, shape_id=2, service_focus="office", mix_seed=(0.3, 0.4, 0.2),
                 road_grid_density=1.2, bus_route_toggle=1, transit_hub_flag=True),
    ]
    global_gene = GlobalGene(target_far=2.8, industrial_quota=0.18)
    features = decoder.decode(genes, global_gene)
    print(features.head())

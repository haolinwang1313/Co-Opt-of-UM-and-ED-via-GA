from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import os

from GeneGenera.decoder import CellGene, GlobalGene

GRID_IDS: Sequence[int] = (
    732,
    733,
    734,
    735,
    794,
    795,
    796,
    797,
    861,
    862,
    863,
    864,
    932,
    933,
    934,
    935,
)

CELL_BOUNDS = {
    : (0.8, 1.05),
    : (8, 11),
    : (-0.15, 0.15),
    : (0.25, 0.4),
    : (0.0, 0.05),
    : (0.0, 0.01),
    : (0.8, 1.1),
    : (0.8, 1.1),
}

GLOBAL_BOUNDS = {
    : (2.4, 2.8),
    : (0.4, 0.6),
    : (0.6, 0.85),
    : (0.1, 0.4),
}

SERVICE_FOCI = ["retail", "office", "education"]
GREEN_PATTERNS = ["linear", "ring", "central"]

cov_env = os.getenv("COOPT_COVERAGE_MAX")
if cov_env:
    CELL_BOUNDS["coverage_factor"] = (CELL_BOUNDS["coverage_factor"][0], float(cov_env))

floor_env = os.getenv("COOPT_FLOOR_HIGH")
if floor_env:
    CELL_BOUNDS["floor_base"] = (CELL_BOUNDS["floor_base"][0], int(floor_env))


def sample_cell_gene(rng, grid_id: int) -> CellGene:
    coverage = float(rng.uniform(*CELL_BOUNDS["coverage_factor"]))
    floors = int(rng.integers(*CELL_BOUNDS["floor_base"]))
    mix_seed = rng.dirichlet([0.05, 0.4, 0.45, 0.1])[:3]
    return CellGene(
        grid_id=grid_id,
        shape_id=int(rng.integers(0, 4)),
        orientation_deg=float(rng.uniform(0, 45)),
        coverage_factor=coverage,
        floor_base=floors,
        floor_gradient=float(rng.uniform(*CELL_BOUNDS["floor_gradient"])),
        podium_ratio=float(rng.uniform(*CELL_BOUNDS["podium_ratio"])),
        mix_seed=tuple(float(x) for x in mix_seed),
        res_type_split=float(rng.uniform(0.1, 0.3)),
        service_focus="retail",
        green_ratio=float(rng.uniform(*CELL_BOUNDS["green_ratio"])),
        blue_ratio=0.0,
        green_pattern=str(rng.choice(GREEN_PATTERNS)),
        road_width_factor=float(rng.uniform(*CELL_BOUNDS["road_width_factor"])),
        road_grid_density=float(rng.uniform(*CELL_BOUNDS["road_grid_density"])),
        bus_route_toggle=int(rng.integers(-1, 2)),
        transit_hub_flag=False,
    )


def sample_global_gene(rng) -> GlobalGene:
    return GlobalGene(
        target_far=float(rng.uniform(*GLOBAL_BOUNDS["target_far"])),
        industrial_quota=float(rng.uniform(*GLOBAL_BOUNDS["industrial_quota"])),
        green_network_continuity=float(rng.uniform(*GLOBAL_BOUNDS["green_network_continuity"])),
        bus_ring_strength=float(rng.uniform(*GLOBAL_BOUNDS["bus_ring_strength"])),
    )

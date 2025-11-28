

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from GeneGenera.decoder import CellGene, GlobalGene
from GeneGenera.evaluator import GeneEvaluator

RNG = np.random.default_rng(42)

GRID_IDS = [732, 733, 734, 735, 794, 795, 796, 797, 861, 862, 863, 864, 932, 933, 934, 935]

BASE_SHARES = np.array([0.05, 0.45, 0.45])  


def sample_mix_seed() -> tuple[float, float, float]:
    alpha = BASE_SHARES * 60  
    draw = RNG.dirichlet(alpha)
    return tuple(float(x) for x in draw[:3])


def sample_cell_gene(grid_id: int) -> CellGene:
    return CellGene(
        grid_id=grid_id,
        shape_id=int(RNG.integers(0, 4)),
        orientation_deg=float(RNG.uniform(0, 45)),
        coverage_factor=float(RNG.uniform(0.8, 1.05)),
        floor_base=int(RNG.integers(8, 12)),
        floor_gradient=float(RNG.uniform(-0.15, 0.15)),
        podium_ratio=float(RNG.uniform(0.25, 0.4)),
        mix_seed=sample_mix_seed(),
        res_type_split=float(RNG.uniform(0.1, 0.3)),
        service_focus="retail",
        green_ratio=float(RNG.uniform(0.0, 0.05)),
        blue_ratio=0.0,
        green_pattern=str(RNG.choice(["linear", "ring"])),
        road_width_factor=float(RNG.uniform(0.8, 1.1)),
        road_grid_density=float(RNG.uniform(0.8, 1.1)),
        bus_route_toggle=int(RNG.integers(-1, 2)),
        transit_hub_flag=False,
    )

def main() -> None:
    cell_genes = [sample_cell_gene(gid) for gid in GRID_IDS]
    global_gene = GlobalGene(
        target_far=float(RNG.uniform(2.4, 2.9)),
        industrial_quota=float(RNG.uniform(0.4, 0.6)),
        green_network_continuity=float(RNG.uniform(0.6, 0.85)),
        bus_ring_strength=float(RNG.uniform(0.1, 0.4)),
    )

    evaluator = GeneEvaluator()
    report = evaluator.evaluate(cell_genes, global_gene)

    out_path = Path("CoOpt/test_report.json")
    payload = {
        : report["summary"],
        : report["outlier_flags"],
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Summary written to {out_path}")
    if report["outlier_flags"]:
        print("Outlier flags:")
        for flag in report["outlier_flags"]:
            print(" -", flag)


if __name__ == "__main__":
    main()

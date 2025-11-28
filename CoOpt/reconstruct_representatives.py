

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from GeneGenera.decoder import CellGene, GlobalGene
from GeneGenera.evaluator import GeneEvaluator

FINAL_DIR = Path('CoOpt/results/final_outputs')
REPS_PATH = FINAL_DIR / 'representatives.json'
GRID_DB = Path('RawData/gpkg/xinwu_grid_250m.gpkg')
OUT_DIR = FINAL_DIR / 'representatives_grids'
OUT_DIR.mkdir(parents=True, exist_ok=True)

COMBINED_PATHS = {
    : Path('CoOpt/results/S1_combined_frontier.json'),
    : Path('CoOpt/results/S2_combined_frontier.json'),
}

EVALUATOR = GeneEvaluator()


def load_grid_mapping() -> dict[int, dict[str, int]]:
    conn = sqlite3.connect(GRID_DB)
    cur = conn.execute('SELECT grid_id_main, row_raw, col_raw FROM grid_250m')
    mapping = {row[0]: {'row': row[1], 'col': row[2]} for row in cur.fetchall()}
    conn.close()
    return mapping


def dict_to_cell(data: dict) -> CellGene:
    if isinstance(data.get('mix_seed'), list):
        data = data.copy()
        data['mix_seed'] = tuple(data['mix_seed'])
    return CellGene(**data)


def dict_to_global(data: dict) -> GlobalGene:
    return GlobalGene(**data)


def evaluate_solution(entry: dict) -> dict:
    genes = entry['genes']
    cells = [dict_to_cell(cell) for cell in genes['cells']]
    global_gene = dict_to_global(genes['global_gene'])
    return EVALUATOR.evaluate(cells, global_gene)


def main() -> None:
    reps = json.loads(REPS_PATH.read_text())
    grid_map = load_grid_mapping()
    for scenario, labels in reps.items():
        combined = json.loads(COMBINED_PATHS[scenario].read_text())
        by_id = {entry['ind_id']: entry for entry in combined}
        for label, info in labels.items():
            ind_id = info.get('ind_id')
            entry = by_id.get(ind_id)
            if entry is None:
                print(f"Warning: ind_id {ind_id} not found for {scenario}:{label}")
                continue
            report = evaluate_solution(entry)
            df = report['predictions']
            rows = []
            for _, row in df.iterrows():
                grid_id = int(row['grid_id_main'])
                pos = grid_map.get(grid_id, {'row': None, 'col': None})
                rows.append({
                    : grid_id,
                    : pos['row'],
                    : pos['col'],
                    : row.get('cooling_kwh_per_m2_pred', None),
                    : row.get('heating_kwh_per_m2_pred', None),
                    : row.get('other_electricity_kwh_per_m2_pred', None),
                    : row.get('multi_family_ha', None),
                    : row.get('facility_industrial_ha', None),
                    : row.get('road_area_ha', None),
                    : row.get('parks_green_ha', None)
                })
            out = {
                : scenario,
                : label,
                : ind_id,
                : entry['summary'],
                : rows,
            }
            out_path = OUT_DIR / f"{scenario}_{label}.json"
            out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
            print(f"Saved {out_path}")


if __name__ == '__main__':
    main()

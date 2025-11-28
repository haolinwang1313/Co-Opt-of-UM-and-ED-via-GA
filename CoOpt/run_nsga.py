from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

SCENARIOS = [
    {"name": "S1", "coverage": 1.05, "floor_high": 11},
    {"name": "S2", "coverage": 1.06, "floor_high": 11},
]
POP = 80
N_GEN = 60
RESULTS_DIR = Path("CoOpt/results")
SUMMARY_PATH = RESULTS_DIR / "summary_runs.json"
RESULTS_DIR.mkdir(exist_ok=True)


def run_scenario(scenario: dict) -> dict:
    env = os.environ.copy()
    env["COOPT_SCENARIO"] = scenario["name"]
    env["COOPT_COVERAGE_MAX"] = f"{scenario['coverage']:.2f}"
    env["COOPT_FLOOR_HIGH"] = str(scenario["floor_high"])
    env["COOPT_POP_SIZE"] = str(POP)
    env["COOPT_N_GEN"] = str(N_GEN)
    result_dir = RESULTS_DIR / scenario["name"]
    result_dir.mkdir(parents=True, exist_ok=True)
    env["COOPT_FRONTIER_PATH"] = str(result_dir / "final_frontier.json")
    env["COOPT_ANALYSIS_PATH"] = str(result_dir / "analysis.json")
    subprocess.run(["python", "CoOpt/nsga_small.py"], check=True, env=env)
    subprocess.run(["python", "CoOpt/analyze_frontier.py"], check=True, env=env)
    analysis = json.loads((result_dir / "analysis.json").read_text())
    return analysis


def main() -> None:
    recs = []
    for scenario in SCENARIOS:
        print(f"Running scenario {scenario['name']}")
        analysis = run_scenario(scenario)
        rec = {
            : scenario["name"],
            : scenario["coverage"],
            : scenario["floor_high"],
            : analysis.get("count", 0),
            : analysis.get("developed_ratio_stats", {}).get("mean"),
            : analysis.get("energy_stats", {}).get("mean"),
        }
        recs.append(rec)
        print(f"  -> feasible {rec['feasible_count']}, dev_mean {rec['developed_ratio_mean']}, energy_mean {rec['energy_mean']}")
    SUMMARY_PATH.write_text(json.dumps(recs, indent=2, ensure_ascii=False))
    print(f"Summary saved to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

BASE_SCENARIOS = {
    : {"coverage": 1.05, "floor_high": 11},
    : {"coverage": 1.06, "floor_high": 11},
}
SEEDS = [101, 202, 303]
POP = 80
N_GEN = 60
RESULTS_DIR = Path("CoOpt/results")
RESULTS_DIR.mkdir(exist_ok=True)


def run_single(scenario: str, coverage: float, floor_high: int, seed: int) -> Path:
    name = f"{scenario}_seed{seed}"
    env = os.environ.copy()
    env["COOPT_SCENARIO"] = name
    env["COOPT_COVERAGE_MAX"] = f"{coverage:.2f}"
    env["COOPT_FLOOR_HIGH"] = str(floor_high)
    env["COOPT_POP_SIZE"] = str(POP)
    env["COOPT_N_GEN"] = str(N_GEN)
    env["COOPT_SEED"] = str(seed)
    result_dir = RESULTS_DIR / name
    result_dir.mkdir(parents=True, exist_ok=True)
    env["COOPT_FRONTIER_PATH"] = str(result_dir / "final_frontier.json")
    env["COOPT_ANALYSIS_PATH"] = str(result_dir / "analysis.json")
    subprocess.run(["python", "CoOpt/nsga_small.py"], check=True, env=env)
    subprocess.run(["python", "CoOpt/analyze_frontier.py"], check=True, env=env)
    return result_dir / "final_frontier.json"


def dominates(f1, f2) -> bool:
    return all(a <= b for a, b in zip(f1, f2)) and any(a < b for a, b in zip(f1, f2))


def combine_frontiers(scenario: str, frontier_paths: list[Path]) -> Path:
    entries = []
    for path in frontier_paths:
        data = json.loads(path.read_text())
        for item in data:
            entries.append(item)
    nondominated: list[dict] = []
    for entry in entries:
        dominated = False
        f = entry["fitness"]
        for other in entries:
            if other is entry:
                continue
            if dominates(other["fitness"], f):
                dominated = True
                break
        if not dominated:
            nondominated.append(entry)
    out_path = RESULTS_DIR / f"{scenario}_combined_frontier.json"
    out_path.write_text(json.dumps(nondominated, indent=2, ensure_ascii=False))
    env = os.environ.copy()
    env["COOPT_FRONTIER_PATH"] = str(out_path)
    env["COOPT_ANALYSIS_PATH"] = str(RESULTS_DIR / f"{scenario}_combined_analysis.json")
    subprocess.run(["python", "CoOpt/analyze_frontier.py"], check=True, env=env)
    return out_path


def main() -> None:
    summary = []
    for scenario, info in BASE_SCENARIOS.items():
        coverage = info["coverage"]
        floor_high = info["floor_high"]
        frontier_paths = []
        for seed in SEEDS:
            print(f"Running {scenario} with seed {seed}")
            frontier_paths.append(
                run_single(scenario, coverage, floor_high, seed)
            )
        combined_path = combine_frontiers(scenario, frontier_paths)
        analysis = json.loads((RESULTS_DIR / f"{scenario}_combined_analysis.json").read_text())
        summary.append({
            : scenario,
            : coverage,
            : floor_high,
            : analysis.get("count", 0),
            : analysis.get("developed_ratio_stats", {}).get("mean"),
            : analysis.get("energy_stats", {}).get("mean"),
        })
        print(f"Scenario {scenario} combined: feasible {summary[-1]['feasible_count']}, dev_mean {summary[-1]['dev_ratio_mean']}, energy_mean {summary[-1]['energy_mean']}")
    (RESULTS_DIR / "multi_seed_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Summary saved to", RESULTS_DIR / "multi_seed_summary.json")


if __name__ == "__main__":
    main()

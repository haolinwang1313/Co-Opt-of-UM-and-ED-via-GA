

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

COMBOS = [
    {"name": "base", "coverage": 1.05, "floor_high": 11},
    {"name": "A", "coverage": 1.06, "floor_high": 11},
    {"name": "B", "coverage": 1.08, "floor_high": 11},
    {"name": "C", "coverage": 1.05, "floor_high": 12},
    {"name": "D", "coverage": 1.06, "floor_high": 12},
]

SUMMARY_PATH = Path(__file__).resolve().with_name("scan_summary.json")


def run_pipeline(env: dict[str, str]) -> dict:
    subprocess.run(["python", "CoOpt/nsga_small.py"], check=True, env=env)
    subprocess.run(["python", "CoOpt/analyze_frontier.py"], check=True, env=env)
    analysis = json.loads(Path("CoOpt/frontier_analysis.json").read_text())
    return analysis


def main() -> None:
    results = []
    for combo in COMBOS:
        env = os.environ.copy()
        env["COOPT_COVERAGE_MAX"] = f"{combo['coverage']:.2f}"
        env["COOPT_FLOOR_HIGH"] = str(combo["floor_high"])
        analysis = run_pipeline(env)
        count = analysis.get("count", 0)
        dev_stats = analysis.get("developed_ratio_stats", {})
        energy_stats = analysis.get("energy_stats", {})
        record = {
            : combo["name"],
            : combo["coverage"],
            : combo["floor_high"],
            : count,
            : dev_stats.get("min"),
            : dev_stats.get("max"),
            : dev_stats.get("mean"),
            : energy_stats.get("min"),
            : energy_stats.get("max"),
            : energy_stats.get("mean"),
        }
        results.append(record)
        print(
            
            
        )
    SUMMARY_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Summary written to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()

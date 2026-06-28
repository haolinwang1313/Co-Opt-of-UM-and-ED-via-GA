

from __future__ import annotations

import json
from pathlib import Path
import os
import numpy as np

DEFAULT_RESULT = Path(__file__).resolve().with_name("nsga_results.json")
DEFAULT_OUT = Path(__file__).resolve().with_name("frontier_analysis.json")
RESULT_PATH = Path(os.getenv("COOPT_FRONTIER_PATH", DEFAULT_RESULT))
OUT_PATH = Path(os.getenv("COOPT_ANALYSIS_PATH", DEFAULT_OUT))

if not RESULT_PATH.exists():
    raise SystemExit(f"Missing {RESULT_PATH}")

raw = json.loads(RESULT_PATH.read_text())
if not raw:
    raise SystemExit("nsga_results.json is empty")

records = []
for entry in raw:
    summary = entry.get("summary", {})
    stats = entry.get("feature_stats", {})
    energy_total = summary.get("cooling_kwh_per_m2_total_mwh", 0.0) + summary.get("heating_kwh_per_m2_total_mwh", 0.0)
    item = {
        "energy_total_mwh": energy_total,
        "cooling_mwh": summary.get("cooling_kwh_per_m2_total_mwh", 0.0),
        "heating_mwh": summary.get("heating_kwh_per_m2_total_mwh", 0.0),
        "other_mwh": summary.get("other_electricity_kwh_per_m2_total_mwh", 0.0),
        "developed_ratio": summary.get("developed_area_ratio", np.nan),
        "energy_balance": summary.get("energy_balance_index", np.nan),
        "multi_family_sum": stats.get("multi_family_ha_sum"),
        "industrial_sum": stats.get("facility_industrial_ha_sum"),
        "road_sum": stats.get("road_area_ha_sum"),
        "ci_mean": stats.get("ci_norm_mean"),
        "vci_mean": stats.get("vci_norm_mean"),
        "gi_mean": stats.get("gi_norm_mean"),
        "li_mean": stats.get("li_norm_mean"),
    }
    records.append(item)

records.sort(key=lambda x: x["energy_total_mwh"])

analysis = {
    "count": len(records),
    "energy_stats": {
        "min": records[0]["energy_total_mwh"],
        "max": records[-1]["energy_total_mwh"],
        "mean": float(np.mean([r["energy_total_mwh"] for r in records])),
    },
    "developed_ratio_stats": {
        "min": float(np.min([r["developed_ratio"] for r in records])),
        "max": float(np.max([r["developed_ratio"] for r in records])),
        "mean": float(np.mean([r["developed_ratio"] for r in records])),
    },
    "solutions": records,
}

OUT_PATH.write_text(json.dumps(analysis, indent=2, ensure_ascii=False))
print(f"Saved analysis to {OUT_PATH}")
for rec in records:
    print(
        f"energy={rec['energy_total_mwh']:.2f} MWh, "
        f"dev_ratio={rec['developed_ratio']:.3f}, balance={rec['energy_balance']:.3f}"
    )

# A surrogate-assisted framework for district-scale urban morphology optimization toward reduced building energy demand

[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.apenergy.2026.128294-blue.svg)](https://doi.org/10.1016/j.apenergy.2026.128294)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Journal](https://img.shields.io/badge/Journal-Applied%20Energy-2f855a.svg)](https://doi.org/10.1016/j.apenergy.2026.128294)

This repository accompanies the paper **"A surrogate-assisted framework for district-scale urban morphology optimization toward reduced building energy demand"** ([*Applied Energy*](https://www.sciencedirect.com/journal/applied-energy), 2026). It contains the public code, released geospatial layers, surrogate model assets, optimization outputs, and reproducibility notes for the district-scale urban morphology optimization workflow.

## Publication

**A surrogate-assisted framework for district-scale urban morphology optimization toward reduced building energy demand**

[*Applied Energy*](https://www.sciencedirect.com/journal/applied-energy)

2026-11 | Journal article

DOI: [10.1016/j.apenergy.2026.128294](https://doi.org/10.1016/j.apenergy.2026.128294)

Authors: Haolin Wang; Zhi Wu; Wei Gu; Pengxiang Liu; Qirun Sun; Wei Wang

## Overview

District-scale morphology optimization requires a workflow that can generate planning-feasible urban forms, estimate building energy demand quickly, and compare competing design objectives. This repository links gene-based urban morphology encoding, surrogate-assisted energy evaluation, NSGA-II optimization, representative-solution reconstruction, and matching against real urban blocks in Xinwu District, Wuxi, China.

The current archive includes:

- a **16-cell, 250 m grid representation** for 1 km district-scale candidate blocks;
- released **Xinwu geospatial layers** for buildings, roads, land use, transport, water, urban form, built environment, and energy attributes;
- a **20-feature surrogate inference pipeline** with scalers, XGBoost model files, feature-importance outputs, and validation metrics;
- gene decoder and evaluator code for converting interpretable morphology genes into surrogate-ready features;
- released **NSGA-II frontier outputs**, multi-seed summaries, representative grids, and scatter data;
- a real-block matching interface that maps optimized representatives to user-prepared 250 m block datasets.

## Repository Structure

The codebase is organized as follows:

```text
.
|-- CoOpt/                         # NSGA-II optimization and released result analysis
|   |-- config.py                  # Scenario ranges, objectives, and optimization constants
|   |-- nsga_small.py              # Lightweight optimization smoke test
|   |-- run_nsga.py                # Main NSGA-II execution entry point
|   |-- run_multi_seeds.py         # Multi-seed optimization driver
|   |-- analyze_frontier.py        # Pareto-frontier and summary analysis
|   |-- reconstruct_representatives.py # Rebuilds representative 16-grid solutions
|   |-- plot_results.py            # Plotting helpers for optimization outputs
|   |-- scan_configs.py            # Scenario/configuration scan utility
|   |-- results/                   # Released frontiers, analyses, and representative JSON files
|   `-- test_run.py                # Small local check for the optimization workflow
|-- GeneGenera/                    # Gene encoding, decoding, calibration, and evaluation
|   |-- decoder.py                 # CellGene/GlobalGene definitions and 20-feature decoder
|   |-- evaluator.py               # Decoder + surrogate evaluation wrapper
|   |-- calibrate.py               # Optional calibration against baseline data
|   |-- calibration.json           # Released calibration parameters
|   `-- README.md                  # Module-level notes
|-- surrogate/                     # Surrogate feature pipeline and inference assets
|   |-- feature_pipeline.py        # Builds normalized 20-feature surrogate inputs
|   |-- predict_energy.py          # Loads scalers/models and predicts energy targets
|   |-- data/                      # Demo dataset, scalers, metrics, and feature importance
|   |-- models/                    # XGBoost model JSON files for three energy targets
|   `-- README.md                  # Surrogate module documentation
|-- RawData/                       # Released geospatial source layers
|   |-- DATA_SOURCES.md            # Source and provenance notes for public GPKG files
|   `-- gpkg/                      # Xinwu buildings, roads, transit, land-use, water, and grid layers
|-- RealDis/                       # Real-block matching against representative solutions
|   |-- dataset.csv.example        # Expected schema for user-prepared local matching data
|   |-- match_reals.py             # Top-N real-block similarity matcher
|   `-- README.md                  # RealDis usage notes
|-- data/                          # Release inventory and checksums
|   |-- README.md                  # Data-release summary
|   |-- catalog.yaml               # Machine-readable release catalog
|   |-- data_dictionary.md         # Field-level data notes
|   `-- rawdata-gpkg.sha256        # Checksums for released geospatial layers
|-- docs/                          # Reproducibility and usage notes
|   |-- reproducibility.md
|   `-- usage.md
|-- activate_surrogate.sh          # Convenience shell activation helper
|-- CITATION.cff                   # GitHub citation metadata
|-- LICENSE                        # MIT License
|-- requirements.txt               # Python dependency list
`-- README.md                      # Project documentation
```

## Dependencies & Installation

This public release runs the surrogate-assisted optimization workflow with Python. EnergyPlus `24.1.0` is the simulation engine used to generate the underlying building-energy labels, and is linked here for provenance; it is not required for the lightweight optimization and analysis commands unless you rebuild the upstream simulation data.

1. **Clone the repository**

```bash
git clone https://github.com/haolinwang1313/Co-Opt-of-UM-and-ED-via-GA.git
cd Co-Opt-of-UM-and-ED-via-GA
```

2. **Create a virtual environment and install Python dependencies**

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3. **Install optional system dependencies**

Some geospatial Python packages may require native libraries such as `GDAL`, `PROJ`, and `GEOS`, depending on your platform and package source.

## Usage

Run all commands from the repository root. The released JSON, GPKG, and surrogate assets are sufficient for lightweight inspection, smoke tests, and representative-solution analysis.

### 1. Run quick checks

```bash
python -m compileall -q CoOpt GeneGenera RealDis surrogate
python CoOpt/nsga_small.py
```

### 2. Analyze released optimization outputs

```bash
python CoOpt/analyze_frontier.py
python CoOpt/reconstruct_representatives.py
```

### 3. Predict energy demand with the surrogate

```bash
python surrogate/predict_energy.py --input path/to/features.csv --output pred.csv
```

### 4. Match optimized representatives to real blocks

Prepare `RealDis/dataset.csv` using the schema in `RealDis/dataset.csv.example`, then run:

```bash
python RealDis/match_reals.py
```

The full `RealDis/dataset.csv` is user-provided because real-block matching depends on locally licensed 250 m data.

## Data Availability

See `data/README.md`, `data/catalog.yaml`, and `RawData/DATA_SOURCES.md` for the released data inventory, source notes, and checksums. This release includes public geospatial layers in `RawData/gpkg/`, surrogate tabular assets in `surrogate/data/`, model files in `surrogate/models/`, and optimization result JSON files in `CoOpt/results/`.

## Citation

If this repository or workflow is useful in your research, please cite:

```bibtex
@article{wang2026surrogate,
  title={A surrogate-assisted framework for district-scale urban morphology optimization toward reduced building energy demand},
  author={Wang, Haolin and Wu, Zhi and Gu, Wei and Liu, Pengxiang and Sun, Qirun and Wang, Wei},
  journal={Applied Energy},
  volume={422},
  pages={128294},
  year={2026},
  publisher={Elsevier},
  doi={10.1016/j.apenergy.2026.128294},
  url={https://doi.org/10.1016/j.apenergy.2026.128294}
}
```

## License

This project is licensed under the [MIT License](LICENSE).

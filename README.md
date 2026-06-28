# A surrogate-assisted framework for district-scale urban morphology optimization toward reduced building energy demand

![License](https://img.shields.io/badge/license-MIT-blue)
![Journal](https://img.shields.io/badge/journal-Applied%20Energy-blue)

This repository accompanies the paper **"A surrogate-assisted framework for district-scale urban morphology optimization toward reduced building energy demand"**, published in *Applied Energy*. It contains public research code, released geospatial data, surrogate assets, optimization outputs, and reproducibility notes for the district-scale urban morphology optimization workflow.

## Overview

The workflow combines gene-based urban morphology encoding, surrogate-assisted building-energy evaluation, NSGA-II optimization, representative-solution reconstruction, and matching against real urban blocks. The repository is intended as a paper-companion archive for inspecting the released workflow and rerunning lightweight checks or selected analysis steps.

## Repository Structure

```text
CoOpt/        NSGA-II optimization, analysis scripts, and released result JSON files
GeneGenera/   Gene representation, decoder, calibration, and evaluator code
RawData/      Released geospatial source layers for the Xinwu study area
RealDis/      Real-block matching script and input schema example
surrogate/    Surrogate feature pipeline, prediction code, metrics, and model JSON files
data/         Release data inventory and checksum manifests
docs/         Usage and reproducibility notes
```

## Dependencies & Installation

Python 3.9 or newer is recommended.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

## Usage

Run commands from the repository root.

```bash
python CoOpt/nsga_small.py
python CoOpt/analyze_frontier.py
python CoOpt/reconstruct_representatives.py
python RealDis/match_reals.py
```

`RealDis/match_reals.py` expects a local `RealDis/dataset.csv` matching the schema in `RealDis/dataset.csv.example`.

## Data Availability

See `data/README.md` and `data/catalog.yaml` for the released data inventory. Source notes for geospatial layers are provided in `RawData/DATA_SOURCES.md`.

## Citation

If this repository is useful in your research, please cite the associated paper. DOI, publication year, and author metadata should be filled once the final bibliographic record is available.

## License

Code and documentation are released under the MIT License. See `LICENSE`.

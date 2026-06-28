# A surrogate-assisted framework for district-scale urban morphology optimization toward reduced building energy demand

![License](https://img.shields.io/badge/license-MIT-blue)
![Journal](https://img.shields.io/badge/journal-Applied%20Energy-blue)
![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.apenergy.2026.128294-blue)

This repository accompanies the paper **"A surrogate-assisted framework for district-scale urban morphology optimization toward reduced building energy demand"**, published in *Applied Energy*, Volume 422, 1 November 2026, Article 128294. It contains public research code, released geospatial data, surrogate assets, optimization outputs, and reproducibility notes for the district-scale urban morphology optimization workflow.

## Paper

- Journal: *Applied Energy*
- Volume: 422
- Publication date: 1 November 2026
- Article number: 128294
- DOI: [10.1016/j.apenergy.2026.128294](https://doi.org/10.1016/j.apenergy.2026.128294)
- Authors: Haolin Wang, Zhi Wu, Wei Gu, Pengxiang Liu, Qirun Sun, and Wei Wang

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

If this repository is useful in your research, please cite the associated paper:

```bibtex
@article{wang2026surrogate,
  title = {A surrogate-assisted framework for district-scale urban morphology optimization toward reduced building energy demand},
  author = {Wang, Haolin and Wu, Zhi and Gu, Wei and Liu, Pengxiang and Sun, Qirun and Wang, Wei},
  journal = {Applied Energy},
  volume = {422},
  pages = {128294},
  year = {2026},
  doi = {10.1016/j.apenergy.2026.128294}
}
```

## License

Code and documentation are released under the MIT License. See `LICENSE`.

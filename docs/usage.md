# Usage

Run all commands from the repository root.

## Quick checks

```bash
python -m compileall -q CoOpt GeneGenera RealDis surrogate
python CoOpt/nsga_small.py
```

## Analysis entry points

```bash
python CoOpt/analyze_frontier.py
python CoOpt/reconstruct_representatives.py
```

## Real-block matching

Prepare `RealDis/dataset.csv` using the schema in `RealDis/dataset.csv.example`, then run:

```bash
python RealDis/match_reals.py
```

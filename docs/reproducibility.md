# Reproducibility Notes

This release preserves the public code paths and released result files needed to inspect the workflow described in the associated paper.

## Environment

Use Python 3.9 or newer and install dependencies from `requirements.txt`.

## Workflow outline

1. Inspect or modify scenario ranges in `CoOpt/config.py`.
2. Run a lightweight optimization smoke test with `python CoOpt/nsga_small.py`.
3. Analyze released frontiers with `python CoOpt/analyze_frontier.py`.
4. Reconstruct representative solution grids with `python CoOpt/reconstruct_representatives.py`.
5. Prepare a local `RealDis/dataset.csv` and run `python RealDis/match_reals.py` for real-block matching.

Full reruns can be computationally heavier than these smoke checks and may require study-specific local data preparation.

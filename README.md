# Datathon 2026 Round 1

This repository contains the final working codebase for our Round 1 submission:

- Part 2: EDA notebooks and curated export pipeline
- Part 3: baseline, benchmark, and final meta-ensemble forecasting pipeline
- Paper: NeurIPS-style write-up and figures

The repository is intentionally script-first:

- use notebooks for exploration and explanation
- use Python scripts to rebuild the final deliverables

## Quick Start

### 1. Create an environment

Use `venv` + `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Rebuild the final submission

```bash
python scripts/run_part3_pipeline.py
```

This writes:

- canonical upload file: `dataset/submission.csv`
- candidate submissions: `dataset/submissions/`
- appendix/modeling figures: `figures/modeling/` and `paper/neurips/appendix/`

### 3. Rebuild Part 2 final figures

```bash
python scripts/run_part2_pipeline.py
```

This executes the Part 2 notebooks and refreshes:

- source figure folders under `figures/`
- final release set under `figures/final_figures/`

## Canonical Entry Points

- Part 2 export pipeline: `scripts/run_part2_pipeline.py`
- Part 3 submission pipeline: `scripts/run_part3_pipeline.py`
- Final Kaggle interpretability notebook:
  - `notebooks/part3_modeling/03_final_meta_fine_horizon_interpretability.ipynb`

## Current Submission Logic

The promoted final submission candidate is controlled in:

- `models/final_meta_regime_ensemble.py`

Specifically:

- `INTERNAL_CV_WINNER_NAME`
- `FINAL_SUBMISSION_CANDIDATE_NAME`

At the moment, the code promotes:

- internal CV winner: `meta_horizon_ridge`
- final submission candidate: `meta_horizon_nnls`

If you want to switch the promoted Kaggle line, change `FINAL_SUBMISSION_CANDIDATE_NAME` and rerun:

```bash
python scripts/run_part3_pipeline.py
```

## Repository Layout

- `dataset/`
  - official raw tables
  - `submission.csv`: canonical upload file
  - `submissions/`: saved candidate outputs
- `models/`
  - reusable forecasting code
  - baseline, shape-medium benchmark, final meta-ensemble
- `scripts/`
  - reproducible rebuild entry points
- `notebooks/`
  - EDA and interpretability notebooks
- `figures/`
  - notebook-generated source figures and final exports
- `paper/`
  - final report source and compiled PDF

## Main Outputs

- Final Kaggle upload file:
  - `dataset/submission.csv`
- Final submission candidates:
  - `dataset/submissions/meta_horizon_nnls.csv`
  - `dataset/submissions/meta_fine_horizon_nnls.csv`
  - `dataset/submissions/meta_horizon_ridge.csv`
- Shape benchmark:
  - `dataset/submissions/final_internal_bottomup_shape_medium.csv`
- Part 2 release figures:
  - `figures/final_figures/`
- Modeling appendix figures:
  - `figures/modeling/`

## Notebook Coverage

- `notebooks/part1_mcq/`
- `notebooks/part2_eda/`
- `notebooks/part3_modeling/`

Part 2 notebooks build charts directly from the raw competition dataset.

Part 3 notebooks explain:

- baseline line
- shape-medium benchmark
- final meta-model interpretability

## Validation

Lightweight validation:

```bash
python -m py_compile models/*.py scripts/*.py
```

Full final submission rebuild:

```bash
python scripts/run_part3_pipeline.py
```

Full Part 2 figure rebuild:

```bash
python scripts/run_part2_pipeline.py
```

## Optional Notebook Kernel

If you want a dedicated Jupyter kernel:

```bash
python -m ipykernel install --user --name datathon --display-name "Python (datathon)"
```

## Notes

- The codebase no longer depends on `eda_results/`.
- The final interpretability notebook rebuilds SHAP summaries directly from model code.
- `paper/neurips/main.tex` and `paper/neurips/main.pdf` are kept as the paper source of record.

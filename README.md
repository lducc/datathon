# Datathon 2026 Round 1

This repo is organized around one final submission decision:

- internal anchor: `internal_bottomup_baseline`
- benchmark correction line: `final_shape_medium`
- conservative meta ablation: `meta_static_nnls`
- internal CV winner: `meta_horizon_ridge`
- final submission model: `meta_fine_horizon_nnls`

## Final decision

- Canonical final submission model: `meta_fine_horizon_nnls`
- Why: internal CV winner did not transfer well to leaderboard behavior on the long forecast horizon, while `meta_fine_horizon_nnls` stayed compliant, interpretable, and performed better in practical submission use.
- Paper shoutout models:
  - `internal_bottomup_baseline`
  - `final_shape_medium`
  - `meta_static_nnls`
  - `meta_fine_horizon_nnls`

## Canonical entrypoints

- Part 2 figures: `python scripts/run_part2_pipeline.py`
- Part 3 package build: `python scripts/run_part3_pipeline.py`

## Canonical artifacts

- Canonical Kaggle upload file: `dataset/submission.csv`
- Final submission artifact: `dataset/submissions/meta_fine_horizon_nnls.csv`
- Internal CV winner artifact: `dataset/submissions/meta_horizon_ridge.csv`
- Shape-medium benchmark artifact: `dataset/submissions/final_internal_bottomup_shape_medium.csv`
- Part 2 final figures: `figures/final_figures/`
- Final paper PDF: `paper/neurips/main.pdf`
- Final paper source: `paper/neurips/main.tex`
- Final interpretability notebook:
  - `notebooks/part3_modeling/03_final_meta_fine_horizon_interpretability.ipynb`

## Environment

- If `uv` is available: `uv sync`
- Required runtime packages are declared in `pyproject.toml`
- Part 2 runner no longer depends on a machine-specific Python path

## Repo map

- `dataset/`
  - official raw tables
  - `submission.csv` canonical final upload file
  - `submissions/` saved candidate outputs
- `models/`
  - baseline, shape-medium, and final meta-ensemble code
- `scripts/`
  - canonical rebuild entrypoints
- `notebooks/`
  - Part 1, Part 2, Part 3 narrative and analysis notebooks
- `paper/neurips/`
  - final write-up source and compiled PDF

## Final notebooks

- `notebooks/part1_mcq/01_mqa_mcq_master.ipynb`
- `notebooks/part2_eda/01_problem_overview.ipynb`
- `notebooks/part2_eda/02_promotion_and_demand.ipynb`
- `notebooks/part2_eda/03_inventory_and_operations.ipynb`
- `notebooks/part2_eda/04_customer_and_returns.ipynb`
- `notebooks/part2_eda/05_final_figures.ipynb`
- `notebooks/part3_modeling/01_internal_bottomup_baseline.ipynb`
- `notebooks/part3_modeling/02_final_shape_medium_submission.ipynb`
- `notebooks/part3_modeling/03_final_meta_fine_horizon_interpretability.ipynb`

## Notes

- Raw dataset files under `dataset/` stay unchanged.
- Part 2 figures are generated directly from official raw tables.
- The final submission line is meta-ensemble based, so Part 3 scripts are the canonical way to regenerate the repo’s final `submission.csv`.
- The final interpretability notebook rebuilds SHAP summaries directly from code and does not depend on `eda_results/`.

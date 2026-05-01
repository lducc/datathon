# Part 3 Modeling

Canonical modeling notebooks:

- `01_internal_bottomup_baseline.ipynb`
- `02_final_shape_medium_submission.ipynb`
- `03_final_meta_fine_horizon_interpretability.ipynb`

These notebooks show the recreated baseline and the shape-medium benchmark line.
The canonical final submission model for the repo is `meta_fine_horizon_nnls`, built by `models/final_meta_regime_ensemble.py` and promoted by `scripts/run_part3_pipeline.py`.
The final interpretability notebook rebuilds SHAP summaries directly from the model code and no longer depends on `eda_results/`.

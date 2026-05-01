# Final Model Manifest

## Final submission decision

- Final submission model: `meta_fine_horizon_nnls`
- Canonical upload file: `dataset/submission.csv`
- Saved submission artifact: `dataset/submissions/meta_fine_horizon_nnls.csv`

## Model roles in this repo

1. `internal_bottomup_baseline`
   - Role: anchor baseline
   - Purpose: transparent internal commercial profile without meta stacking
   - Artifact: `dataset/submissions/internal_bottomup_baseline.csv`

2. `final_shape_medium`
   - Role: benchmark correction line
   - Purpose: improve the baseline by adjusting daily shape over the horizon
   - Artifact: `dataset/submissions/final_internal_bottomup_shape_medium.csv`

3. `meta_static_nnls`
   - Role: conservative explainable meta ablation
   - Purpose: clean non-negative blend with fixed weights
   - Artifact: `dataset/submissions/meta_static_nnls.csv`

4. `meta_horizon_ridge`
   - Role: internal CV winner
   - Purpose: strongest model under internal out-of-fold objective
   - Artifact: `dataset/submissions/meta_horizon_ridge.csv`
   - Note: retained for benchmarking only; not the final submission line

5. `meta_fine_horizon_nnls`
   - Role: final submission model
   - Purpose: leaderboard-aligned compliant meta ensemble with finer horizon routing
   - Artifact: `dataset/submissions/meta_fine_horizon_nnls.csv`

## Paper shoutout models

The paper should mention only the lines that help explain the final decision:

- `internal_bottomup_baseline`
- `final_shape_medium`
- `meta_static_nnls`
- `meta_fine_horizon_nnls`

## Final interpretability entrypoint

- notebook: `notebooks/part3_modeling/03_final_meta_fine_horizon_interpretability.ipynb`
- runtime source: `models/final_meta_regime_ensemble.py`

## Decision rationale

- `meta_horizon_ridge` wins internal CV but does not transfer well enough to leaderboard behavior.
- `meta_fine_horizon_nnls` remains compliant, explainable, and routes weight differently across early and late horizon segments.
- That routing is better aligned with the project’s business story: near horizon should react to recent demand signals, while long horizon should lean more on component and regime structure.

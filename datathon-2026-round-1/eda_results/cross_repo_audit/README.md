# Cross-Repo Solution Audit

Date: 2026-04-27

## Scope

Compared models, submissions, and EDA outputs from:

- `/home/lducc/code/datathon/DATATHON-2026-r1`
- `/home/lducc/code/datathon/DATATHON2026`
- `/home/lducc/code/datathon/Datathon_Vin`
- `/home/lducc/code/datathon/datathon2026`
- Main target repo: `/home/lducc/code/datathon/datathon-2026-round-1`

## Notebook Execution Status

Full sweep result across these repos:

- Total notebooks found: 28
- Executed successfully: 20
- Failed: 8

Main failure causes:

- Missing dependencies (`xgboost`, `missingno`)
- File-path assumptions (`dataset/sales.csv` not present in current cwd)
- One notebook permission issue

## External Submission Validity

Raw external submissions were checked against competition schema:

- Required columns: `Date, Revenue, COGS`
- Exact date index match with sample submission (548 rows)
- No nulls
- `Revenue > 0`
- `COGS < Revenue`

Important finding: most external submissions are schema-compatible but violate `COGS < Revenue` on many rows.

See:

- `submission_validity_audit.csv`
- `candidate_fix_and_blend_summary.csv`
- `candidate_compliance_checks.csv`

## Fixed Candidate Files

Generated compliant candidates in:

- `/home/lducc/code/datathon/datathon-2026-round-1/dataset/cross_repo_candidates`

Files:

- `ours_current_fixed.csv`
- `r1_ensemble_fixed.csv`
- `r1_lgb_fixed.csv`
- `r1_residual_ensemble_fixed.csv`
- `r1_opt_baseline_fixed.csv`
- `datathon2026_submission_fixed.csv`
- `submission_blend_A.csv`
- `submission_blend_B.csv`

All 8 pass schema and hard constraints after minimal fix/clamp.

## EDA Insight Validation (Main Repo Recompute)

Recomputed directly from main raw tables to confirm cross-repo claims:

- Promo usage percent: 38.6635%
- Top cancelled payment method: `credit_card` (28452)
- Top Streetwear return reason: `wrong_size` (7626)
- Top age group by avg orders/customer: `55+`
- Top region by revenue proxy: `East`
- Overall top return reason: `wrong_size` (13967)

One notable discrepancy vs some external notebooks:

- Top segment by margin in this recompute is `Trendy`, not `Standard`

See:

- `eda_claims_recomputed_main_repo.csv`
- `eda_cancelled_payment_counts.csv`
- `eda_streetwear_return_reason_counts.csv`
- `eda_age_group_orders_per_customer.csv`
- `eda_region_revenue_proxy.csv`
- `eda_segment_margin.csv`

## Worth-Trying Submission Shortlist

Based on (1) model credibility from source repo, (2) correction severity, (3) growth plausibility vs 2022, (4) diversity:

1. Safe:
   - `ours_current_fixed.csv`
   - Lowest risk, closest to current validated pipeline

2. Balanced:
   - `r1_residual_ensemble_fixed.csv`
   - Mean level close to 2022, good diversity vs current

3. Medium-risk blend:
   - `submission_blend_B.csv`
   - Cross-repo blend with moderate uplift profile

4. Aggressive:
   - `r1_ensemble_fixed.csv` or `submission_blend_A.csv`
   - Higher upside if post-2022 growth is strong, but much higher forecast level

## Recommendation

Use quick leaderboard A/B order:

1. `ours_current_fixed.csv`
2. `r1_residual_ensemble_fixed.csv`
3. `submission_blend_B.csv`
4. `submission_blend_A.csv`
5. `r1_ensemble_fixed.csv`

If rank worsens with 2 and 3, external models are overfitting their own validation regime and should not be merged further.

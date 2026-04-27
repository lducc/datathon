# Team Dataset + Model Story (Detailed)

Generated on: 2026-04-27 13:42 UTC

## 1. Executive Summary

1. Revenue concentration and profitability moved unevenly over time: peak revenue year was 2016 (2,104,640,678), lowest was 2012 (741,497,748); full-period CAGR is 4.66%.
2. Demand is strongly cyclical by month and weekday: highest weekday revenue is Wed (4,680,065/day) vs lowest Sat (3,906,581/day).
3. Promotion pressure and return pressure can erode margin quickly: highest-discount bucket (20%+) has margin -24.56% vs 0% at 19.96%.
4. Operational leakage is measurable: highest cancellation payment method is cod (16.00% cancel rate), and highest-risk order source is referral (9.34%).
5. Model side: `core_long` remains best (Rev MAE 988,549); external submissions often fail constraints and require correction before use.

## 2. Demand + Financial Story

- See figures:
  - `fig01_yearly_financials.png`
  - `fig02_monthly_heatmap.png`
  - `fig03_weekday_pattern.png`

- What the numbers say:
  - Total revenue over whole history: 16,430,476,586
  - Total COGS over whole history: 14,163,450,519
  - Total gross profit over whole history: 2,267,026,066
  - Best year gross margin: 20.77% ; worst year gross margin: 9.77%
  - Highest monthly demand concentration implies inventory and staffing should not be flat through the year.

- Business interpretation:
  - Budgeting and procurement should follow seasonal intensity, not annual averages.
  - Weekly playbooks should prioritize high-yield weekdays for premium offers and reserve low-yield days for clearance and reactivation.

## 3. Product + Pricing + Promo Story

- See figures:
  - `fig04_category_scatter.png`
  - `fig05_segment_profitability.png`
  - `fig06_discount_buckets.png`
  - `fig07_promo_type.png`

- What the numbers say:
  - Top revenue category: Streetwear (12,558,477,099 net revenue, 9.28% margin).
  - Weakest margin category: Casual (7.66% margin).
  - Discount effectiveness drops in higher discount buckets: margins compress materially as discount rate rises.

- Business interpretation:
  - Category-level margin management should be explicit in campaign planning.
  - High-discount campaigns should need a stricter approval gate and a post-mortem target (incremental profit, not just revenue lift).

## 4. Returns + Cancellation + Service Leakage

- See figures:
  - `fig08_returns_reasons.png`
  - `fig09_cancellation_rates.png`
  - `fig13_inventory_stress.png`

- What the numbers say:
  - Top return reason has structural dominance (wrong size), indicating product/fit or expectation mismatch.
  - Cancellation risk differs sharply across payment and source channels.
  - Peak stockout-rate month: 2015-11 at 72.54%.
  - Lowest fill-rate month: 2013-12 at 94.38%.

- Business interpretation:
  - Reduce return leakage with sizing/fit interventions and category-specific PDP copy adjustments.
  - Target payment/source combinations with highest cancellation risk first for checkout friction fixes.
  - Use monthly stockout and fill-rate thresholds as operational guardrails before campaign launches.

## 5. Customer + Channel + Geography Story

- See figures:
  - `fig10_customer_cohorts.png`
  - `fig11_age_group_orders.png`
  - `fig12_region_trend.png`
  - `fig14_web_traffic_corr.png`

- What the numbers say:
  - Peak acquisition month: 2022-12 with 1,883 new customers.
  - Highest orders/customer segment: 55+ at 7.27.
  - Latest-year top region: East (520,816,706).
  - Sessions to revenue correlation: 0.321; sessions to orders correlation: 0.191.

- Business interpretation:
  - Lifecycle plans should be age-group aware, with distinct retention programs for high-frequency cohorts.
  - Regional inventory and media budgets should follow region-specific revenue trajectories.
  - Traffic quality matters as much as traffic volume; session growth alone is not enough.

## 6. Model Story + Cross-Repo Validity

- See figures:
  - `fig15_feature_set_performance.png`
  - `fig16_feature_importance_top20.png`
  - `fig17_candidate_comparison.png`
  - `fig18_candidate_fix_impact.png`

- What the numbers say:
  - Best feature set: `core_long` with Rev MAE 988,549, COGS MAE 869,556.
  - Worst tested set (`core_long_plus_biz730`) is +10.80% worse on Revenue MAE.
  - External candidates often required substantial correction for `COGS < Revenue` before becoming valid.
  - Safe candidate profile (ours): revenue mean delta vs 2022 = -1.76%.
  - Balanced external candidate (`r1_residual_ensemble_fixed.csv`) delta vs 2022 = 0.19%.
  - Most aggressive candidate in pack: `r1_ensemble_fixed.csv` at +38.50% vs 2022 mean.

- Business interpretation:
  - Model governance should include hard output constraints and plausibility checks, not just fold metrics.
  - Candidate choice can be portfolio-managed: safe, balanced, aggressive, depending on risk appetite.

## 7. Team Discussion Checklist

1. Which categories can sustain discounting without damaging gross profit?
2. Which payment/source pairs should be prioritized for cancellation reduction in next sprint?
3. What stockout threshold should trigger pre-emptive replenishment before major campaigns?
4. Should planning use the safe forecast profile or a balanced profile for Q3/Q4 decisions?
5. Which two metrics become weekly operating KPIs for leakage control (returns + cancellation + stockout)?

## 8. Output Inventory

- Figures: `eda_results/team_story/figures/fig01_...fig18_...`
- Tables: `eda_results/team_story/tables/*.csv`
- This report: `eda_results/team_story/team_story_detailed.md`

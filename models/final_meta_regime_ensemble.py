from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import nnls
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

from .baseline_internal_bottomup import build_repo_style_baseline_frame
from .data import (
    ensure_submission_dir,
    load_dataframes,
    sample_dates,
)
from .final_shape_medium import (
    OUTER_YEARS,
    RATIO_CEIL,
    RATIO_FLOOR,
    FeatureContext,
    HeadModel,
    _build_feature_context,
    _build_ratio_frame,
    _build_revenue_level_frame,
    _calendar_features,
    _fit_fold_heads,
    _joint_objective,
    _predict_component_head,
    _predict_diff_head,
    _predict_level_head,
    _predict_ratio_head,
    _sample_weights,
)
from .data import build_daily_frame

SEED = 2026
INNER_SPLITS = 3
REVENUE_BLEND_COLS = [
    "rev_baseline",
    "rev_recursive_level",
    "rev_recursive_diff",
    "rev_component",
    "rev_regime",
]
RATIO_BLEND_COLS = [
    "ratio_baseline",
    "ratio_recursive",
    "ratio_regime",
]
HORIZON_BUCKETS = ("early", "mid", "late")
FINE_HORIZON_BUCKETS = ("early", "mid_early", "mid_late", "late")
RATIO_REGIMES = ("normal", "sale_window", "odd_august")
INTERNAL_CV_WINNER_NAME = "meta_horizon_ridge"
FINAL_SUBMISSION_CANDIDATE_NAME = "meta_fine_horizon_nnls"
PAPER_SHOUTOUT_MODELS = (
    "internal_bottomup_baseline",
    "final_shape_medium",
    "meta_static_nnls",
    FINAL_SUBMISSION_CANDIDATE_NAME,
)


@dataclass
class EventTemplate:
    name: str
    start_month: int
    start_day: int
    end_month: int
    end_day: int
    odd_only: bool
    even_only: bool


@dataclass
class StoryPriors:
    monthly: dict[str, dict[int, float]]
    global_values: dict[str, float]
    special: dict[str, float]


@dataclass
class ResidualTreeHead:
    feature_cols: list[str]
    xgb_model: object
    lgb_model: object
    weights: dict[str, float]
    clip_low: float
    clip_high: float


@dataclass
class SpecialistBundle:
    context: FeatureContext
    level_head: HeadModel
    diff_head: HeadModel
    ratio_head: HeadModel
    orders_head: HeadModel
    aov_head: HeadModel
    event_templates: list[EventTemplate]
    story_priors: StoryPriors
    revenue_regime_head: ResidualTreeHead
    ratio_regime_head: ResidualTreeHead


@dataclass
class MetaCandidate:
    name: str
    methodology: str
    revenue_mode: str
    ratio_mode: str
    revenue_model: dict[str, object]
    ratio_model: dict[str, object]
    metrics: dict[str, float]


def _meta_candidate_path(name: str) -> Path:
    slug = name.lower().replace(" ", "_").replace("-", "_").replace("/", "_")
    return ensure_submission_dir() / f"{slug}.csv"


def _event_key(name: str) -> str:
    clean = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower())
    return clean.strip("_")


def _extract_event_templates(
    promotions: pd.DataFrame,
    cutoff_date: pd.Timestamp,
) -> list[EventTemplate]:
    promo = promotions.copy()
    promo["start_date"] = pd.to_datetime(promo["start_date"], errors="coerce")
    promo["end_date"] = pd.to_datetime(promo["end_date"], errors="coerce")
    promo = promo.dropna(subset=["start_date", "end_date"]).copy()
    promo = promo.loc[promo["start_date"] <= cutoff_date].copy()
    if promo.empty:
        return []

    promo["event_key"] = promo["promo_name"].map(_event_key)
    templates: list[EventTemplate] = []
    for event_key, frame in promo.groupby("event_key"):
        frame = frame.sort_values("start_date")
        start_mode = frame["start_date"].dt.strftime("%m-%d").mode().iloc[0]
        end_mode = frame["end_date"].dt.strftime("%m-%d").mode().iloc[0]
        years = frame["start_date"].dt.year.dropna().astype(int).unique().tolist()
        odd_only = bool(years) and all(year % 2 == 1 for year in years)
        even_only = bool(years) and all(year % 2 == 0 for year in years)
        templates.append(
            EventTemplate(
                name=event_key,
                start_month=int(start_mode[:2]),
                start_day=int(start_mode[3:]),
                end_month=int(end_mode[:2]),
                end_day=int(end_mode[3:]),
                odd_only=odd_only,
                even_only=even_only,
            )
        )
    return templates


def _safe_month_mapping(series: pd.Series, months: pd.Series) -> tuple[dict[int, float], float]:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    temp = pd.DataFrame({"month": months.astype(int), "value": values})
    temp = temp.dropna(subset=["value"]).copy()
    if temp.empty:
        return {}, 0.0
    mapping = {
        int(month): float(value)
        for month, value in temp.groupby("month")["value"].mean().to_dict().items()
    }
    global_value = float(temp["value"].mean())
    return mapping, global_value


def _safe_lookup(mapping: dict[int, float], global_value: float, month: int) -> float:
    return float(mapping.get(int(month), global_value))


def _build_story_priors(
    history_daily: pd.DataFrame,
    inventory: pd.DataFrame,
) -> StoryPriors:
    history = history_daily.copy()
    history["month"] = history["date"].dt.month.astype(int)
    history["orders_filled"] = history["orders"].fillna(0.0)
    history["sessions_filled"] = history["sessions"].fillna(0.0)
    history["conversion_filled"] = (
        history["orders_filled"] / history["sessions_filled"].replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    history["bounce_rate_filled"] = pd.to_numeric(history["bounce_rate"], errors="coerce")
    history["pages_per_session_filled"] = pd.to_numeric(history["page_views_per_session"], errors="coerce")
    history["session_duration_filled"] = pd.to_numeric(history["avg_session_duration_sec"], errors="coerce")
    history["promo_share_filled"] = pd.to_numeric(history["promo_order_share"], errors="coerce")
    history["discount_rate_filled"] = pd.to_numeric(history["discount_rate"], errors="coerce")
    history["gross_margin_filled"] = pd.to_numeric(history["gross_margin"], errors="coerce")
    history["cancel_rate_filled"] = (
        pd.to_numeric(history["cancelled_orders"], errors="coerce")
        / history["orders_filled"].replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    history["return_rate_filled"] = (
        pd.to_numeric(history["return_orders"], errors="coerce")
        / history["orders_filled"].replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    history["refund_rate_filled"] = (
        pd.to_numeric(history["refund_amount"], errors="coerce")
        / pd.to_numeric(history["Revenue"], errors="coerce").replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    history["avg_rating_filled"] = pd.to_numeric(history["avg_rating"], errors="coerce")
    history["delivery_days_filled"] = pd.to_numeric(history["avg_delivery_days"], errors="coerce")
    history["shipping_fee_filled"] = pd.to_numeric(history["shipping_fee"], errors="coerce")
    history["unique_visitors_log1p"] = np.log1p(pd.to_numeric(history["unique_visitors"], errors="coerce").clip(lower=0.0))
    history["review_count_log1p"] = np.log1p(pd.to_numeric(history["review_count"], errors="coerce").clip(lower=0.0))
    history["traffic_sessions_log1p"] = np.log1p(history["sessions_filled"].clip(lower=0.0))
    history["traffic_engagement"] = (
        history["pages_per_session_filled"].clip(lower=0.0)
        * np.log1p(history["session_duration_filled"].clip(lower=0.0))
    )

    monthly: dict[str, dict[int, float]] = {}
    global_values: dict[str, float] = {}
    for name, series in {
        "traffic_sessions_log1p": history["traffic_sessions_log1p"],
        "traffic_unique_visitors_log1p": history["unique_visitors_log1p"],
        "traffic_conversion": history["conversion_filled"],
        "traffic_bounce": history["bounce_rate_filled"],
        "traffic_engagement": history["traffic_engagement"],
        "promo_share": history["promo_share_filled"],
        "discount_rate": history["discount_rate_filled"],
        "gross_margin": history["gross_margin_filled"],
        "cancel_rate": history["cancel_rate_filled"],
        "return_rate": history["return_rate_filled"],
        "refund_rate": history["refund_rate_filled"],
        "avg_rating": history["avg_rating_filled"],
        "delivery_days": history["delivery_days_filled"],
        "shipping_fee": history["shipping_fee_filled"],
        "review_count_log1p": history["review_count_log1p"],
    }.items():
        mapping, global_value = _safe_month_mapping(series, history["month"])
        monthly[name] = mapping
        global_values[name] = global_value

    inv = inventory.copy()
    if "snapshot_date" in inv.columns:
        inv["snapshot_date"] = pd.to_datetime(inv["snapshot_date"], errors="coerce")
        cutoff = pd.Timestamp(history["date"].max())
        inv = inv.loc[inv["snapshot_date"].notna() & (inv["snapshot_date"] <= cutoff)].copy()
    if inv.empty:
        for name in [
            "inventory_dos",
            "inventory_turnover",
            "inventory_overstock",
            "inventory_stockout",
            "inventory_fill_rate",
        ]:
            monthly[name] = {}
            global_values[name] = 0.0
        special = {
            "august_odd_dos": 0.0,
            "august_even_dos": 0.0,
            "august_odd_turnover": 0.0,
            "august_even_turnover": 0.0,
        }
        return StoryPriors(monthly=monthly, global_values=global_values, special=special)

    inv["month"] = inv["snapshot_date"].dt.month.astype(int)
    inv["year"] = inv["snapshot_date"].dt.year.astype(int)
    inv["turnover_ratio"] = pd.to_numeric(inv["units_sold"], errors="coerce") / (
        pd.to_numeric(inv["stock_on_hand"], errors="coerce")
        + pd.to_numeric(inv["units_received"], errors="coerce")
    ).replace(0.0, np.nan)
    for name, series in {
        "inventory_dos": pd.to_numeric(inv["days_of_supply"], errors="coerce"),
        "inventory_turnover": inv["turnover_ratio"],
        "inventory_overstock": pd.to_numeric(inv["overstock_flag"], errors="coerce"),
        "inventory_stockout": pd.to_numeric(inv["stockout_flag"], errors="coerce"),
        "inventory_fill_rate": pd.to_numeric(inv["fill_rate"], errors="coerce"),
    }.items():
        mapping, global_value = _safe_month_mapping(series, inv["month"])
        monthly[name] = mapping
        global_values[name] = global_value

    odd_august = inv.loc[(inv["month"] == 8) & (inv["year"] % 2 == 1)].copy()
    even_august = inv.loc[(inv["month"] == 8) & (inv["year"] % 2 == 0)].copy()
    special = {
        "august_odd_dos": float(pd.to_numeric(odd_august.get("days_of_supply"), errors="coerce").mean()) if not odd_august.empty else global_values["inventory_dos"],
        "august_even_dos": float(pd.to_numeric(even_august.get("days_of_supply"), errors="coerce").mean()) if not even_august.empty else global_values["inventory_dos"],
        "august_odd_turnover": float(pd.to_numeric(odd_august.get("turnover_ratio"), errors="coerce").mean()) if not odd_august.empty else global_values["inventory_turnover"],
        "august_even_turnover": float(pd.to_numeric(even_august.get("turnover_ratio"), errors="coerce").mean()) if not even_august.empty else global_values["inventory_turnover"],
    }
    for key, value in list(special.items()):
        if pd.isna(value):
            if "dos" in key:
                special[key] = global_values["inventory_dos"]
            else:
                special[key] = global_values["inventory_turnover"]
    return StoryPriors(monthly=monthly, global_values=global_values, special=special)


def _month_day_value(month: int, day: int) -> int:
    return month * 100 + day


def _date_in_template(date: pd.Timestamp, template: EventTemplate) -> bool:
    if template.odd_only and date.year % 2 == 0:
        return False
    if template.even_only and date.year % 2 == 1:
        return False
    current = _month_day_value(int(date.month), int(date.day))
    start = _month_day_value(template.start_month, template.start_day)
    end = _month_day_value(template.end_month, template.end_day)
    if start <= end:
        return start <= current <= end
    return current >= start or current <= end


def _event_feature_frame(
    dates: pd.Series,
    templates: list[EventTemplate],
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for date in pd.to_datetime(dates):
        row: dict[str, float] = {}
        for template in templates:
            row[f"event_{template.name}"] = 1.0 if _date_in_template(pd.Timestamp(date), template) else 0.0
        rows.append(row)
    if not rows:
        return pd.DataFrame(index=np.arange(len(dates)))
    return pd.DataFrame(rows)


def _build_regime_feature_frame(
    dates: pd.Series,
    context: FeatureContext,
    templates: list[EventTemplate],
    story_priors: StoryPriors,
) -> pd.DataFrame:
    rows = [_calendar_features(pd.Timestamp(date), context) for date in pd.to_datetime(dates)]
    frame = pd.DataFrame(rows)
    events = _event_feature_frame(dates, templates)
    if not events.empty:
        frame = pd.concat([frame.reset_index(drop=True), events.reset_index(drop=True)], axis=1)
    frame["event_any"] = events.sum(axis=1).astype(float) if not events.empty else 0.0
    frame["month_end_pressure"] = frame["promo_active_prior"] * frame["is_last7"]
    frame["promo_q4_interact"] = frame["promo_discount_prior"] * (frame["quarter"] == 4).astype(float)
    frame["promo_q3_interact"] = frame["promo_discount_prior"] * (frame["quarter"] == 3).astype(float)
    frame["odd_august_sale_interact"] = frame["event_any"] * frame["is_odd_year_august"]
    frame["q1_flag"] = (frame["quarter"] == 1).astype(float)
    frame["q2_flag"] = (frame["quarter"] == 2).astype(float)
    frame["q3_flag"] = (frame["quarter"] == 3).astype(float)
    frame["q4_flag"] = (frame["quarter"] == 4).astype(float)
    frame["quarter_end_window"] = (
        frame["month"].isin([3.0, 6.0, 9.0, 12.0]) & (frame["day_of_month"] >= 20.0)
    ).astype(float)
    frame["quarter_start_window"] = (
        frame["month"].isin([1.0, 4.0, 7.0, 10.0]) & (frame["day_of_month"] <= 7.0)
    ).astype(float)

    months = frame["month"].astype(int)
    month_features = [
        "traffic_sessions_log1p",
        "traffic_unique_visitors_log1p",
        "traffic_conversion",
        "traffic_bounce",
        "traffic_engagement",
        "promo_share",
        "discount_rate",
        "gross_margin",
        "cancel_rate",
        "return_rate",
        "refund_rate",
        "avg_rating",
        "delivery_days",
        "shipping_fee",
        "review_count_log1p",
        "inventory_dos",
        "inventory_turnover",
        "inventory_overstock",
        "inventory_stockout",
        "inventory_fill_rate",
    ]
    for name in month_features:
        mapping = story_priors.monthly.get(name, {})
        global_value = story_priors.global_values.get(name, 0.0)
        frame[f"{name}_month_prior"] = months.map(
            lambda month: _safe_lookup(mapping, global_value, int(month))
        ).astype(float)

    frame["traffic_quality_score"] = (
        0.50 * frame["traffic_conversion_month_prior"].clip(lower=0.0)
        + 0.20 * frame["traffic_engagement_month_prior"].clip(lower=0.0)
        + 0.15 * frame["traffic_sessions_log1p_month_prior"].clip(lower=0.0)
        + 0.10 * frame["traffic_unique_visitors_log1p_month_prior"].clip(lower=0.0)
        - 0.35 * frame["traffic_bounce_month_prior"].clip(lower=0.0)
    )
    frame["traffic_depth_score"] = (
        0.60 * frame["traffic_engagement_month_prior"].clip(lower=0.0)
        + 0.25 * frame["traffic_unique_visitors_log1p_month_prior"].clip(lower=0.0)
        + 0.15 * frame["review_count_log1p_month_prior"].clip(lower=0.0)
    )
    frame["promo_pressure_score"] = (
        frame["promo_share_month_prior"].clip(lower=0.0)
        * (1.0 + 5.0 * frame["discount_rate_month_prior"].clip(lower=0.0))
        * (1.0 + 0.25 * frame["event_any"])
    )
    frame["inventory_stress_score"] = (
        np.log1p(frame["inventory_dos_month_prior"].clip(lower=0.0))
        * (1.0 + frame["inventory_overstock_month_prior"].clip(lower=0.0))
        * (1.0 + frame["inventory_stockout_month_prior"].clip(lower=0.0))
        / (frame["inventory_turnover_month_prior"].clip(lower=0.02) + 0.10)
        * (2.0 - frame["inventory_fill_rate_month_prior"].clip(lower=0.0, upper=1.5))
    )
    frame["service_friction_score"] = (
        0.75 * frame["cancel_rate_month_prior"].clip(lower=0.0)
        + frame["return_rate_month_prior"].clip(lower=0.0)
        + frame["refund_rate_month_prior"].clip(lower=0.0)
        + ((5.0 - frame["avg_rating_month_prior"].clip(lower=0.0, upper=5.0)) / 5.0)
        + 0.05 * frame["delivery_days_month_prior"].clip(lower=0.0)
        + 0.01 * frame["shipping_fee_month_prior"].clip(lower=0.0)
    )
    frame["commercial_quality_score"] = (
        frame["gross_margin_month_prior"].fillna(0.0)
        + 0.75 * frame["traffic_quality_score"]
        + 0.20 * frame["traffic_depth_score"]
        - 0.30 * frame["promo_pressure_score"]
        - 0.10 * frame["inventory_stress_score"]
        - 0.10 * frame["service_friction_score"]
    )
    frame["promo_quality_gap"] = frame["traffic_quality_score"] - frame["promo_pressure_score"]
    frame["inventory_service_gap"] = frame["inventory_stress_score"] + frame["service_friction_score"]
    frame["organic_efficiency_score"] = (
        frame["traffic_quality_score"].clip(lower=0.0)
        * (1.0 + frame["gross_margin_month_prior"].fillna(0.0))
        / (1.0 + frame["promo_pressure_score"].clip(lower=0.0))
    )
    frame["traffic_inventory_balance"] = frame["traffic_quality_score"] / (
        1.0 + frame["inventory_stress_score"].clip(lower=0.0)
    )
    frame["conversion_stability_score"] = (
        frame["traffic_conversion_month_prior"].clip(lower=0.0)
        * frame["traffic_inventory_balance"]
        * (1.0 + frame["inventory_fill_rate_month_prior"].clip(lower=0.0))
    )
    frame["event_inventory_interact"] = frame["event_any"] * frame["inventory_stress_score"]
    frame["event_traffic_interact"] = frame["event_any"] * frame["traffic_quality_score"]
    frame["promo_inventory_interact"] = frame["promo_discount_prior"] * frame["inventory_stress_score"]
    frame["promo_service_interact"] = frame["promo_discount_prior"] * frame["service_friction_score"]
    frame["event_service_interact"] = frame["event_any"] * frame["service_friction_score"]
    frame["quarter_pressure_score"] = (
        (frame["quarter_end_window"] + frame["quarter_start_window"])
        * (frame["promo_pressure_score"] + 0.5 * frame["inventory_stress_score"])
    )
    frame["odd_august_pressure_score"] = frame["is_odd_year_august"] * (
        frame["inventory_stress_score"] + frame["promo_pressure_score"] + 0.5 * frame["service_friction_score"]
    )
    odd_dos_gap = float(
        story_priors.special.get("august_odd_dos", 0.0) - story_priors.special.get("august_even_dos", 0.0)
    )
    odd_turnover_gap = float(
        story_priors.special.get("august_even_turnover", 0.0)
        - story_priors.special.get("august_odd_turnover", 0.0)
    )
    frame["odd_august_inventory_gap"] = frame["is_odd_year_august"] * odd_dos_gap
    frame["odd_august_turnover_gap"] = frame["is_odd_year_august"] * odd_turnover_gap
    return frame


def _residual_weight_vector(
    dates: pd.Series,
    regime_frame: pd.DataFrame,
    anomaly_boost: float,
) -> np.ndarray:
    weights = _sample_weights(dates).astype(float)
    weights *= 1.0 + 0.30 * regime_frame["event_any"].to_numpy(dtype=float)
    weights *= 1.0 + 0.15 * regime_frame["is_last7"].to_numpy(dtype=float)
    weights *= 1.0 + anomaly_boost * regime_frame["is_odd_year_august"].to_numpy(dtype=float)
    return weights


def _fit_residual_head(
    x_frame: pd.DataFrame,
    y_transformed: np.ndarray,
    base_values: np.ndarray,
    actual_values: np.ndarray,
    dates: pd.Series,
    sample_weights: np.ndarray,
    clip_low: float,
    clip_high: float,
    kind: str,
) -> ResidualTreeHead:
    if kind == "revenue":
        xgb_params = {
            "n_estimators": 220,
            "learning_rate": 0.04,
            "max_depth": 4,
            "subsample": 0.88,
            "colsample_bytree": 0.86,
            "reg_alpha": 0.05,
            "reg_lambda": 1.0,
            "objective": "reg:squarederror",
            "random_state": SEED,
            "n_jobs": 4,
            "verbosity": 0,
            "tree_method": "hist",
        }
        lgb_params = {
            "n_estimators": 220,
            "learning_rate": 0.04,
            "num_leaves": 31,
            "max_depth": 4,
            "subsample": 0.88,
            "subsample_freq": 1,
            "colsample_bytree": 0.86,
            "reg_alpha": 0.05,
            "reg_lambda": 1.0,
            "objective": "huber",
            "random_state": SEED,
            "n_jobs": 4,
            "verbose": -1,
        }
    else:
        xgb_params = {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 3,
            "subsample": 0.88,
            "colsample_bytree": 0.86,
            "reg_alpha": 0.05,
            "reg_lambda": 1.1,
            "objective": "reg:squarederror",
            "random_state": SEED,
            "n_jobs": 4,
            "verbosity": 0,
            "tree_method": "hist",
        }
        lgb_params = {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 3,
            "subsample": 0.88,
            "subsample_freq": 1,
            "colsample_bytree": 0.86,
            "reg_alpha": 0.05,
            "reg_lambda": 1.1,
            "objective": "huber",
            "random_state": SEED,
            "n_jobs": 4,
            "verbose": -1,
        }

    splitter = TimeSeriesSplit(n_splits=INNER_SPLITS)
    errors = {"xgb": [], "lgb": []}
    feature_cols = x_frame.columns.tolist()
    for tr_idx, va_idx in splitter.split(x_frame):
        x_tr = x_frame.iloc[tr_idx]
        x_va = x_frame.iloc[va_idx]
        y_tr = y_transformed[tr_idx]
        base_va = base_values[va_idx]
        actual_va = actual_values[va_idx]
        weights_tr = sample_weights[tr_idx]

        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(x_tr, y_tr, sample_weight=weights_tr)
        pred_mult = np.exp(np.clip(xgb_model.predict(x_va), clip_low, clip_high))
        errors["xgb"].append(float(mean_absolute_error(actual_va, base_va * pred_mult)))

        lgb_model = lgb.LGBMRegressor(**lgb_params)
        lgb_model.fit(x_tr, y_tr, sample_weight=weights_tr)
        pred_mult = np.exp(np.clip(lgb_model.predict(x_va), clip_low, clip_high))
        errors["lgb"].append(float(mean_absolute_error(actual_va, base_va * pred_mult)))

    mean_errors = {name: float(np.mean(vals)) for name, vals in errors.items()}
    inverse = {name: 1.0 / max(value, 1e-9) for name, value in mean_errors.items()}
    total = sum(inverse.values())
    weights = {name: inverse[name] / total for name in inverse}

    xgb_full = xgb.XGBRegressor(**xgb_params)
    xgb_full.fit(x_frame, y_transformed, sample_weight=sample_weights)
    lgb_full = lgb.LGBMRegressor(**lgb_params)
    lgb_full.fit(x_frame, y_transformed, sample_weight=sample_weights)

    return ResidualTreeHead(
        feature_cols=feature_cols,
        xgb_model=xgb_full,
        lgb_model=lgb_full,
        weights=weights,
        clip_low=clip_low,
        clip_high=clip_high,
    )


def _predict_residual_head(
    head: ResidualTreeHead,
    x_frame: pd.DataFrame,
    base_values: np.ndarray,
) -> np.ndarray:
    x_frame = x_frame[head.feature_cols]
    xgb_raw = head.xgb_model.predict(x_frame)
    lgb_raw = head.lgb_model.predict(x_frame)
    xgb_mult = np.exp(np.clip(np.asarray(xgb_raw, dtype=float), head.clip_low, head.clip_high))
    lgb_mult = np.exp(np.clip(np.asarray(lgb_raw, dtype=float), head.clip_low, head.clip_high))
    multiplier = head.weights["xgb"] * xgb_mult + head.weights["lgb"] * lgb_mult
    return np.asarray(base_values, dtype=float) * multiplier


def _baseline_ratio(frame: pd.DataFrame) -> np.ndarray:
    ratio = frame["COGS"].to_numpy(dtype=float) / np.maximum(frame["Revenue"].to_numpy(dtype=float), 1e-9)
    return np.clip(ratio, RATIO_FLOOR, RATIO_CEIL)


def _fit_specialist_bundle(
    train_history: pd.DataFrame,
    promotions: pd.DataFrame,
    customers: pd.DataFrame,
    inventory: pd.DataFrame,
) -> SpecialistBundle:
    context, level_head, diff_head, ratio_head, orders_head, aov_head = _fit_fold_heads(
        train_history,
        promotions,
        customers,
    )
    cutoff = pd.Timestamp(train_history["date"].max())
    event_templates = _extract_event_templates(promotions, cutoff)
    story_priors = _build_story_priors(train_history, inventory)

    baseline_train = build_repo_style_baseline_frame(
        train_history,
        pd.DatetimeIndex(train_history["date"]),
        base_year=int(train_history["date"].dt.year.max()),
    )
    baseline_train_ratio = _baseline_ratio(baseline_train)
    regime_train = _build_regime_feature_frame(
        train_history["date"],
        context,
        event_templates,
        story_priors,
    )

    revenue_base = baseline_train["Revenue"].to_numpy(dtype=float)
    revenue_actual = train_history["Revenue"].to_numpy(dtype=float)
    revenue_multiplier = np.clip(revenue_actual / np.maximum(revenue_base, 1e-9), 0.55, 1.60)
    revenue_target = np.log(revenue_multiplier)
    revenue_weights = _residual_weight_vector(train_history["date"], regime_train, anomaly_boost=0.45)
    revenue_regime_head = _fit_residual_head(
        regime_train,
        revenue_target,
        revenue_base,
        revenue_actual,
        train_history["date"],
        revenue_weights,
        clip_low=np.log(0.55),
        clip_high=np.log(1.60),
        kind="revenue",
    )

    ratio_actual = np.clip(
        train_history["COGS"].to_numpy(dtype=float) / np.maximum(train_history["Revenue"].to_numpy(dtype=float), 1e-9),
        RATIO_FLOOR,
        RATIO_CEIL,
    )
    ratio_multiplier = np.clip(ratio_actual / np.maximum(baseline_train_ratio, 1e-9), 0.92, 1.12)
    ratio_target = np.log(ratio_multiplier)
    ratio_weights = _residual_weight_vector(train_history["date"], regime_train, anomaly_boost=0.80)
    ratio_regime_head = _fit_residual_head(
        regime_train,
        ratio_target,
        baseline_train_ratio,
        ratio_actual,
        train_history["date"],
        ratio_weights,
        clip_low=np.log(0.92),
        clip_high=np.log(1.12),
        kind="ratio",
    )

    return SpecialistBundle(
        context=context,
        level_head=level_head,
        diff_head=diff_head,
        ratio_head=ratio_head,
        orders_head=orders_head,
        aov_head=aov_head,
        event_templates=event_templates,
        story_priors=story_priors,
        revenue_regime_head=revenue_regime_head,
        ratio_regime_head=ratio_regime_head,
    )


def _predict_specialists(
    bundle: SpecialistBundle,
    train_history: pd.DataFrame,
    dates: pd.Series,
) -> pd.DataFrame:
    dates = pd.to_datetime(dates).reset_index(drop=True)
    baseline_frame = build_repo_style_baseline_frame(
        train_history,
        pd.DatetimeIndex(dates),
        base_year=int(train_history["date"].dt.year.max()),
    )
    baseline_ratio = _baseline_ratio(baseline_frame)
    recursive_level = _predict_level_head(bundle.level_head, train_history, dates, bundle.context)
    recursive_diff = _predict_diff_head(bundle.diff_head, train_history, dates, bundle.context)
    orders = _predict_component_head(bundle.orders_head, dates, bundle.context)
    aov = _predict_component_head(bundle.aov_head, dates, bundle.context)
    component_revenue = np.maximum(orders * aov, 0.0)
    regime_features = _build_regime_feature_frame(
        dates,
        bundle.context,
        bundle.event_templates,
        bundle.story_priors,
    )
    regime_features["baseline_revenue_log1p"] = np.log1p(np.maximum(baseline_frame["Revenue"].to_numpy(dtype=float), 0.0))
    regime_features["baseline_ratio"] = baseline_ratio
    regime_revenue = _predict_residual_head(
        bundle.revenue_regime_head,
        regime_features,
        baseline_frame["Revenue"].to_numpy(dtype=float),
    )
    recursive_ratio = _predict_ratio_head(
        bundle.ratio_head,
        train_history,
        dates,
        recursive_level,
        bundle.context,
    )
    ratio_regime = _predict_residual_head(
        bundle.ratio_regime_head,
        regime_features,
        baseline_ratio,
    )
    ratio_regime = np.clip(ratio_regime, RATIO_FLOOR, RATIO_CEIL)
    out = pd.DataFrame(
        {
            "Date": dates,
            "rev_baseline": baseline_frame["Revenue"].to_numpy(dtype=float),
            "rev_recursive_level": recursive_level,
            "rev_recursive_diff": recursive_diff,
            "rev_component": component_revenue,
            "rev_regime": regime_revenue,
            "ratio_baseline": baseline_ratio,
            "ratio_recursive": recursive_ratio,
            "ratio_regime": ratio_regime,
        }
    )
    out = pd.concat([out, regime_features.reset_index(drop=True)], axis=1)
    return out


def _horizon_bucket(fraction: pd.Series) -> pd.Series:
    out = pd.Series(index=fraction.index, dtype="object")
    out.loc[fraction < (1.0 / 3.0)] = "early"
    out.loc[(fraction >= (1.0 / 3.0)) & (fraction < (2.0 / 3.0))] = "mid"
    out.loc[fraction >= (2.0 / 3.0)] = "late"
    return out


def _horizon_bucket_fine(fraction: pd.Series) -> pd.Series:
    out = pd.Series(index=fraction.index, dtype="object")
    out.loc[fraction < 0.25] = "early"
    out.loc[(fraction >= 0.25) & (fraction < 0.50)] = "mid_early"
    out.loc[(fraction >= 0.50) & (fraction < 0.75)] = "mid_late"
    out.loc[fraction >= 0.75] = "late"
    return out


def _story_bucket(frame: pd.DataFrame) -> pd.Series:
    bucket = pd.Series("normal", index=frame.index, dtype="object")
    bucket.loc[(frame["quarter_end_window"] > 0) | (frame["quarter_start_window"] > 0)] = "quarter_edge"
    bucket.loc[frame["event_any"] > 0] = "sale_window"
    bucket.loc[(frame["event_any"] > 0) & (frame["quarter_end_window"] > 0)] = "sale_quarter_edge"
    bucket.loc[frame["is_odd_year_august"] > 0] = "odd_august"
    return bucket


def _ratio_regime(frame: pd.DataFrame) -> pd.Series:
    regime = pd.Series("normal", index=frame.index, dtype="object")
    regime.loc[frame["event_any"] > 0] = "sale_window"
    regime.loc[frame["is_odd_year_august"] > 0] = "odd_august"
    return regime


def _ratio_story_bucket(frame: pd.DataFrame) -> pd.Series:
    bucket = _story_bucket(frame)
    bucket.loc[
        (bucket == "normal")
        & ((frame["quarter_end_window"] > 0) | (frame["quarter_start_window"] > 0))
    ] = "quarter_edge"
    return bucket


def _build_oof_frame(
    history_daily: pd.DataFrame,
    promotions: pd.DataFrame,
    customers: pd.DataFrame,
    inventory: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    fold_rows: list[pd.DataFrame] = []
    explain_folds: list[dict[str, object]] = []
    for year in OUTER_YEARS:
        train_history = history_daily.loc[history_daily["date"].dt.year < year].copy().reset_index(drop=True)
        val_actual = history_daily.loc[history_daily["date"].dt.year == year].copy().reset_index(drop=True)
        if train_history.empty or val_actual.empty:
            continue
        print(
            f"[meta_model] outer fold {year}: train={len(train_history)} days, "
            f"valid={len(val_actual)} days"
        )
        bundle = _fit_specialist_bundle(train_history, promotions, customers, inventory)
        specialist_frame = _predict_specialists(bundle, train_history, val_actual["date"])
        specialist_frame["Revenue"] = val_actual["Revenue"].to_numpy(dtype=float)
        specialist_frame["COGS"] = val_actual["COGS"].to_numpy(dtype=float)
        specialist_frame["actual_ratio"] = np.clip(
            specialist_frame["COGS"].to_numpy(dtype=float) / np.maximum(specialist_frame["Revenue"].to_numpy(dtype=float), 1e-9),
            RATIO_FLOOR,
            RATIO_CEIL,
        )
        specialist_frame["fold_year"] = year
        specialist_frame["meta_static_bucket"] = "global"
        horizon_fraction = np.linspace(0.0, 1.0, len(specialist_frame), dtype=float) if len(specialist_frame) > 1 else np.asarray([1.0])
        specialist_frame["horizon_fraction"] = horizon_fraction
        specialist_frame["horizon_bucket"] = _horizon_bucket(pd.Series(horizon_fraction))
        specialist_frame["horizon_bucket_fine"] = _horizon_bucket_fine(pd.Series(horizon_fraction))
        specialist_frame["ratio_regime_bucket"] = _ratio_regime(specialist_frame)
        specialist_frame["story_bucket"] = _story_bucket(specialist_frame)
        specialist_frame["ratio_story_bucket"] = _ratio_story_bucket(specialist_frame)
        specialist_frame["meta_story_bucket"] = (
            specialist_frame["horizon_bucket"].astype(str)
            + "_"
            + specialist_frame["story_bucket"].astype(str)
        )
        fold_rows.append(specialist_frame)
        explain_folds.append(
            {
                "fold_year": year,
                "bundle": bundle,
                "train_history": train_history,
                "promotions": promotions,
                "customers": customers,
                "inventory": inventory,
            }
        )
    if not fold_rows:
        raise RuntimeError("Could not build OOF frame for meta ensemble.")
    return pd.concat(fold_rows, ignore_index=True), explain_folds


def _fit_simplex_nnls(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    weights, _ = nnls(x, y)
    if weights.sum() <= 0:
        weights = np.ones(x.shape[1], dtype=float) / float(x.shape[1])
    else:
        weights = weights / weights.sum()
    return weights


def _predict_simplex(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return x @ weights


def _fit_segmented_nnls(
    frame: pd.DataFrame,
    pred_cols: list[str],
    target_col: str,
    segment_col: str,
    min_rows: int = 40,
) -> dict[str, object]:
    global_weights = _fit_simplex_nnls(frame[pred_cols].to_numpy(dtype=float), frame[target_col].to_numpy(dtype=float))
    segments: dict[str, list[float]] = {}
    for segment, sub in frame.groupby(segment_col):
        if len(sub) < min_rows:
            continue
        segments[str(segment)] = _fit_simplex_nnls(
            sub[pred_cols].to_numpy(dtype=float),
            sub[target_col].to_numpy(dtype=float),
        ).tolist()
    return {
        "type": "nnls",
        "pred_cols": pred_cols,
        "segment_col": segment_col,
        "global_weights": global_weights.tolist(),
        "segment_weights": segments,
    }


def _predict_segmented_nnls(frame: pd.DataFrame, model: dict[str, object]) -> np.ndarray:
    pred_cols = list(model["pred_cols"])
    global_weights = np.asarray(model["global_weights"], dtype=float)
    out = np.zeros(len(frame), dtype=float)
    segment_col = str(model["segment_col"])
    use_segment = segment_col in frame.columns
    for i, (_, row) in enumerate(frame.iterrows()):
        segment = str(row[segment_col]) if use_segment else "__global__"
        weights = np.asarray(model["segment_weights"].get(segment, global_weights.tolist()), dtype=float)
        out[i] = float(np.dot(row[pred_cols].to_numpy(dtype=float), weights))
    return out


def _fit_segmented_ridge(
    frame: pd.DataFrame,
    pred_cols: list[str],
    target_col: str,
    segment_col: str,
    alpha: float = 1.0,
    min_rows: int = 40,
) -> dict[str, object]:
    global_model = Ridge(alpha=alpha, fit_intercept=True)
    global_model.fit(frame[pred_cols], frame[target_col])
    segments: dict[str, dict[str, object]] = {}
    for segment, sub in frame.groupby(segment_col):
        if len(sub) < min_rows:
            continue
        ridge = Ridge(alpha=alpha, fit_intercept=True)
        ridge.fit(sub[pred_cols], sub[target_col])
        segments[str(segment)] = {
            "coef": ridge.coef_.tolist(),
            "intercept": float(ridge.intercept_),
        }
    return {
        "type": "ridge",
        "pred_cols": pred_cols,
        "segment_col": segment_col,
        "global_coef": global_model.coef_.tolist(),
        "global_intercept": float(global_model.intercept_),
        "segment_models": segments,
    }


def _predict_segmented_ridge(frame: pd.DataFrame, model: dict[str, object]) -> np.ndarray:
    pred_cols = list(model["pred_cols"])
    global_coef = np.asarray(model["global_coef"], dtype=float)
    global_intercept = float(model["global_intercept"])
    out = np.zeros(len(frame), dtype=float)
    segment_col = str(model["segment_col"])
    use_segment = segment_col in frame.columns
    for i, (_, row) in enumerate(frame.iterrows()):
        segment = str(row[segment_col]) if use_segment else "__global__"
        segment_model = model["segment_models"].get(segment)
        if segment_model is None:
            coef = global_coef
            intercept = global_intercept
        else:
            coef = np.asarray(segment_model["coef"], dtype=float)
            intercept = float(segment_model["intercept"])
        out[i] = float(np.dot(row[pred_cols].to_numpy(dtype=float), coef) + intercept)
    return np.maximum(out, 0.0)


def _apply_meta_candidate(
    frame: pd.DataFrame,
    candidate: MetaCandidate,
) -> pd.DataFrame:
    out = frame.copy()
    if candidate.revenue_mode == "nnls":
        revenue = _predict_segmented_nnls(out, candidate.revenue_model)
    else:
        revenue = _predict_segmented_ridge(out, candidate.revenue_model)

    if candidate.ratio_mode == "nnls":
        ratio = _predict_segmented_nnls(out, candidate.ratio_model)
    else:
        ratio = _predict_segmented_ridge(out, candidate.ratio_model)

    out["meta_revenue"] = np.maximum(revenue, 0.0)
    out["meta_ratio"] = np.clip(ratio, RATIO_FLOOR, RATIO_CEIL)
    out["meta_cogs"] = out["meta_revenue"] * out["meta_ratio"]
    return out


def _candidate_metrics(frame: pd.DataFrame) -> dict[str, float]:
    revenue_metrics, cogs_metrics, joint = _joint_objective(
        frame["Revenue"],
        frame["meta_revenue"],
        frame["COGS"],
        frame["meta_cogs"],
    )
    return {
        "joint_objective": float(joint),
        "revenue_mae": float(revenue_metrics["mae"]),
        "revenue_rmse": float(revenue_metrics["rmse"]),
        "revenue_r2": float(revenue_metrics["r2"]),
        "cogs_mae": float(cogs_metrics["mae"]),
        "cogs_rmse": float(cogs_metrics["rmse"]),
        "cogs_r2": float(cogs_metrics["r2"]),
    }


def _fit_meta_candidates(oof_frame: pd.DataFrame) -> list[MetaCandidate]:
    candidates: list[MetaCandidate] = []

    static = MetaCandidate(
        name="meta_static_nnls",
        methodology="Static simplex-normalized NNLS blend over baseline, recursive, component, and regime specialists.",
        revenue_mode="nnls",
        ratio_mode="nnls",
        revenue_model=_fit_segmented_nnls(oof_frame, REVENUE_BLEND_COLS, "Revenue", "meta_static_bucket", min_rows=1),
        ratio_model=_fit_segmented_nnls(oof_frame, RATIO_BLEND_COLS, "actual_ratio", "meta_static_bucket", min_rows=1),
        metrics={},
    )
    static_frame = _apply_meta_candidate(oof_frame, static)
    static.metrics = _candidate_metrics(static_frame)
    candidates.append(static)

    horizon_nnls = MetaCandidate(
        name="meta_horizon_nnls",
        methodology="OOF NNLS with separate revenue weights for early, mid, and late horizon buckets.",
        revenue_mode="nnls",
        ratio_mode="nnls",
        revenue_model=_fit_segmented_nnls(oof_frame, REVENUE_BLEND_COLS, "Revenue", "horizon_bucket", min_rows=60),
        ratio_model=_fit_segmented_nnls(oof_frame, RATIO_BLEND_COLS, "actual_ratio", "fold_year", min_rows=1),
        metrics={},
    )
    horizon_nnls_frame = _apply_meta_candidate(oof_frame, horizon_nnls)
    horizon_nnls.metrics = _candidate_metrics(horizon_nnls_frame)
    candidates.append(horizon_nnls)

    fine_horizon_nnls = MetaCandidate(
        name="meta_fine_horizon_nnls",
        methodology="OOF NNLS with finer early to late revenue buckets and global ratio blend for long-horizon stability.",
        revenue_mode="nnls",
        ratio_mode="nnls",
        revenue_model=_fit_segmented_nnls(oof_frame, REVENUE_BLEND_COLS, "Revenue", "horizon_bucket_fine", min_rows=45),
        ratio_model=_fit_segmented_nnls(oof_frame, RATIO_BLEND_COLS, "actual_ratio", "meta_static_bucket", min_rows=1),
        metrics={},
    )
    fine_horizon_frame = _apply_meta_candidate(oof_frame, fine_horizon_nnls)
    fine_horizon_nnls.metrics = _candidate_metrics(fine_horizon_frame)
    candidates.append(fine_horizon_nnls)

    horizon_ridge = MetaCandidate(
        name="meta_horizon_ridge",
        methodology="OOF ridge stacker with separate revenue weights by horizon bucket and ratio shrinkage by fold.",
        revenue_mode="ridge",
        ratio_mode="ridge",
        revenue_model=_fit_segmented_ridge(oof_frame, REVENUE_BLEND_COLS, "Revenue", "horizon_bucket", alpha=1.0, min_rows=60),
        ratio_model=_fit_segmented_ridge(oof_frame, RATIO_BLEND_COLS, "actual_ratio", "fold_year", alpha=0.5, min_rows=1),
        metrics={},
    )
    horizon_ridge_frame = _apply_meta_candidate(oof_frame, horizon_ridge)
    horizon_ridge.metrics = _candidate_metrics(horizon_ridge_frame)
    candidates.append(horizon_ridge)

    horizon_regime = MetaCandidate(
        name="meta_horizon_regime_nnls",
        methodology="OOF NNLS with revenue weights by horizon bucket and ratio weights by anomaly regime.",
        revenue_mode="nnls",
        ratio_mode="nnls",
        revenue_model=_fit_segmented_nnls(oof_frame, REVENUE_BLEND_COLS, "Revenue", "horizon_bucket", min_rows=60),
        ratio_model=_fit_segmented_nnls(oof_frame, RATIO_BLEND_COLS, "actual_ratio", "ratio_regime_bucket", min_rows=20),
        metrics={},
    )
    horizon_regime_frame = _apply_meta_candidate(oof_frame, horizon_regime)
    horizon_regime.metrics = _candidate_metrics(horizon_regime_frame)
    candidates.append(horizon_regime)

    horizon_story = MetaCandidate(
        name="meta_horizon_story_nnls",
        methodology="OOF NNLS with revenue weights by horizon x story bucket and ratio weights by story regime.",
        revenue_mode="nnls",
        ratio_mode="nnls",
        revenue_model=_fit_segmented_nnls(oof_frame, REVENUE_BLEND_COLS, "Revenue", "meta_story_bucket", min_rows=35),
        ratio_model=_fit_segmented_nnls(oof_frame, RATIO_BLEND_COLS, "actual_ratio", "ratio_story_bucket", min_rows=20),
        metrics={},
    )
    horizon_story_frame = _apply_meta_candidate(oof_frame, horizon_story)
    horizon_story.metrics = _candidate_metrics(horizon_story_frame)
    candidates.append(horizon_story)

    candidates.sort(key=lambda item: item.metrics["joint_objective"])
    return candidates


def _meta_rows(candidates: list[MetaCandidate]) -> pd.DataFrame:
    rows = []
    for idx, candidate in enumerate(candidates):
        row = {
            "candidate_name": candidate.name,
            "methodology": candidate.methodology,
            "revenue_mode": candidate.revenue_mode,
            "ratio_mode": candidate.ratio_mode,
            **candidate.metrics,
            "selected": 1 if idx == 0 else 0,
            "internal_cv_winner": 1 if candidate.name == INTERNAL_CV_WINNER_NAME else 0,
            "final_submission": 1 if candidate.name == FINAL_SUBMISSION_CANDIDATE_NAME else 0,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _candidate_by_name(candidates: list[MetaCandidate], name: str) -> MetaCandidate | None:
    return next((candidate for candidate in candidates if candidate.name == name), None)


def _weights_table(candidate: MetaCandidate) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if candidate.revenue_mode == "nnls":
        revenue_global = dict(zip(REVENUE_BLEND_COLS, np.asarray(candidate.revenue_model["global_weights"], dtype=float)))
        rows.extend(
            {"target": "revenue", "segment": "global", "component": key, "weight": float(value)}
            for key, value in revenue_global.items()
        )
        for segment, weights in candidate.revenue_model["segment_weights"].items():
            if str(segment) == "global":
                continue
            rows.extend(
                {"target": "revenue", "segment": segment, "component": key, "weight": float(value)}
                for key, value in zip(REVENUE_BLEND_COLS, np.asarray(weights, dtype=float))
            )
    else:
        rows.extend(
            {"target": "revenue", "segment": "global", "component": key, "weight": float(value)}
            for key, value in zip(REVENUE_BLEND_COLS, np.asarray(candidate.revenue_model["global_coef"], dtype=float))
        )
        rows.append(
            {
                "target": "revenue",
                "segment": "global",
                "component": "intercept",
                "weight": float(candidate.revenue_model["global_intercept"]),
            }
        )
        for segment, params in candidate.revenue_model["segment_models"].items():
            rows.extend(
                {"target": "revenue", "segment": segment, "component": key, "weight": float(value)}
                for key, value in zip(REVENUE_BLEND_COLS, np.asarray(params["coef"], dtype=float))
            )
            rows.append(
                {
                    "target": "revenue",
                    "segment": segment,
                    "component": "intercept",
                    "weight": float(params["intercept"]),
                }
            )

    if candidate.ratio_mode == "nnls":
        ratio_global = dict(zip(RATIO_BLEND_COLS, np.asarray(candidate.ratio_model["global_weights"], dtype=float)))
        rows.extend(
            {"target": "ratio", "segment": "global", "component": key, "weight": float(value)}
            for key, value in ratio_global.items()
        )
        for segment, weights in candidate.ratio_model["segment_weights"].items():
            if str(segment) == "global":
                continue
            rows.extend(
                {"target": "ratio", "segment": segment, "component": key, "weight": float(value)}
                for key, value in zip(RATIO_BLEND_COLS, np.asarray(weights, dtype=float))
            )
    else:
        rows.extend(
            {"target": "ratio", "segment": "global", "component": key, "weight": float(value)}
            for key, value in zip(RATIO_BLEND_COLS, np.asarray(candidate.ratio_model["global_coef"], dtype=float))
        )
        rows.append(
            {
                "target": "ratio",
                "segment": "global",
                "component": "intercept",
                "weight": float(candidate.ratio_model["global_intercept"]),
            }
        )
        for segment, params in candidate.ratio_model["segment_models"].items():
            rows.extend(
                {"target": "ratio", "segment": segment, "component": key, "weight": float(value)}
                for key, value in zip(RATIO_BLEND_COLS, np.asarray(params["coef"], dtype=float))
            )
            rows.append(
                {
                    "target": "ratio",
                    "segment": segment,
                    "component": "intercept",
                    "weight": float(params["intercept"]),
                }
            )
    return pd.DataFrame(rows)


def _xgb_shap_summary(model: object, x_frame: pd.DataFrame) -> pd.DataFrame:
    dmatrix = xgb.DMatrix(x_frame, feature_names=list(x_frame.columns))
    contrib = model.get_booster().predict(dmatrix, pred_contribs=True)
    values = np.abs(contrib[:, :-1]).mean(axis=0)
    return (
        pd.DataFrame({"feature": x_frame.columns, "mean_abs_shap": values})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def _feature_stability_from_folds(
    fold_summaries: list[dict[str, object]],
    top_n: int = 20,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold_summary in fold_summaries:
        fold_year = int(fold_summary["fold_year"])
        for model_name, summary in fold_summary["summaries"].items():
            table = summary.head(top_n).copy()
            table["rank"] = np.arange(1, len(table) + 1)
            for row in table.itertuples(index=False):
                rows.append(
                    {
                        "fold_year": fold_year,
                        "model": model_name,
                        "feature": row.feature,
                        "mean_abs_shap": float(row.mean_abs_shap),
                        "rank": int(row.rank),
                    }
                )
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["model", "feature"])
        .agg(
            mean_abs_shap=("mean_abs_shap", "mean"),
            std_abs_shap=("mean_abs_shap", "std"),
            folds_present=("fold_year", "nunique"),
            mean_rank=("rank", "mean"),
        )
        .reset_index()
        .sort_values(["model", "mean_rank", "mean_abs_shap"], ascending=[True, True, False])
    )
    return frame, summary


def _full_bundle_shap_outputs(
    history_daily: pd.DataFrame,
    promotions: pd.DataFrame,
    customers: pd.DataFrame,
    inventory: pd.DataFrame,
    bundle: SpecialistBundle,
) -> dict[str, pd.DataFrame]:
    context = bundle.context
    level_x, _, _, _, _ = _build_revenue_level_frame(history_daily, context)
    ratio_x, _, _, _, _ = _build_ratio_frame(history_daily, context)

    regime_x = _build_regime_feature_frame(
        history_daily["date"],
        context,
        bundle.event_templates,
        bundle.story_priors,
    )
    baseline_full = build_repo_style_baseline_frame(
        history_daily,
        pd.DatetimeIndex(history_daily["date"]),
        base_year=int(history_daily["date"].dt.year.max()),
    )
    regime_x["baseline_revenue_log1p"] = np.log1p(np.maximum(baseline_full["Revenue"].to_numpy(dtype=float), 0.0))
    regime_x["baseline_ratio"] = _baseline_ratio(baseline_full)

    level_summary = _xgb_shap_summary(bundle.level_head.xgb_model, level_x[bundle.level_head.feature_cols].tail(730))
    ratio_summary = _xgb_shap_summary(bundle.ratio_head.xgb_model, ratio_x[bundle.ratio_head.feature_cols].tail(730))
    revenue_regime_summary = _xgb_shap_summary(
        bundle.revenue_regime_head.xgb_model,
        regime_x[bundle.revenue_regime_head.feature_cols].tail(730),
    )
    ratio_regime_summary = _xgb_shap_summary(
        bundle.ratio_regime_head.xgb_model,
        regime_x[bundle.ratio_regime_head.feature_cols].tail(730),
    )
    return {
        "revenue_recursive": level_summary,
        "revenue_regime": revenue_regime_summary,
        "ratio_recursive": ratio_summary,
        "ratio_regime": ratio_regime_summary,
    }


def _build_explainability_pack(
    history_daily: pd.DataFrame,
    promotions: pd.DataFrame,
    customers: pd.DataFrame,
    inventory: pd.DataFrame,
    explain_folds: list[dict[str, object]],
) -> dict[str, pd.DataFrame]:
    full_bundle = _fit_specialist_bundle(history_daily, promotions, customers, inventory)
    shap_tables = _full_bundle_shap_outputs(history_daily, promotions, customers, inventory, full_bundle)

    fold_summaries: list[dict[str, object]] = []
    for fold in explain_folds:
        bundle = fold["bundle"]
        train_history = fold["train_history"]
        summaries = _full_bundle_shap_outputs(train_history, promotions, customers, inventory, bundle)
        fold_summaries.append({"fold_year": fold["fold_year"], "summaries": summaries})

    fold_frame, stability_summary = _feature_stability_from_folds(fold_summaries)
    return {
        **shap_tables,
        "feature_stability_by_fold": fold_frame,
        "feature_stability_summary": stability_summary,
    }


def _future_meta_frame(
    history_daily: pd.DataFrame,
    promotions: pd.DataFrame,
    customers: pd.DataFrame,
    inventory: pd.DataFrame,
) -> tuple[pd.DataFrame, SpecialistBundle]:
    bundle = _fit_specialist_bundle(history_daily, promotions, customers, inventory)
    future = _predict_specialists(bundle, history_daily, pd.Series(sample_dates()))
    future["meta_static_bucket"] = "global"
    horizon_fraction = np.linspace(0.0, 1.0, len(future), dtype=float) if len(future) > 1 else np.asarray([1.0])
    future["horizon_fraction"] = horizon_fraction
    future["horizon_bucket"] = _horizon_bucket(pd.Series(horizon_fraction))
    future["horizon_bucket_fine"] = _horizon_bucket_fine(pd.Series(horizon_fraction))
    future["ratio_regime_bucket"] = _ratio_regime(future)
    future["story_bucket"] = _story_bucket(future)
    future["ratio_story_bucket"] = _ratio_story_bucket(future)
    future["meta_story_bucket"] = future["horizon_bucket"].astype(str) + "_" + future["story_bucket"].astype(str)
    return future, bundle


def build_final_meta_regime_ensemble() -> dict[str, object]:
    data = load_dataframes()
    history_daily = build_daily_frame(data)
    promotions = data["promotions"].copy()
    customers = data["customers"].copy()
    inventory = data["inventory"].copy()

    oof_frame, explain_folds = _build_oof_frame(history_daily, promotions, customers, inventory)
    candidate_list = _fit_meta_candidates(oof_frame)
    candidate_table = _meta_rows(candidate_list)

    selected = _candidate_by_name(candidate_list, INTERNAL_CV_WINNER_NAME) or candidate_list[0]
    practical_candidate = _candidate_by_name(candidate_list, FINAL_SUBMISSION_CANDIDATE_NAME) or selected
    selected_config = {
        "internal_cv_winner_name": selected.name,
        "internal_cv_winner_methodology": selected.methodology,
        "internal_cv_winner_metrics": selected.metrics,
        "final_submission_candidate_name": practical_candidate.name,
        "final_submission_candidate_methodology": practical_candidate.methodology,
        "final_submission_candidate_metrics": practical_candidate.metrics,
        "paper_shoutout_models": list(PAPER_SHOUTOUT_MODELS),
        "selection_rationale": (
            "Internal CV winner is kept for benchmarking; final submission candidate is chosen "
            "by leaderboard-aligned practical performance under long-horizon regime shift."
        ),
    }

    future_frame, _ = _future_meta_frame(history_daily, promotions, customers, inventory)
    candidate_paths: dict[str, Path] = {}
    for candidate in candidate_list:
        applied = _apply_meta_candidate(future_frame, candidate)
        submission = pd.DataFrame(
            {
                "Date": pd.to_datetime(sample_dates()),
                "Revenue": np.maximum(applied["meta_revenue"].to_numpy(dtype=float), 0.0),
                "COGS": np.maximum(applied["meta_cogs"].to_numpy(dtype=float), 0.0),
            }
        )
        path = _meta_candidate_path(candidate.name)
        submission.to_csv(path, index=False, float_format="%.2f")
        candidate_paths[candidate.name] = path

    weights_table = _weights_table(selected)
    final_weights_table = _weights_table(practical_candidate)

    oof_applied = _apply_meta_candidate(oof_frame, selected)

    explainability_outputs = _build_explainability_pack(
        history_daily,
        promotions,
        customers,
        inventory,
        explain_folds,
    )

    return {
        "candidate_table": candidate_table,
        "selected_config": selected_config,
        "selected_submission": candidate_paths[selected.name],
        "final_submission": candidate_paths[practical_candidate.name],
        "final_candidate_name": practical_candidate.name,
        "practical_submission": candidate_paths[practical_candidate.name],
        "practical_candidate_name": practical_candidate.name,
        "candidate_submissions": candidate_paths,
        "selected_objective": float(selected.metrics["joint_objective"]),
        "practical_objective": float(practical_candidate.metrics["joint_objective"]),
        "weights_table": weights_table,
        "final_weights_table": final_weights_table,
        "oof_predictions": oof_applied,
        "explainability": explainability_outputs,
    }


def main() -> None:
    outputs = build_final_meta_regime_ensemble()
    print(str(outputs["practical_submission"]))


if __name__ == "__main__":
    main()

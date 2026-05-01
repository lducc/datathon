from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from .data import (
    build_daily_frame,
    final_submission_path,
    load_dataframes,
    sample_dates,
)
RATIO_FLOOR = 0.72
RATIO_CEIL = 0.995
LAGS = (1, 2, 3, 7, 14, 28)
ROLL_WINDOWS = (3, 7, 14)
EMA_SPANS = (3, 7)
BETA_GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
TIME_DECAY_CONFIGS = (
    (0.00, 0.00),
    (0.05, 0.35),
    (0.10, 0.40),
    (0.10, 0.50),
    (0.15, 0.60),
)
OUTER_YEARS = (2019, 2020, 2021, 2022)
INNER_SPLITS = 3
FINALIST_CANDIDATES = 6
SEED = 2026


@dataclass
class FeatureContext:
    promo_active_doy: dict[int, float]
    promo_discount_doy: dict[int, float]
    promo_stackable_doy: dict[int, float]
    promo_channel_div_doy: dict[int, float]
    promo_active_global: float
    promo_discount_global: float
    promo_stackable_global: float
    promo_channel_div_global: float
    customer_growth: dict[pd.Timestamp, float]
    revenue_month_mean: dict[int, float]
    orders_month_mean: dict[int, float]
    aov_month_mean: dict[int, float]
    revenue_dow_mean: dict[int, float]
    orders_dow_mean: dict[int, float]
    aov_dow_mean: dict[int, float]


@dataclass
class HeadSpec:
    name: str
    xgb_params: dict[str, object]
    lgb_params: dict[str, object]
    transform_target: Callable[[pd.Series], np.ndarray]
    inverse_batch: Callable[[np.ndarray, pd.DataFrame], np.ndarray]
    inverse_scalar: Callable[[float, dict[str, float]], float]
    clip_low: float
    clip_high: float


@dataclass
class HeadModel:
    spec: HeadSpec
    feature_cols: list[str]
    xgb_model: object
    lgb_model: object
    weights: dict[str, float]


def _metric_block(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    actual_arr = actual.to_numpy(dtype=float)
    pred_arr = predicted.to_numpy(dtype=float)
    error = pred_arr - actual_arr
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    denom = float(np.mean(np.abs(actual_arr))) if np.mean(np.abs(actual_arr)) > 0 else 1.0
    nmae = mae / denom
    nrmse = rmse / denom
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((actual_arr - actual_arr.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    objective = nmae + nrmse + 0.50 * max(0.0, 1.0 - r2)
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "nmae": nmae,
        "nrmse": nrmse,
        "objective": objective,
    }


def _joint_objective(
    revenue_actual: pd.Series,
    revenue_pred: pd.Series,
    cogs_actual: pd.Series,
    cogs_pred: pd.Series,
) -> tuple[dict[str, float], dict[str, float], float]:
    revenue_metrics = _metric_block(revenue_actual, revenue_pred)
    cogs_metrics = _metric_block(cogs_actual, cogs_pred)
    return revenue_metrics, cogs_metrics, revenue_metrics["objective"] + cogs_metrics["objective"]


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _expand_promotions(promotions: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.DataFrame:
    promo = promotions.copy()
    promo["start_date"] = pd.to_datetime(promo["start_date"], errors="coerce")
    promo["end_date"] = pd.to_datetime(promo["end_date"], errors="coerce")
    promo = promo.dropna(subset=["start_date", "end_date"]).copy()
    promo = promo.loc[promo["start_date"] <= cutoff_date].copy()
    if promo.empty:
        return pd.DataFrame(columns=["date", "active", "discount_pct", "stackable", "channel"])

    rows: list[dict[str, object]] = []
    for row in promo.itertuples(index=False):
        start = pd.Timestamp(row.start_date)
        end = min(pd.Timestamp(row.end_date), cutoff_date)
        if end < start:
            continue
        promo_type = str(getattr(row, "promo_type", "") or "").strip().lower()
        discount_value = float(getattr(row, "discount_value", 0.0) or 0.0)
        discount_pct = discount_value if promo_type == "percentage" else 0.0
        stackable = float(getattr(row, "stackable_flag", 0.0) or 0.0)
        channel = str(getattr(row, "promo_channel", "unknown") or "unknown").strip().lower()
        for date in pd.date_range(start, end, freq="D"):
            rows.append(
                {
                    "date": date,
                    "active": 1.0,
                    "discount_pct": discount_pct,
                    "stackable": stackable,
                    "channel": channel,
                }
            )
    return pd.DataFrame(rows)


def _build_feature_context(
    history_daily: pd.DataFrame,
    promotions: pd.DataFrame,
    customers: pd.DataFrame,
) -> FeatureContext:
    cutoff = pd.Timestamp(history_daily["date"].max())
    promo_daily = _expand_promotions(promotions, cutoff)
    if promo_daily.empty:
        promo_active_doy: dict[int, float] = {}
        promo_discount_doy: dict[int, float] = {}
        promo_stackable_doy: dict[int, float] = {}
        promo_channel_div_doy: dict[int, float] = {}
        promo_active_global = 0.0
        promo_discount_global = 0.0
        promo_stackable_global = 0.0
        promo_channel_div_global = 0.0
    else:
        promo_agg = (
            promo_daily.groupby("date")
            .agg(
                active=("active", "sum"),
                discount_pct=("discount_pct", "sum"),
                stackable=("stackable", "mean"),
                channel_diversity=("channel", "nunique"),
            )
            .reset_index()
        )
        promo_agg["doy"] = promo_agg["date"].dt.dayofyear
        by_doy = (
            promo_agg.groupby("doy")
            .agg(
                active=("active", "mean"),
                discount_pct=("discount_pct", "mean"),
                stackable=("stackable", "mean"),
                channel_diversity=("channel_diversity", "mean"),
            )
            .reset_index()
        )
        promo_active_doy = {int(k): float(v) for k, v in zip(by_doy["doy"], by_doy["active"])}
        promo_discount_doy = {int(k): float(v) for k, v in zip(by_doy["doy"], by_doy["discount_pct"])}
        promo_stackable_doy = {int(k): float(v) for k, v in zip(by_doy["doy"], by_doy["stackable"])}
        promo_channel_div_doy = {
            int(k): float(v) for k, v in zip(by_doy["doy"], by_doy["channel_diversity"])
        }
        promo_active_global = float(promo_agg["active"].mean())
        promo_discount_global = float(promo_agg["discount_pct"].mean())
        promo_stackable_global = float(promo_agg["stackable"].mean())
        promo_channel_div_global = float(promo_agg["channel_diversity"].mean())

    customer_growth: dict[pd.Timestamp, float] = {}
    cust = customers.copy()
    if "signup_date" in cust.columns:
        cust["signup_date"] = pd.to_datetime(cust["signup_date"], errors="coerce").dt.normalize()
        cust = cust.dropna(subset=["signup_date"]).copy()
        cust = cust.loc[cust["signup_date"] <= cutoff].copy()
        if not cust.empty:
            daily_new = cust.groupby("signup_date").size().sort_index()
            rolling_new = daily_new.rolling(7, min_periods=1).mean()
            customer_growth = {pd.Timestamp(k): float(v) for k, v in rolling_new.to_dict().items()}

    history = history_daily.copy()
    history["month"] = history["date"].dt.month
    history["dow"] = history["date"].dt.dayofweek
    history["orders_filled"] = history["orders"].fillna(0.0)
    history["aov_filled"] = (
        history["Revenue"] / history["orders_filled"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    aov_median = float(history["aov_filled"].median()) if history["aov_filled"].notna().any() else 0.0
    history["aov_filled"] = history["aov_filled"].fillna(aov_median).clip(lower=0.0)

    return FeatureContext(
        promo_active_doy=promo_active_doy,
        promo_discount_doy=promo_discount_doy,
        promo_stackable_doy=promo_stackable_doy,
        promo_channel_div_doy=promo_channel_div_doy,
        promo_active_global=promo_active_global,
        promo_discount_global=promo_discount_global,
        promo_stackable_global=promo_stackable_global,
        promo_channel_div_global=promo_channel_div_global,
        customer_growth=customer_growth,
        revenue_month_mean={int(k): float(v) for k, v in history.groupby("month")["Revenue"].mean().to_dict().items()},
        orders_month_mean={int(k): float(v) for k, v in history.groupby("month")["orders_filled"].mean().to_dict().items()},
        aov_month_mean={int(k): float(v) for k, v in history.groupby("month")["aov_filled"].mean().to_dict().items()},
        revenue_dow_mean={int(k): float(v) for k, v in history.groupby("dow")["Revenue"].mean().to_dict().items()},
        orders_dow_mean={int(k): float(v) for k, v in history.groupby("dow")["orders_filled"].mean().to_dict().items()},
        aov_dow_mean={int(k): float(v) for k, v in history.groupby("dow")["aov_filled"].mean().to_dict().items()},
    )


def _calendar_features(date: pd.Timestamp, context: FeatureContext) -> dict[str, float]:
    year = int(date.year)
    month = int(date.month)
    day = int(date.day)
    dow = int(date.dayofweek)
    doy = int(date.dayofyear)
    days_in_month = int(date.days_in_month)
    days_to_month_end = int(days_in_month - day)
    is_weekend = 1.0 if dow >= 5 else 0.0
    is_payday = 1.0 if (day >= 25 or day <= 3) else 0.0
    is_mega_1111 = 1.0 if (month == 11 and day == 11) else 0.0
    is_mega_1212 = 1.0 if (month == 12 and day == 12) else 0.0
    is_first3 = 1.0 if day <= 3 else 0.0
    is_first7 = 1.0 if day <= 7 else 0.0
    is_last1 = 1.0 if days_to_month_end == 0 else 0.0
    is_last3 = 1.0 if days_to_month_end <= 2 else 0.0
    is_last7 = 1.0 if days_to_month_end <= 6 else 0.0
    is_post_2019 = 1.0 if year >= 2019 else 0.0
    is_odd_year_august = 1.0 if (month == 8 and year % 2 == 1) else 0.0
    is_even_year_august = 1.0 if (month == 8 and year % 2 == 0) else 0.0
    recent_user_growth = context.customer_growth.get(
        pd.Timestamp(date).normalize() - pd.Timedelta(days=7),
        0.0,
    )
    promo_active_prior = float(context.promo_active_doy.get(doy, context.promo_active_global))
    promo_discount_prior = float(context.promo_discount_doy.get(doy, context.promo_discount_global))
    return {
        "year": float(year),
        "day_of_year": float(doy),
        "day_of_month": float(day),
        "day_of_week": float(dow),
        "month": float(month),
        "week_of_year": float(date.isocalendar().week),
        "quarter": float(date.quarter),
        "days_in_month": float(days_in_month),
        "days_to_month_end": float(days_to_month_end),
        "days_from_mend": float(days_to_month_end),
        "is_weekend": is_weekend,
        "is_month_start": float(date.is_month_start),
        "is_month_end": float(date.is_month_end),
        "is_payday": is_payday,
        "is_first3": is_first3,
        "is_first7": is_first7,
        "is_last1": is_last1,
        "is_last3": is_last3,
        "is_last7": is_last7,
        "is_mega_1111": is_mega_1111,
        "is_mega_1212": is_mega_1212,
        "is_post_2019": is_post_2019,
        "is_odd_year_august": is_odd_year_august,
        "is_even_year_august": is_even_year_august,
        "doy_sin": float(np.sin(2.0 * np.pi * doy / 365.25)),
        "doy_cos": float(np.cos(2.0 * np.pi * doy / 365.25)),
        "dow_sin": float(np.sin(2.0 * np.pi * dow / 7.0)),
        "dow_cos": float(np.cos(2.0 * np.pi * dow / 7.0)),
        "month_sin": float(np.sin(2.0 * np.pi * month / 12.0)),
        "month_cos": float(np.cos(2.0 * np.pi * month / 12.0)),
        "promo_active_prior": promo_active_prior,
        "promo_discount_prior": promo_discount_prior,
        "promo_stackable_prior": float(context.promo_stackable_doy.get(doy, context.promo_stackable_global)),
        "promo_channel_div_prior": float(
            context.promo_channel_div_doy.get(doy, context.promo_channel_div_global)
        ),
        "promo_payday_interact": promo_active_prior * is_payday,
        "promo_weekend_interact": promo_active_prior * is_weekend,
        "promo_mega1111_interact": promo_discount_prior * is_mega_1111,
        "promo_mega1212_interact": promo_discount_prior * is_mega_1212,
        "promo_odd_august_interact": promo_discount_prior * is_odd_year_august,
        "payday_last7_interact": is_payday * is_last7,
        "recent_user_growth": float(recent_user_growth),
        "hist_month_revenue_log1p": float(np.log1p(max(context.revenue_month_mean.get(month, 0.0), 0.0))),
        "hist_month_orders_log1p": float(np.log1p(max(context.orders_month_mean.get(month, 0.0), 0.0))),
        "hist_month_aov_log1p": float(np.log1p(max(context.aov_month_mean.get(month, 0.0), 0.0))),
        "hist_dow_revenue_log1p": float(np.log1p(max(context.revenue_dow_mean.get(dow, 0.0), 0.0))),
        "hist_dow_orders_log1p": float(np.log1p(max(context.orders_dow_mean.get(dow, 0.0), 0.0))),
        "hist_dow_aov_log1p": float(np.log1p(max(context.aov_dow_mean.get(dow, 0.0), 0.0))),
    }


def _sample_weights(dates: pd.Series) -> np.ndarray:
    series = pd.to_datetime(dates).reset_index(drop=True)
    base_idx = np.arange(len(series), dtype=float)
    recency = 0.80 + 0.60 * (base_idx / max(len(series) - 1, 1))
    special = np.ones(len(series), dtype=float)
    special[(series.dt.month == 11) & (series.dt.day == 11)] = 2.5
    special[(series.dt.month == 12) & (series.dt.day == 12)] = 2.5
    special[(series.dt.day >= 25) | (series.dt.day <= 3)] += 0.35
    special[series.dt.year >= 2019] += 0.20
    return recency * special


def _build_autoreg_base_frame(
    dates: pd.Series,
    target_series: pd.Series,
    context: FeatureContext,
    prefix: str,
) -> tuple[pd.DataFrame, list[str]]:
    feats = pd.DataFrame(index=target_series.index)
    for lag in LAGS:
        feats[f"{prefix}_lag_{lag}"] = target_series.shift(lag)
    shifted = target_series.shift(1)
    for window in ROLL_WINDOWS:
        feats[f"{prefix}_roll_mean_{window}"] = shifted.rolling(window).mean()
        feats[f"{prefix}_roll_std_{window}"] = shifted.rolling(window).std()
    for span in EMA_SPANS:
        feats[f"{prefix}_ema_{span}"] = shifted.ewm(span=span, adjust=False).mean()

    calendar_rows = [_calendar_features(pd.Timestamp(date), context) for date in pd.to_datetime(dates)]
    feats = pd.concat([feats, pd.DataFrame(calendar_rows, index=target_series.index)], axis=1)
    feature_cols = feats.columns.tolist()
    return feats, feature_cols


def _build_revenue_level_frame(
    history_daily: pd.DataFrame,
    context: FeatureContext,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.Series, list[str]]:
    revenue = history_daily["Revenue"].astype(float)
    feats, feature_cols = _build_autoreg_base_frame(history_daily["date"], revenue, context, "rev")
    combo = feats.copy()
    combo["target"] = np.log1p(revenue)
    combo["actual"] = revenue
    combo["date"] = pd.to_datetime(history_daily["date"])
    combo = combo.dropna().reset_index(drop=True)
    return (
        combo[feature_cols],
        combo["target"].to_numpy(dtype=float),
        combo["actual"].to_numpy(dtype=float),
        combo["date"],
        feature_cols,
    )


def _build_revenue_diff_frame(
    history_daily: pd.DataFrame,
    context: FeatureContext,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.Series, list[str]]:
    revenue = history_daily["Revenue"].astype(float)
    feats, feature_cols = _build_autoreg_base_frame(history_daily["date"], revenue, context, "rev")
    combo = feats.copy()
    combo["target"] = revenue.diff()
    combo["actual"] = revenue
    combo["date"] = pd.to_datetime(history_daily["date"])
    combo = combo.dropna().reset_index(drop=True)
    return (
        combo[feature_cols],
        combo["target"].to_numpy(dtype=float),
        combo["actual"].to_numpy(dtype=float),
        combo["date"],
        feature_cols,
    )


def _build_ratio_frame(
    history_daily: pd.DataFrame,
    context: FeatureContext,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.Series, list[str]]:
    ratio = _safe_ratio(history_daily["COGS"], history_daily["Revenue"]).clip(RATIO_FLOOR, RATIO_CEIL)
    revenue = history_daily["Revenue"].astype(float)
    feats, feature_cols = _build_autoreg_base_frame(history_daily["date"], ratio, context, "ratio")
    for lag in (1, 7, 14, 28):
        feats[f"rev_lag_{lag}"] = revenue.shift(lag)
    feats["rev_roll_mean_14"] = revenue.shift(1).rolling(14).mean()
    feats["ratio_rev_interact"] = feats["ratio_lag_1"] * feats["rev_lag_1"]
    feature_cols = feats.columns.tolist()
    combo = feats.copy()
    combo["target"] = np.log(ratio)
    combo["actual"] = ratio
    combo["date"] = pd.to_datetime(history_daily["date"])
    combo = combo.dropna().reset_index(drop=True)
    return (
        combo[feature_cols],
        combo["target"].to_numpy(dtype=float),
        combo["actual"].to_numpy(dtype=float),
        combo["date"],
        feature_cols,
    )


def _build_component_frame(
    history_daily: pd.DataFrame,
    context: FeatureContext,
    component: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.Series, list[str]]:
    series = history_daily[component].astype(float)
    rows = [_calendar_features(pd.Timestamp(date), context) for date in pd.to_datetime(history_daily["date"])]
    feats = pd.DataFrame(rows)
    feats["day_x_payday"] = feats["day_of_month"] * feats["is_payday"]
    feats["promo_x_month"] = feats["promo_active_prior"] * feats["month"]
    feature_cols = feats.columns.tolist()
    combo = feats.copy()
    combo["target"] = np.log1p(series.clip(lower=0.0))
    combo["actual"] = series.clip(lower=0.0)
    combo["date"] = pd.to_datetime(history_daily["date"])
    combo = combo.dropna().reset_index(drop=True)
    return (
        combo[feature_cols],
        combo["target"].to_numpy(dtype=float),
        combo["actual"].to_numpy(dtype=float),
        combo["date"],
        feature_cols,
    )


def _level_inverse_batch(pred: np.ndarray, _: pd.DataFrame) -> np.ndarray:
    return np.maximum(np.expm1(pred), 0.0)


def _level_inverse_scalar(pred: float, _: dict[str, float]) -> float:
    return float(max(np.expm1(pred), 0.0))


def _diff_inverse_batch(pred: np.ndarray, x_frame: pd.DataFrame) -> np.ndarray:
    return np.maximum(x_frame["rev_lag_1"].to_numpy(dtype=float) + pred, 0.0)


def _diff_inverse_scalar(pred: float, row: dict[str, float]) -> float:
    return float(max(row["rev_lag_1"] + pred, 0.0))


def _ratio_inverse_batch(pred: np.ndarray, _: pd.DataFrame) -> np.ndarray:
    return np.clip(np.exp(pred), RATIO_FLOOR, RATIO_CEIL)


def _ratio_inverse_scalar(pred: float, _: dict[str, float]) -> float:
    return float(np.clip(np.exp(pred), RATIO_FLOOR, RATIO_CEIL))


def _component_inverse_batch(pred: np.ndarray, _: pd.DataFrame) -> np.ndarray:
    return np.maximum(np.expm1(pred), 0.0)


def _component_inverse_scalar(pred: float, _: dict[str, float]) -> float:
    return float(max(np.expm1(pred), 0.0))


def _specs() -> dict[str, HeadSpec]:
    return {
        "level": HeadSpec(
            name="level",
            xgb_params={
                "n_estimators": 240,
                "learning_rate": 0.04,
                "max_depth": 6,
                "subsample": 0.86,
                "colsample_bytree": 0.84,
                "reg_alpha": 0.05,
                "reg_lambda": 1.0,
                "objective": "reg:squarederror",
                "random_state": SEED,
                "n_jobs": 4,
                "verbosity": 0,
                "tree_method": "hist",
            },
            lgb_params={
                "n_estimators": 240,
                "learning_rate": 0.04,
                "num_leaves": 63,
                "max_depth": 6,
                "subsample": 0.86,
                "subsample_freq": 1,
                "colsample_bytree": 0.84,
                "reg_alpha": 0.05,
                "reg_lambda": 1.0,
                "objective": "huber",
                "random_state": SEED,
                "n_jobs": 4,
                "verbose": -1,
            },
            transform_target=lambda y: np.log1p(y.clip(lower=0.0)),
            inverse_batch=_level_inverse_batch,
            inverse_scalar=_level_inverse_scalar,
            clip_low=0.0,
            clip_high=8.0e7,
        ),
        "diff": HeadSpec(
            name="diff",
            xgb_params={
                "n_estimators": 220,
                "learning_rate": 0.05,
                "max_depth": 5,
                "subsample": 0.86,
                "colsample_bytree": 0.84,
                "reg_alpha": 0.05,
                "reg_lambda": 1.2,
                "objective": "reg:squarederror",
                "random_state": SEED,
                "n_jobs": 4,
                "verbosity": 0,
                "tree_method": "hist",
            },
            lgb_params={
                "n_estimators": 220,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "max_depth": 5,
                "subsample": 0.86,
                "subsample_freq": 1,
                "colsample_bytree": 0.84,
                "reg_alpha": 0.05,
                "reg_lambda": 1.2,
                "objective": "huber",
                "random_state": SEED,
                "n_jobs": 4,
                "verbose": -1,
            },
            transform_target=lambda y: y.astype(float),
            inverse_batch=_diff_inverse_batch,
            inverse_scalar=_diff_inverse_scalar,
            clip_low=0.0,
            clip_high=8.0e7,
        ),
        "ratio": HeadSpec(
            name="ratio",
            xgb_params={
                "n_estimators": 200,
                "learning_rate": 0.05,
                "max_depth": 4,
                "subsample": 0.86,
                "colsample_bytree": 0.84,
                "reg_alpha": 0.05,
                "reg_lambda": 1.2,
                "objective": "reg:squarederror",
                "random_state": SEED,
                "n_jobs": 4,
                "verbosity": 0,
                "tree_method": "hist",
            },
            lgb_params={
                "n_estimators": 200,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "max_depth": 4,
                "subsample": 0.86,
                "subsample_freq": 1,
                "colsample_bytree": 0.84,
                "reg_alpha": 0.05,
                "reg_lambda": 1.2,
                "objective": "huber",
                "random_state": SEED,
                "n_jobs": 4,
                "verbose": -1,
            },
            transform_target=lambda y: np.log(y.clip(lower=RATIO_FLOOR, upper=RATIO_CEIL)),
            inverse_batch=_ratio_inverse_batch,
            inverse_scalar=_ratio_inverse_scalar,
            clip_low=RATIO_FLOOR,
            clip_high=RATIO_CEIL,
        ),
        "component": HeadSpec(
            name="component",
            xgb_params={
                "n_estimators": 200,
                "learning_rate": 0.04,
                "max_depth": 5,
                "subsample": 0.88,
                "colsample_bytree": 0.86,
                "reg_alpha": 0.05,
                "reg_lambda": 1.0,
                "objective": "reg:squarederror",
                "random_state": SEED,
                "n_jobs": 4,
                "verbosity": 0,
                "tree_method": "hist",
            },
            lgb_params={
                "n_estimators": 200,
                "learning_rate": 0.04,
                "num_leaves": 31,
                "max_depth": 5,
                "subsample": 0.88,
                "subsample_freq": 1,
                "colsample_bytree": 0.86,
                "reg_alpha": 0.05,
                "reg_lambda": 1.0,
                "objective": "huber",
                "random_state": SEED,
                "n_jobs": 4,
                "verbose": -1,
            },
            transform_target=lambda y: np.log1p(y.clip(lower=0.0)),
            inverse_batch=_component_inverse_batch,
            inverse_scalar=_component_inverse_scalar,
            clip_low=0.0,
            clip_high=8.0e7,
        ),
    }


SPECS = _specs()


def _fit_head(
    x_frame: pd.DataFrame,
    y_transformed: np.ndarray,
    y_actual: np.ndarray,
    dates: pd.Series,
    feature_cols: list[str],
    spec: HeadSpec,
) -> HeadModel:
    splitter = TimeSeriesSplit(n_splits=INNER_SPLITS)
    errors: dict[str, list[float]] = {"xgb": [], "lgb": []}
    for tr_idx, va_idx in splitter.split(x_frame):
        x_tr = x_frame.iloc[tr_idx]
        x_va = x_frame.iloc[va_idx]
        y_tr = y_transformed[tr_idx]
        y_va_actual = y_actual[va_idx]
        weights = _sample_weights(dates.iloc[tr_idx])

        xgb_model = xgb.XGBRegressor(**spec.xgb_params)
        xgb_model.fit(x_tr, y_tr, sample_weight=weights)
        xgb_pred = spec.inverse_batch(xgb_model.predict(x_va), x_va)
        errors["xgb"].append(float(mean_absolute_error(y_va_actual, xgb_pred)))

        lgb_model = lgb.LGBMRegressor(**spec.lgb_params)
        lgb_model.fit(x_tr, y_tr, sample_weight=weights)
        lgb_pred = spec.inverse_batch(lgb_model.predict(x_va), x_va)
        errors["lgb"].append(float(mean_absolute_error(y_va_actual, lgb_pred)))

    mean_errors = {name: float(np.mean(vals)) for name, vals in errors.items()}
    inverse = {name: 1.0 / max(val, 1e-9) for name, val in mean_errors.items()}
    total = sum(inverse.values())
    weights_map = {name: inverse[name] / total for name in inverse}

    full_weights = _sample_weights(dates)
    xgb_full = xgb.XGBRegressor(**spec.xgb_params)
    xgb_full.fit(x_frame, y_transformed, sample_weight=full_weights)
    lgb_full = lgb.LGBMRegressor(**spec.lgb_params)
    lgb_full.fit(x_frame, y_transformed, sample_weight=full_weights)
    return HeadModel(
        spec=spec,
        feature_cols=feature_cols,
        xgb_model=xgb_full,
        lgb_model=lgb_full,
        weights=weights_map,
    )


def _weighted_model_output(
    x_frame: pd.DataFrame,
    head: HeadModel,
    row_cache: dict[str, float] | None = None,
) -> float | np.ndarray:
    xgb_pred = head.xgb_model.predict(x_frame)
    lgb_pred = head.lgb_model.predict(x_frame)
    if row_cache is None:
        natural_xgb = head.spec.inverse_batch(np.asarray(xgb_pred, dtype=float), x_frame)
        natural_lgb = head.spec.inverse_batch(np.asarray(lgb_pred, dtype=float), x_frame)
        return head.weights["xgb"] * natural_xgb + head.weights["lgb"] * natural_lgb
    xgb_nat = head.spec.inverse_scalar(float(xgb_pred[0]), row_cache)
    lgb_nat = head.spec.inverse_scalar(float(lgb_pred[0]), row_cache)
    return head.weights["xgb"] * xgb_nat + head.weights["lgb"] * lgb_nat


def _lag(values: list[float], lag: int) -> float:
    return float(values[-lag]) if len(values) >= lag else float(values[0])


def _rmean(values: list[float], window: int) -> float:
    chunk = values[-window:] if len(values) >= window else values
    return float(np.mean(chunk))


def _rstd(values: list[float], window: int) -> float:
    chunk = values[-window:] if len(values) >= window else values
    return float(np.std(chunk))


def _ema(values: list[float], span: int) -> float:
    alpha = 2.0 / (span + 1.0)
    out = float(values[0])
    for value in values[1:]:
        out = alpha * float(value) + (1.0 - alpha) * out
    return float(out)


def _row_autoreg_dict(
    date: pd.Timestamp,
    history: list[float],
    context: FeatureContext,
    prefix: str,
) -> dict[str, float]:
    row: dict[str, float] = {}
    for lag in LAGS:
        row[f"{prefix}_lag_{lag}"] = _lag(history, lag)
    for window in ROLL_WINDOWS:
        row[f"{prefix}_roll_mean_{window}"] = _rmean(history, window)
        row[f"{prefix}_roll_std_{window}"] = _rstd(history, window)
    for span in EMA_SPANS:
        row[f"{prefix}_ema_{span}"] = _ema(history, span)
    row.update(_calendar_features(date, context))
    return row


def _predict_level_head(
    head: HeadModel,
    history_daily: pd.DataFrame,
    dates: pd.Series,
    context: FeatureContext,
) -> np.ndarray:
    revenue_hist = history_daily["Revenue"].astype(float).tolist()
    preds: list[float] = []
    for date in pd.to_datetime(dates):
        row = _row_autoreg_dict(pd.Timestamp(date), revenue_hist, context, "rev")
        x_row = pd.DataFrame([row])[head.feature_cols]
        pred = float(_weighted_model_output(x_row, head, row))
        pred = float(np.clip(pred, head.spec.clip_low, head.spec.clip_high))
        preds.append(pred)
        revenue_hist.append(pred)
    return np.asarray(preds, dtype=float)


def _predict_diff_head(
    head: HeadModel,
    history_daily: pd.DataFrame,
    dates: pd.Series,
    context: FeatureContext,
) -> np.ndarray:
    revenue_hist = history_daily["Revenue"].astype(float).tolist()
    preds: list[float] = []
    for date in pd.to_datetime(dates):
        row = _row_autoreg_dict(pd.Timestamp(date), revenue_hist, context, "rev")
        x_row = pd.DataFrame([row])[head.feature_cols]
        pred = float(_weighted_model_output(x_row, head, row))
        pred = float(np.clip(pred, head.spec.clip_low, head.spec.clip_high))
        preds.append(pred)
        revenue_hist.append(pred)
    return np.asarray(preds, dtype=float)


def _predict_ratio_head(
    head: HeadModel,
    history_daily: pd.DataFrame,
    dates: pd.Series,
    revenue_path: np.ndarray,
    context: FeatureContext,
) -> np.ndarray:
    ratio_hist = _safe_ratio(history_daily["COGS"], history_daily["Revenue"]).clip(RATIO_FLOOR, RATIO_CEIL)
    ratio_values = ratio_hist.astype(float).tolist()
    revenue_hist = history_daily["Revenue"].astype(float).tolist()
    preds: list[float] = []
    for i, date in enumerate(pd.to_datetime(dates)):
        row = _row_autoreg_dict(pd.Timestamp(date), ratio_values, context, "ratio")
        for lag in (1, 7, 14, 28):
            row[f"rev_lag_{lag}"] = _lag(revenue_hist, lag)
        row["rev_roll_mean_14"] = _rmean(revenue_hist, 14)
        row["ratio_rev_interact"] = row["ratio_lag_1"] * row["rev_lag_1"]
        x_row = pd.DataFrame([row])[head.feature_cols]
        pred = float(_weighted_model_output(x_row, head, row))
        pred = float(np.clip(pred, RATIO_FLOOR, RATIO_CEIL))
        preds.append(pred)
        ratio_values.append(pred)
        revenue_hist.append(float(revenue_path[i]))
    return np.asarray(preds, dtype=float)


def _predict_component_head(
    head: HeadModel,
    dates: pd.Series,
    context: FeatureContext,
) -> np.ndarray:
    rows = []
    for date in pd.to_datetime(dates):
        row = _calendar_features(pd.Timestamp(date), context)
        row["day_x_payday"] = row["day_of_month"] * row["is_payday"]
        row["promo_x_month"] = row["promo_active_prior"] * row["month"]
        rows.append(row)
    x_frame = pd.DataFrame(rows)[head.feature_cols]
    pred = _weighted_model_output(x_frame, head)
    return np.clip(np.asarray(pred, dtype=float), head.spec.clip_low, head.spec.clip_high)


def _build_component_history(history_daily: pd.DataFrame) -> pd.DataFrame:
    frame = history_daily[["date", "Revenue", "orders"]].copy()
    frame["total_orders"] = frame["orders"].fillna(0.0).clip(lower=0.0)
    frame["AOV"] = (
        frame["Revenue"] / frame["total_orders"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    aov_fill = float(frame["AOV"].median()) if frame["AOV"].notna().any() else 0.0
    frame["AOV"] = frame["AOV"].fillna(aov_fill).clip(lower=0.0)
    return frame


def _fit_fold_heads(
    train_history: pd.DataFrame,
    promotions: pd.DataFrame,
    customers: pd.DataFrame,
) -> tuple[FeatureContext, HeadModel, HeadModel, HeadModel, HeadModel, HeadModel]:
    train_end = pd.Timestamp(train_history["date"].max()).date()
    print(f"[final_model] fitting heads on history through {train_end} ({len(train_history)} days)")
    context = _build_feature_context(train_history, promotions, customers)

    x_level, y_level, actual_level, dates_level, cols_level = _build_revenue_level_frame(train_history, context)
    level_head = _fit_head(x_level, y_level, actual_level, dates_level, cols_level, SPECS["level"])

    x_diff, y_diff, actual_diff, dates_diff, cols_diff = _build_revenue_diff_frame(train_history, context)
    diff_head = _fit_head(x_diff, y_diff, actual_diff, dates_diff, cols_diff, SPECS["diff"])

    x_ratio, y_ratio, actual_ratio, dates_ratio, cols_ratio = _build_ratio_frame(train_history, context)
    ratio_head = _fit_head(x_ratio, y_ratio, actual_ratio, dates_ratio, cols_ratio, SPECS["ratio"])

    component_history = _build_component_history(train_history)
    x_orders, y_orders, actual_orders, dates_orders, cols_orders = _build_component_frame(
        component_history.rename(columns={"date": "date"}), context, "total_orders"
    )
    orders_head = _fit_head(x_orders, y_orders, actual_orders, dates_orders, cols_orders, SPECS["component"])

    x_aov, y_aov, actual_aov, dates_aov, cols_aov = _build_component_frame(
        component_history.rename(columns={"date": "date"}), context, "AOV"
    )
    aov_head = _fit_head(x_aov, y_aov, actual_aov, dates_aov, cols_aov, SPECS["component"])
    return context, level_head, diff_head, ratio_head, orders_head, aov_head


def _time_decay_weights(length: int, start: float, end: float) -> np.ndarray:
    if length <= 1:
        return np.asarray([end], dtype=float)
    return np.linspace(start, end, length, dtype=float)


def _build_revenue_path(
    level_revenue: np.ndarray,
    diff_revenue: np.ndarray,
    oneshot_revenue: np.ndarray,
    beta_diff: float,
    oneshot_start: float,
    oneshot_end: float,
) -> np.ndarray:
    recursive_revenue = (1.0 - beta_diff) * level_revenue + beta_diff * diff_revenue
    decay = _time_decay_weights(len(level_revenue), oneshot_start, oneshot_end)
    return (1.0 - decay) * recursive_revenue + decay * oneshot_revenue


def _fold_prediction_frame(
    train_history: pd.DataFrame,
    val_actual: pd.DataFrame,
    promotions: pd.DataFrame,
    customers: pd.DataFrame,
) -> pd.DataFrame:
    context, level_head, diff_head, ratio_head, orders_head, aov_head = _fit_fold_heads(
        train_history,
        promotions,
        customers,
    )
    val_dates = val_actual["date"].reset_index(drop=True)
    level_revenue = _predict_level_head(level_head, train_history, val_dates, context)
    diff_revenue = _predict_diff_head(diff_head, train_history, val_dates, context)
    oneshot_orders = _predict_component_head(orders_head, val_dates, context)
    oneshot_aov = _predict_component_head(aov_head, val_dates, context)
    oneshot_revenue = np.maximum(oneshot_orders * oneshot_aov, 0.0)
    return pd.DataFrame(
        {
            "Date": val_dates.to_numpy(),
            "Revenue": val_actual["Revenue"].to_numpy(dtype=float),
            "COGS": val_actual["COGS"].to_numpy(dtype=float),
            "level_revenue": level_revenue,
            "diff_revenue": diff_revenue,
            "oneshot_revenue": oneshot_revenue,
            "train_rows": len(train_history),
            "context_ready": 1,
        }
    ), context, ratio_head, level_head, diff_head, orders_head, aov_head


def _evaluate_candidates(
    fold_artifacts: list[tuple[pd.DataFrame, pd.DataFrame, FeatureContext, HeadModel]],
) -> tuple[pd.DataFrame, dict[str, float]]:
    total_candidates = len(BETA_GRID) * len(TIME_DECAY_CONFIGS)
    print(f"[final_model] screening {total_candidates} blend candidates on revenue first")
    rows: list[dict[str, float]] = []
    for beta in BETA_GRID:
        for start, end in TIME_DECAY_CONFIGS:
            all_rev_actual: list[np.ndarray] = []
            all_rev_pred: list[np.ndarray] = []
            for candidate_frame, train_history, context, ratio_head in fold_artifacts:
                recursive_revenue = (
                    (1.0 - beta) * candidate_frame["level_revenue"].to_numpy(dtype=float)
                    + beta * candidate_frame["diff_revenue"].to_numpy(dtype=float)
                )
                one_shot = candidate_frame["oneshot_revenue"].to_numpy(dtype=float)
                decay = _time_decay_weights(len(candidate_frame), start, end)
                final_revenue = (1.0 - decay) * recursive_revenue + decay * one_shot
                all_rev_actual.append(candidate_frame["Revenue"].to_numpy(dtype=float))
                all_rev_pred.append(final_revenue)
            revenue_actual = pd.Series(np.concatenate(all_rev_actual))
            revenue_pred = pd.Series(np.concatenate(all_rev_pred))
            revenue_metrics = _metric_block(revenue_actual, revenue_pred)
            row = {
                "beta_diff": float(beta),
                "oneshot_start": float(start),
                "oneshot_end": float(end),
                "screen_revenue_objective": revenue_metrics["objective"],
                "revenue_mae": revenue_metrics["mae"],
                "revenue_rmse": revenue_metrics["rmse"],
                "revenue_r2": revenue_metrics["r2"],
                "cogs_mae": np.nan,
                "cogs_rmse": np.nan,
                "cogs_r2": np.nan,
                "joint_objective": np.nan,
            }
            rows.append(row)

    comparison = pd.DataFrame(rows).sort_values("screen_revenue_objective").reset_index(drop=True)
    finalists = comparison.head(min(FINALIST_CANDIDATES, len(comparison))).copy()
    print(f"[final_model] evaluating joint revenue + COGS objective on top {len(finalists)} finalists")

    best: dict[str, float] = {"joint_objective": float("inf")}
    for idx, finalist in finalists.iterrows():
        beta = float(finalist["beta_diff"])
        start = float(finalist["oneshot_start"])
        end = float(finalist["oneshot_end"])
        all_rev_actual: list[np.ndarray] = []
        all_cogs_actual: list[np.ndarray] = []
        all_rev_pred: list[np.ndarray] = []
        all_cogs_pred: list[np.ndarray] = []
        for candidate_frame, train_history, context, ratio_head in fold_artifacts:
            recursive_revenue = (
                (1.0 - beta) * candidate_frame["level_revenue"].to_numpy(dtype=float)
                + beta * candidate_frame["diff_revenue"].to_numpy(dtype=float)
            )
            one_shot = candidate_frame["oneshot_revenue"].to_numpy(dtype=float)
            decay = _time_decay_weights(len(candidate_frame), start, end)
            final_revenue = (1.0 - decay) * recursive_revenue + decay * one_shot
            ratio_pred = _predict_ratio_head(
                ratio_head,
                train_history,
                candidate_frame["Date"],
                final_revenue,
                context,
            )
            final_cogs = final_revenue * ratio_pred
            all_rev_actual.append(candidate_frame["Revenue"].to_numpy(dtype=float))
            all_cogs_actual.append(candidate_frame["COGS"].to_numpy(dtype=float))
            all_rev_pred.append(final_revenue)
            all_cogs_pred.append(final_cogs)
        revenue_actual = pd.Series(np.concatenate(all_rev_actual))
        cogs_actual = pd.Series(np.concatenate(all_cogs_actual))
        revenue_pred = pd.Series(np.concatenate(all_rev_pred))
        cogs_pred = pd.Series(np.concatenate(all_cogs_pred))
        revenue_metrics, cogs_metrics, joint = _joint_objective(
            revenue_actual,
            revenue_pred,
            cogs_actual,
            cogs_pred,
        )
        comparison.loc[idx, "revenue_mae"] = revenue_metrics["mae"]
        comparison.loc[idx, "revenue_rmse"] = revenue_metrics["rmse"]
        comparison.loc[idx, "revenue_r2"] = revenue_metrics["r2"]
        comparison.loc[idx, "cogs_mae"] = cogs_metrics["mae"]
        comparison.loc[idx, "cogs_rmse"] = cogs_metrics["rmse"]
        comparison.loc[idx, "cogs_r2"] = cogs_metrics["r2"]
        comparison.loc[idx, "joint_objective"] = joint
        if joint < best["joint_objective"]:
            best = comparison.loc[idx].to_dict()

    comparison = comparison.sort_values(
        by=["joint_objective", "screen_revenue_objective"],
        na_position="last",
    ).reset_index(drop=True)
    comparison["selected"] = 0
    mask = (
        (comparison["beta_diff"] == best["beta_diff"])
        & (comparison["oneshot_start"] == best["oneshot_start"])
        & (comparison["oneshot_end"] == best["oneshot_end"])
    )
    comparison.loc[mask, "selected"] = 1
    return comparison, best


def _candidate_slug(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def _named_candidate_rows(comparison: pd.DataFrame, best: dict[str, float]) -> list[dict[str, float | str]]:
    named: list[dict[str, float | str]] = []
    seen: set[tuple[float, float, float]] = set()

    def add_row(label: str, family: str, methodology: str, row: pd.Series) -> None:
        key = (
            float(row["beta_diff"]),
            float(row["oneshot_start"]),
            float(row["oneshot_end"]),
        )
        if key in seen:
            return
        seen.add(key)
        named.append(
            {
                "candidate_name": label,
                "family": family,
                "methodology": methodology,
                "beta_diff": key[0],
                "oneshot_start": key[1],
                "oneshot_end": key[2],
                "joint_objective": float(row["joint_objective"]),
                "revenue_mae": float(row["revenue_mae"]),
                "revenue_rmse": float(row["revenue_rmse"]),
                "revenue_r2": float(row["revenue_r2"]),
                "cogs_mae": float(row["cogs_mae"]),
                "cogs_rmse": float(row["cogs_rmse"]),
                "cogs_r2": float(row["cogs_r2"]),
            }
        )

    recursive = (
        comparison.loc[
            (comparison["oneshot_start"] == 0.0) & (comparison["oneshot_end"] == 0.0)
        ]
        .sort_values(["joint_objective", "screen_revenue_objective"])
        .head(1)
    )
    if not recursive.empty:
        add_row(
            "recursive_multihead_clean",
            "mouis_recursive_clean",
            "Recursive revenue-level plus separate COGS-ratio head, without component horizon blending.",
            recursive.iloc[0],
        )

    conservative = (
        comparison.loc[
            (comparison["oneshot_end"] > 0.0)
            & (comparison["oneshot_end"] <= 0.50)
        ]
        .sort_values(["joint_objective", "screen_revenue_objective"])
        .head(1)
    )
    if not conservative.empty:
        add_row(
            "time_decay_hybrid_conservative",
            "kcmt_mouis_hybrid",
            "Recursive revenue heads blended with one-shot orders-times-AOV using a conservative time-decay schedule.",
            conservative.iloc[0],
        )

    selected = comparison.loc[comparison["selected"] == 1].head(1)
    if not selected.empty:
        add_row(
            "time_decay_hybrid_selected",
            "kcmt_mouis_hybrid",
            "Best clean internal hybrid: KCMT-style timing features plus Mouis-style recursive and one-shot heads with time-decay blending.",
            selected.iloc[0],
        )

    long_horizon = (
        comparison.loc[comparison["oneshot_end"] >= 0.60]
        .sort_values(["joint_objective", "screen_revenue_objective"])
        .head(2)
    )
    for _, row in long_horizon.iterrows():
        before = len(named)
        add_row(
            "time_decay_hybrid_long_horizon",
            "component_stability_hybrid",
            "Higher late-horizon one-shot weight to reduce recursive drift over the 548-day forecast window.",
            row,
        )
        if len(named) > before:
            break

    return named


def _candidate_path(candidate_name: str) -> Path:
    submissions_dir = final_submission_path().parent
    return submissions_dir / f"{_candidate_slug(candidate_name)}.csv"


def _evaluate_named_candidate_on_folds(
    fold_artifacts: list[tuple[pd.DataFrame, pd.DataFrame, FeatureContext, HeadModel]],
    candidate: dict[str, float | str],
) -> dict[str, float]:
    all_rev_actual: list[np.ndarray] = []
    all_cogs_actual: list[np.ndarray] = []
    all_rev_pred: list[np.ndarray] = []
    all_cogs_pred: list[np.ndarray] = []
    for candidate_frame, train_history, context, ratio_head in fold_artifacts:
        revenue = _build_revenue_path(
            candidate_frame["level_revenue"].to_numpy(dtype=float),
            candidate_frame["diff_revenue"].to_numpy(dtype=float),
            candidate_frame["oneshot_revenue"].to_numpy(dtype=float),
            float(candidate["beta_diff"]),
            float(candidate["oneshot_start"]),
            float(candidate["oneshot_end"]),
        )
        ratio = _predict_ratio_head(
            ratio_head,
            train_history,
            candidate_frame["Date"],
            revenue,
            context,
        )
        all_rev_actual.append(candidate_frame["Revenue"].to_numpy(dtype=float))
        all_cogs_actual.append(candidate_frame["COGS"].to_numpy(dtype=float))
        all_rev_pred.append(revenue)
        all_cogs_pred.append(revenue * ratio)

    revenue_actual = pd.Series(np.concatenate(all_rev_actual))
    cogs_actual = pd.Series(np.concatenate(all_cogs_actual))
    revenue_pred = pd.Series(np.concatenate(all_rev_pred))
    cogs_pred = pd.Series(np.concatenate(all_cogs_pred))
    revenue_metrics, cogs_metrics, joint = _joint_objective(
        revenue_actual,
        revenue_pred,
        cogs_actual,
        cogs_pred,
    )
    return {
        "joint_objective": joint,
        "revenue_mae": revenue_metrics["mae"],
        "revenue_rmse": revenue_metrics["rmse"],
        "revenue_r2": revenue_metrics["r2"],
        "cogs_mae": cogs_metrics["mae"],
        "cogs_rmse": cogs_metrics["rmse"],
        "cogs_r2": cogs_metrics["r2"],
    }


def _build_candidate_frame(
    future_dates: pd.Series,
    history_daily: pd.DataFrame,
    context: FeatureContext,
    ratio_head: HeadModel,
    level_revenue: np.ndarray,
    diff_revenue: np.ndarray,
    oneshot_revenue: np.ndarray,
    candidate: dict[str, float | str],
) -> pd.DataFrame:
    revenue = _build_revenue_path(
        level_revenue,
        diff_revenue,
        oneshot_revenue,
        float(candidate["beta_diff"]),
        float(candidate["oneshot_start"]),
        float(candidate["oneshot_end"]),
    )
    ratio = _predict_ratio_head(
        ratio_head,
        history_daily,
        pd.Series(future_dates),
        revenue,
        context,
    )
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(future_dates),
            "Revenue": np.maximum(revenue, 0.0),
            "COGS": np.maximum(revenue * ratio, 0.0),
        }
    )
    return frame


def _fit_full_heads(
    history_daily: pd.DataFrame,
    promotions: pd.DataFrame,
    customers: pd.DataFrame,
) -> tuple[FeatureContext, HeadModel, HeadModel, HeadModel, HeadModel, HeadModel]:
    return _fit_fold_heads(history_daily, promotions, customers)


def build_final_shape_medium_submission() -> dict[str, object]:
    data = load_dataframes()
    history_daily = build_daily_frame(data)
    promotions = data["promotions"].copy()
    customers = data["customers"].copy()
    future_dates = sample_dates()

    fold_artifacts: list[tuple[pd.DataFrame, pd.DataFrame, FeatureContext, HeadModel]] = []
    fold_frames: list[pd.DataFrame] = []
    last_heads: tuple[FeatureContext, HeadModel, HeadModel, HeadModel, HeadModel, HeadModel] | None = None

    for year in OUTER_YEARS:
        train_history = history_daily.loc[history_daily["date"].dt.year < year].copy().reset_index(drop=True)
        val_actual = history_daily.loc[history_daily["date"].dt.year == year].copy().reset_index(drop=True)
        if train_history.empty or val_actual.empty:
            continue
        print(
            f"[final_model] outer fold {year}: train={len(train_history)} days, "
            f"valid={len(val_actual)} days"
        )
        candidate_frame, context, ratio_head, level_head, diff_head, orders_head, aov_head = _fold_prediction_frame(
            train_history,
            val_actual,
            promotions,
            customers,
        )
        candidate_frame["fold_year"] = year
        fold_frames.append(candidate_frame)
        fold_artifacts.append((candidate_frame, train_history, context, ratio_head))
        last_heads = (context, level_head, diff_head, ratio_head, orders_head, aov_head)

    if not fold_artifacts or last_heads is None:
        raise RuntimeError("Could not build validation folds for final model selection.")

    comparison, best = _evaluate_candidates(fold_artifacts)
    fold_predictions = pd.concat(fold_frames, ignore_index=True)
    print(
        "[final_model] selected config:",
        f"beta_diff={best['beta_diff']:.2f}",
        f"oneshot_start={best['oneshot_start']:.2f}",
        f"oneshot_end={best['oneshot_end']:.2f}",
        f"joint={best['joint_objective']:.4f}",
    )

    context, level_head, diff_head, ratio_head, orders_head, aov_head = _fit_full_heads(
        history_daily,
        promotions,
        customers,
    )
    level_revenue = _predict_level_head(level_head, history_daily, pd.Series(future_dates), context)
    diff_revenue = _predict_diff_head(diff_head, history_daily, pd.Series(future_dates), context)
    orders_pred = _predict_component_head(orders_head, pd.Series(future_dates), context)
    aov_pred = _predict_component_head(aov_head, pd.Series(future_dates), context)
    oneshot_revenue = np.maximum(orders_pred * aov_pred, 0.0)

    candidate_rows = _named_candidate_rows(comparison, best)
    candidate_rows = [
        {**candidate, **_evaluate_named_candidate_on_folds(fold_artifacts, candidate)}
        for candidate in candidate_rows
    ]
    candidate_frames: dict[str, pd.DataFrame] = {}
    candidate_paths: dict[str, Path] = {}
    for candidate in candidate_rows:
        candidate_frame = _build_candidate_frame(
            pd.Series(future_dates),
            history_daily,
            context,
            ratio_head,
            level_revenue,
            diff_revenue,
            oneshot_revenue,
            candidate,
        )
        candidate_name = str(candidate["candidate_name"])
        candidate_path = _candidate_path(candidate_name)
        candidate_frame.to_csv(candidate_path, index=False, float_format="%.2f")
        candidate_frames[candidate_name] = candidate_frame
        candidate_paths[candidate_name] = candidate_path

    final_frame = candidate_frames["time_decay_hybrid_selected"]

    final_path = final_submission_path()
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_frame.to_csv(final_path, index=False, float_format="%.2f")
    candidate_table = pd.DataFrame(candidate_rows)
    candidate_table["submission_file"] = candidate_table["candidate_name"].map(
        lambda name: str(candidate_paths[str(name)])
    )
    candidate_table["selected"] = (
        candidate_table["candidate_name"] == "time_decay_hybrid_selected"
    ).astype(int)
    return {
        "final_submission": final_path,
        "candidate_model_table": candidate_table,
        "candidate_submissions": candidate_paths,
        "selected_objective": float(best["joint_objective"]),
    }


def main() -> None:
    outputs = build_final_shape_medium_submission()
    print(str(outputs["final_submission"]))


if __name__ == "__main__":
    main()

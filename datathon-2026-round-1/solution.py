"""
Single-model, leak-free forecasting pipeline for Datathon 2026 Round 1.

What this script does:
1. Runs long-horizon (548-day) rolling backtests with recursive inference.
2. Compares multiple valid feature sets (no unavailable future inputs).
3. Selects the best feature set by backtest MAE/RMSE/R2.
4. Fits final LightGBM models (Revenue + Margin) on full training history.
5. Generates `dataset/submission.csv` and experiment/importance artifacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SEED = 42
DATA_DIR = Path("dataset")

BACKTEST_CUTOFFS = [
    pd.Timestamp("2018-12-31"),
    pd.Timestamp("2019-12-31"),
    pd.Timestamp("2020-12-31"),
]
BACKTEST_HORIZON_DAYS = 548

TET = {
    2012: ("01-22", "01-28"),
    2013: ("02-09", "02-15"),
    2014: ("01-30", "02-05"),
    2015: ("02-17", "02-23"),
    2016: ("02-07", "02-13"),
    2017: ("01-27", "02-02"),
    2018: ("02-14", "02-20"),
    2019: ("02-02", "02-08"),
    2020: ("01-23", "01-29"),
    2021: ("02-10", "02-16"),
    2022: ("01-29", "02-04"),
    2023: ("01-20", "01-26"),
    2024: ("02-08", "02-14"),
}


@dataclass(frozen=True)
class FeatureSet:
    name: str
    target_lags: tuple[int, ...]
    roll_windows: tuple[int, ...]
    biz_lags: tuple[int, ...]
    use_yoy_ratio: bool = True
    use_trend: bool = True


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true.values, y_pred.values),
        "r2": float(r2_score(y_true, y_pred)),
    }


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sales = pd.read_csv(DATA_DIR / "sales.csv", parse_dates=["Date"]).sort_values("Date")
    test = pd.read_csv(DATA_DIR / "sample_submission.csv", parse_dates=["Date"]).sort_values("Date")
    biz = pd.read_csv(DATA_DIR / "daily_business_features.csv", parse_dates=["Date"]).sort_values("Date")
    return sales, test, biz


def build_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.DataFrame(index=index)
    df["year"] = index.year
    df["month"] = index.month
    df["day"] = index.day
    df["dow"] = index.dayofweek
    df["doy"] = index.dayofyear
    df["quarter"] = index.quarter
    df["week"] = index.isocalendar().week.astype(int).values
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_month_start"] = (df["day"] <= 3).astype(int)
    df["is_month_end"] = (((index + pd.offsets.MonthEnd(0)) - index).days <= 2).astype(int)

    for k in (1, 2, 3, 4):
        df[f"sin_y_{k}"] = np.sin(2.0 * np.pi * k * df["doy"] / 365.25)
        df[f"cos_y_{k}"] = np.cos(2.0 * np.pi * k * df["doy"] / 365.25)
    for k in (1, 2):
        df[f"sin_w_{k}"] = np.sin(2.0 * np.pi * k * df["dow"] / 7.0)
        df[f"cos_w_{k}"] = np.cos(2.0 * np.pi * k * df["dow"] / 7.0)
    for k in (1, 2):
        df[f"sin_m_{k}"] = np.sin(2.0 * np.pi * k * df["day"] / 30.44)
        df[f"cos_m_{k}"] = np.cos(2.0 * np.pi * k * df["day"] / 30.44)

    df["is_tet"] = 0
    df["is_tet_pre"] = 0
    df["is_tet_post"] = 0
    df["is_holiday"] = 0

    for yr, (start, end) in TET.items():
        ts = pd.Timestamp(f"{yr}-{start}")
        te = pd.Timestamp(f"{yr}-{end}")
        df.loc[(df.index >= ts) & (df.index <= te), "is_tet"] = 1
        df.loc[(df.index >= ts - pd.Timedelta(days=7)) & (df.index < ts), "is_tet_pre"] = 1
        df.loc[(df.index > te) & (df.index <= te + pd.Timedelta(days=7)), "is_tet_post"] = 1
        for md in ("01-01", "04-30", "05-01", "09-02"):
            h = pd.Timestamp(f"{yr}-{md}")
            if h in df.index:
                df.loc[h, "is_holiday"] = 1

    return df


def build_feature_row(
    dt: pd.Timestamp,
    history: Dict[pd.Timestamp, float],
    calendar: pd.DataFrame,
    biz_lookup: Dict[str, Dict[pd.Timestamp, float]],
    biz_cols: Iterable[str],
    cfg: FeatureSet,
    trend_lookup: Dict[pd.Timestamp, float] | None = None,
) -> Dict[str, float]:
    row = calendar.loc[dt].to_dict()

    for lag in cfg.target_lags:
        row[f"lag_{lag}"] = history.get(dt - pd.Timedelta(days=lag), np.nan)

    for w in cfg.roll_windows:
        vals = [history.get(dt - pd.Timedelta(days=i), np.nan) for i in range(1, w + 1)]
        vals = np.array(vals, dtype=float)
        row[f"roll{w}_mean"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
        row[f"roll{w}_std"] = float(np.nanstd(vals)) if np.isfinite(vals).any() else np.nan

    if cfg.use_yoy_ratio:
        lag365 = history.get(dt - pd.Timedelta(days=365), np.nan)
        lag730 = history.get(dt - pd.Timedelta(days=730), np.nan)
        if np.isfinite(lag365) and np.isfinite(lag730) and abs(lag730) > 1e-9:
            row["yoy_365_730"] = lag365 / lag730
        else:
            row["yoy_365_730"] = np.nan

    if cfg.use_trend and trend_lookup is not None:
        tr = trend_lookup.get(dt, np.nan)
        tr_365 = trend_lookup.get(dt - pd.Timedelta(days=365), np.nan)
        row["trend_pred"] = tr
        if np.isfinite(tr) and np.isfinite(tr_365) and abs(tr_365) > 1e-9:
            row["trend_ratio_365"] = tr / tr_365
        else:
            row["trend_ratio_365"] = np.nan

    for blag in cfg.biz_lags:
        bdate = dt - pd.Timedelta(days=blag)
        for col in biz_cols:
            row[f"{col}_L{blag}"] = biz_lookup[col].get(bdate, np.nan)

    return row


def required_history_available(
    dt: pd.Timestamp,
    history: Dict[pd.Timestamp, float],
    cfg: FeatureSet,
    biz_index_set: set[pd.Timestamp],
) -> bool:
    for lag in cfg.target_lags:
        if (dt - pd.Timedelta(days=lag)) not in history:
            return False
    for blag in cfg.biz_lags:
        if (dt - pd.Timedelta(days=blag)) not in biz_index_set:
            return False
    return True


def make_train_matrix(
    dates: pd.DatetimeIndex,
    target_series: pd.Series,
    calendar: pd.DataFrame,
    biz_lookup: Dict[str, Dict[pd.Timestamp, float]],
    biz_cols: List[str],
    cfg: FeatureSet,
    biz_index_set: set[pd.Timestamp],
    trend_lookup: Dict[pd.Timestamp, float] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    history = {pd.Timestamp(d): float(v) for d, v in target_series.items()}
    feats: List[Dict[str, float]] = []
    ys: List[float] = []
    idx: List[pd.Timestamp] = []

    for dt in dates:
        dt = pd.Timestamp(dt)
        if dt not in history:
            continue
        if not required_history_available(dt, history, cfg, biz_index_set):
            continue
        feats.append(
            build_feature_row(
                dt,
                history,
                calendar,
                biz_lookup,
                biz_cols,
                cfg,
                trend_lookup=trend_lookup,
            )
        )
        ys.append(history[dt])
        idx.append(dt)

    X = pd.DataFrame(feats, index=pd.DatetimeIndex(idx))
    y = pd.Series(ys, index=X.index, name=target_series.name)
    return X, y


def fit_lgbm(
    X: pd.DataFrame, y: pd.Series, seed: int, log_target: bool = False
) -> tuple[lgb.LGBMRegressor, pd.Series]:
    med = X.median(numeric_only=True)
    Xf = X.fillna(med)
    model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=1400,
        learning_rate=0.025,
        num_leaves=63,
        min_child_samples=25,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=0.3,
        random_state=seed,
        verbosity=-1,
    )
    y_fit = np.log1p(y.values) if log_target else y.values
    model.fit(Xf, y_fit)
    return model, med


def fit_ridge(
    X: pd.DataFrame,
    y: pd.Series,
    alpha: float = 16.0,
    log_target: bool = False,
) -> tuple[object, pd.Series]:
    med = X.median(numeric_only=True)
    Xf = X.fillna(med)
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    y_fit = np.log1p(y.values) if log_target else y.values
    model.fit(Xf, y_fit)
    return model, med


def recursive_forecast(
    model: object,
    med: pd.Series,
    horizon_dates: pd.DatetimeIndex,
    history: Dict[pd.Timestamp, float],
    calendar: pd.DataFrame,
    biz_lookup: Dict[str, Dict[pd.Timestamp, float]],
    biz_cols: List[str],
    cfg: FeatureSet,
    trend_lookup: Dict[pd.Timestamp, float] | None = None,
    log_target: bool = False,
) -> pd.Series:
    preds = []
    hist = dict(history)
    for dt in horizon_dates:
        dt = pd.Timestamp(dt)
        row = build_feature_row(
            dt,
            hist,
            calendar,
            biz_lookup,
            biz_cols,
            cfg,
            trend_lookup=trend_lookup,
        )
        x = pd.DataFrame([row], index=[dt]).fillna(med)
        yhat = float(model.predict(x)[0])
        if log_target:
            yhat = float(np.expm1(yhat))
        hist[dt] = yhat
        preds.append(yhat)
    return pd.Series(preds, index=horizon_dates)


def fit_trend_lookup(
    train_series: pd.Series, target_dates: pd.DatetimeIndex, clip_min: float = 0.0
) -> Dict[pd.Timestamp, float]:
    train_series = train_series.sort_index()
    x = train_series.index.map(pd.Timestamp.toordinal).astype(float).to_numpy()
    y = np.log1p(np.maximum(train_series.values, clip_min))
    coeff = np.polyfit(x, y, deg=2)
    x_all = pd.DatetimeIndex(target_dates).map(pd.Timestamp.toordinal).astype(float).to_numpy()
    y_all = np.polyval(coeff, x_all)
    pred_all = np.expm1(y_all)
    pred_all = np.clip(pred_all, clip_min, None)
    return {pd.Timestamp(d): float(v) for d, v in zip(pd.DatetimeIndex(target_dates), pred_all)}


def run_backtest_for_set(
    sales: pd.DataFrame,
    calendar: pd.DataFrame,
    biz_lookup: Dict[str, Dict[pd.Timestamp, float]],
    biz_cols: List[str],
    cfg: FeatureSet,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sales = sales.copy().sort_values("Date")
    sales_idx = pd.DatetimeIndex(sales["Date"])
    sales = sales.set_index("Date")
    biz_index_set = set(sales_idx)
    margin = (sales["COGS"] / sales["Revenue"]).clip(0.40, 0.98)

    fold_rows = []
    agg_actual_rev = []
    agg_pred_rev = []
    agg_actual_cogs = []
    agg_pred_cogs = []

    for fold_id, cutoff in enumerate(BACKTEST_CUTOFFS, start=1):
        val_start = cutoff + pd.Timedelta(days=1)
        val_end = val_start + pd.Timedelta(days=BACKTEST_HORIZON_DAYS - 1)
        val_dates = sales_idx[(sales_idx >= val_start) & (sales_idx <= val_end)]
        if len(val_dates) != BACKTEST_HORIZON_DAYS:
            continue

        train_dates = sales_idx[sales_idx <= cutoff]
        train_rev = sales.loc[train_dates, "Revenue"]
        train_margin = margin.loc[train_dates]
        fold_dates = pd.DatetimeIndex(train_dates.union(val_dates))

        rev_trend_lookup = fit_trend_lookup(train_rev, fold_dates, clip_min=100.0)
        margin_trend_lookup = fit_trend_lookup(train_margin, fold_dates, clip_min=0.01)

        X_rev, y_rev = make_train_matrix(
            train_dates,
            train_rev,
            calendar,
            biz_lookup,
            biz_cols,
            cfg,
            biz_index_set,
            trend_lookup=rev_trend_lookup,
        )
        rev_model, rev_med = fit_lgbm(X_rev, y_rev, seed=SEED + fold_id, log_target=True)
        rev_pred = recursive_forecast(
            rev_model,
            rev_med,
            val_dates,
            {pd.Timestamp(d): float(v) for d, v in train_rev.items()},
            calendar,
            biz_lookup,
            biz_cols,
            cfg,
            trend_lookup=rev_trend_lookup,
            log_target=True,
        )
        rev_pred = rev_pred.clip(lower=100.0)

        X_margin, y_margin = make_train_matrix(
            train_dates,
            train_margin,
            calendar,
            biz_lookup,
            biz_cols,
            cfg,
            biz_index_set,
            trend_lookup=margin_trend_lookup,
        )
        margin_model, margin_med = fit_lgbm(X_margin, y_margin, seed=SEED + 100 + fold_id)
        margin_pred = recursive_forecast(
            margin_model,
            margin_med,
            val_dates,
            {pd.Timestamp(d): float(v) for d, v in train_margin.items()},
            calendar,
            biz_lookup,
            biz_cols,
            cfg,
            trend_lookup=margin_trend_lookup,
        )
        margin_pred = margin_pred.clip(lower=0.40, upper=0.98)

        cogs_pred = (rev_pred * margin_pred).clip(lower=50.0)
        cogs_pred = np.minimum(cogs_pred, rev_pred * 0.99)
        cogs_pred = pd.Series(cogs_pred, index=val_dates)

        actual_rev = sales.loc[val_dates, "Revenue"]
        actual_cogs = sales.loc[val_dates, "COGS"]

        rev_m = compute_metrics(actual_rev, rev_pred)
        cogs_m = compute_metrics(actual_cogs, cogs_pred)

        fold_rows.append(
            {
                "feature_set": cfg.name,
                "fold": fold_id,
                "cutoff": cutoff.strftime("%Y-%m-%d"),
                "horizon_days": len(val_dates),
                "rev_mae": rev_m["mae"],
                "rev_rmse": rev_m["rmse"],
                "rev_r2": rev_m["r2"],
                "cogs_mae": cogs_m["mae"],
                "cogs_rmse": cogs_m["rmse"],
                "cogs_r2": cogs_m["r2"],
            }
        )

        agg_actual_rev.append(actual_rev)
        agg_pred_rev.append(rev_pred)
        agg_actual_cogs.append(actual_cogs)
        agg_pred_cogs.append(cogs_pred)

    fold_df = pd.DataFrame(fold_rows)
    if fold_df.empty:
        return fold_df, fold_df

    rev_all_actual = pd.concat(agg_actual_rev).sort_index()
    rev_all_pred = pd.concat(agg_pred_rev).sort_index()
    cogs_all_actual = pd.concat(agg_actual_cogs).sort_index()
    cogs_all_pred = pd.concat(agg_pred_cogs).sort_index()

    rev_total = compute_metrics(rev_all_actual, rev_all_pred)
    cogs_total = compute_metrics(cogs_all_actual, cogs_all_pred)
    summary = pd.DataFrame(
        [
            {
                "feature_set": cfg.name,
                "n_folds": int(fold_df["fold"].nunique()),
                "horizon_days_total": int(len(rev_all_actual)),
                "rev_mae": rev_total["mae"],
                "rev_rmse": rev_total["rmse"],
                "rev_r2": rev_total["r2"],
                "cogs_mae": cogs_total["mae"],
                "cogs_rmse": cogs_total["rmse"],
                "cogs_r2": cogs_total["r2"],
            }
        ]
    )
    return fold_df, summary


def calibrate_blend_for_feature_set(
    sales: pd.DataFrame,
    calendar: pd.DataFrame,
    biz_lookup: Dict[str, Dict[pd.Timestamp, float]],
    biz_cols: List[str],
    cfg: FeatureSet,
) -> Dict[str, float]:
    sales = sales.copy().sort_values("Date")
    sales_idx = pd.DatetimeIndex(sales["Date"])
    sales = sales.set_index("Date")
    biz_index_set = set(sales_idx)
    margin = (sales["COGS"] / sales["Revenue"]).clip(0.40, 0.98)

    agg_actual_rev: List[pd.Series] = []
    agg_actual_cogs: List[pd.Series] = []
    agg_rev_lgb: List[pd.Series] = []
    agg_rev_ridge: List[pd.Series] = []
    agg_margin_lgb: List[pd.Series] = []
    agg_margin_ridge: List[pd.Series] = []

    valid_folds = 0
    for fold_id, cutoff in enumerate(BACKTEST_CUTOFFS, start=1):
        val_start = cutoff + pd.Timedelta(days=1)
        val_end = val_start + pd.Timedelta(days=BACKTEST_HORIZON_DAYS - 1)
        val_dates = sales_idx[(sales_idx >= val_start) & (sales_idx <= val_end)]
        if len(val_dates) != BACKTEST_HORIZON_DAYS:
            continue
        valid_folds += 1

        train_dates = sales_idx[sales_idx <= cutoff]
        train_rev = sales.loc[train_dates, "Revenue"]
        train_margin = margin.loc[train_dates]
        fold_dates = pd.DatetimeIndex(train_dates.union(val_dates))

        rev_trend_lookup = fit_trend_lookup(train_rev, fold_dates, clip_min=100.0)
        margin_trend_lookup = fit_trend_lookup(train_margin, fold_dates, clip_min=0.01)

        X_rev, y_rev = make_train_matrix(
            train_dates,
            train_rev,
            calendar,
            biz_lookup,
            biz_cols,
            cfg,
            biz_index_set,
            trend_lookup=rev_trend_lookup,
        )
        rev_lgb_model, rev_lgb_med = fit_lgbm(X_rev, y_rev, seed=SEED + fold_id, log_target=True)
        rev_ridge_model, rev_ridge_med = fit_ridge(X_rev, y_rev, log_target=True)
        history_rev = {pd.Timestamp(d): float(v) for d, v in train_rev.items()}

        rev_pred_lgb = recursive_forecast(
            rev_lgb_model,
            rev_lgb_med,
            val_dates,
            history_rev,
            calendar,
            biz_lookup,
            biz_cols,
            cfg,
            trend_lookup=rev_trend_lookup,
            log_target=True,
        ).clip(lower=100.0)
        rev_pred_ridge = recursive_forecast(
            rev_ridge_model,
            rev_ridge_med,
            val_dates,
            history_rev,
            calendar,
            biz_lookup,
            biz_cols,
            cfg,
            trend_lookup=rev_trend_lookup,
            log_target=True,
        ).clip(lower=100.0)

        X_margin, y_margin = make_train_matrix(
            train_dates,
            train_margin,
            calendar,
            biz_lookup,
            biz_cols,
            cfg,
            biz_index_set,
            trend_lookup=margin_trend_lookup,
        )
        margin_lgb_model, margin_lgb_med = fit_lgbm(
            X_margin, y_margin, seed=SEED + 100 + fold_id
        )
        margin_ridge_model, margin_ridge_med = fit_ridge(X_margin, y_margin)
        history_margin = {pd.Timestamp(d): float(v) for d, v in train_margin.items()}

        margin_pred_lgb = recursive_forecast(
            margin_lgb_model,
            margin_lgb_med,
            val_dates,
            history_margin,
            calendar,
            biz_lookup,
            biz_cols,
            cfg,
            trend_lookup=margin_trend_lookup,
        ).clip(lower=0.40, upper=0.98)
        margin_pred_ridge = recursive_forecast(
            margin_ridge_model,
            margin_ridge_med,
            val_dates,
            history_margin,
            calendar,
            biz_lookup,
            biz_cols,
            cfg,
            trend_lookup=margin_trend_lookup,
        ).clip(lower=0.40, upper=0.98)

        agg_actual_rev.append(sales.loc[val_dates, "Revenue"])
        agg_actual_cogs.append(sales.loc[val_dates, "COGS"])
        agg_rev_lgb.append(rev_pred_lgb)
        agg_rev_ridge.append(rev_pred_ridge)
        agg_margin_lgb.append(margin_pred_lgb)
        agg_margin_ridge.append(margin_pred_ridge)

    if not agg_actual_rev:
        return {
            "use_blend": 0.0,
            "w_rev_lgb": 1.0,
            "w_margin_lgb": 1.0,
            "n_folds": 0.0,
            "horizon_days_total": 0.0,
        }

    actual_rev = pd.concat(agg_actual_rev).sort_index()
    actual_cogs = pd.concat(agg_actual_cogs).sort_index()
    rev_lgb = pd.concat(agg_rev_lgb).sort_index()
    rev_ridge = pd.concat(agg_rev_ridge).sort_index()
    margin_lgb = pd.concat(agg_margin_lgb).sort_index()
    margin_ridge = pd.concat(agg_margin_ridge).sort_index()

    cogs_lgb = np.minimum((rev_lgb * margin_lgb).clip(lower=50.0), rev_lgb * 0.99)
    cogs_lgb = pd.Series(cogs_lgb, index=actual_cogs.index)
    base_rev = compute_metrics(actual_rev, rev_lgb)
    base_cogs = compute_metrics(actual_cogs, cogs_lgb)

    best = {
        "w_rev_lgb": 1.0,
        "w_margin_lgb": 1.0,
        "score": 2.0,
        "rev_metrics": base_rev,
        "cogs_metrics": base_cogs,
    }
    grid = np.linspace(0.0, 1.0, 21)
    for w_rev in grid:
        rev_blend = (w_rev * rev_lgb + (1.0 - w_rev) * rev_ridge).clip(lower=100.0)
        rev_metrics = compute_metrics(actual_rev, rev_blend)
        for w_margin in grid:
            margin_blend = (w_margin * margin_lgb + (1.0 - w_margin) * margin_ridge).clip(
                lower=0.40,
                upper=0.98,
            )
            cogs_blend = np.minimum(
                (rev_blend * margin_blend).clip(lower=50.0),
                rev_blend * 0.99,
            )
            cogs_blend = pd.Series(cogs_blend, index=actual_cogs.index)
            cogs_metrics = compute_metrics(actual_cogs, cogs_blend)
            score = (
                rev_metrics["mae"] / max(base_rev["mae"], 1e-9)
                + cogs_metrics["mae"] / max(base_cogs["mae"], 1e-9)
            )
            if score < best["score"]:
                best = {
                    "w_rev_lgb": float(w_rev),
                    "w_margin_lgb": float(w_margin),
                    "score": float(score),
                    "rev_metrics": rev_metrics,
                    "cogs_metrics": cogs_metrics,
                }

    use_blend = (
        best["score"] < 1.995
        and best["rev_metrics"]["mae"] <= base_rev["mae"] * 1.005
        and best["cogs_metrics"]["mae"] <= base_cogs["mae"] * 1.005
        and (
            best["rev_metrics"]["mae"] < base_rev["mae"]
            or best["cogs_metrics"]["mae"] < base_cogs["mae"]
        )
    )

    selected = best if use_blend else {
        "w_rev_lgb": 1.0,
        "w_margin_lgb": 1.0,
        "score": 2.0,
        "rev_metrics": base_rev,
        "cogs_metrics": base_cogs,
    }
    return {
        "use_blend": float(1 if use_blend else 0),
        "w_rev_lgb": selected["w_rev_lgb"],
        "w_margin_lgb": selected["w_margin_lgb"],
        "n_folds": float(valid_folds),
        "horizon_days_total": float(len(actual_rev)),
        "baseline_rev_mae": base_rev["mae"],
        "baseline_cogs_mae": base_cogs["mae"],
        "selected_rev_mae": selected["rev_metrics"]["mae"],
        "selected_cogs_mae": selected["cogs_metrics"]["mae"],
        "selected_rev_rmse": selected["rev_metrics"]["rmse"],
        "selected_cogs_rmse": selected["cogs_metrics"]["rmse"],
        "selected_rev_r2": selected["rev_metrics"]["r2"],
        "selected_cogs_r2": selected["cogs_metrics"]["r2"],
        "selection_score": selected["score"],
    }


def feature_group(name: str) -> str:
    if "_L730" in name or "_L1095" in name:
        return "Business Lag Drivers"
    if name.startswith("lag_") or name.startswith("roll") or name == "yoy_365_730":
        return "Autoregressive"
    if name.startswith("is_tet") or name == "is_holiday":
        return "Holidays"
    return "Calendar & Seasonality"


def build_feature_sets() -> List[FeatureSet]:
    return [
        FeatureSet(
            name="core_long",
            target_lags=(364, 365, 366, 730, 1095),
            roll_windows=(7, 30),
            biz_lags=(),
        ),
        FeatureSet(
            name="core_long_plus_short",
            target_lags=(7, 14, 28, 56, 364, 365, 366, 730, 1095),
            roll_windows=(7, 30, 90),
            biz_lags=(),
        ),
        FeatureSet(
            name="core_long_plus_biz730",
            target_lags=(364, 365, 366, 730, 1095),
            roll_windows=(7, 30),
            biz_lags=(730,),
        ),
        FeatureSet(
            name="core_long_plus_biz730_1095",
            target_lags=(364, 365, 366, 730, 1095),
            roll_windows=(7, 30),
            biz_lags=(730, 1095),
        ),
        FeatureSet(
            name="all_valid",
            target_lags=(7, 14, 28, 56, 364, 365, 366, 730, 1095),
            roll_windows=(7, 30, 90),
            biz_lags=(730,),
        ),
    ]


def main() -> None:
    sales, test, biz = load_data()
    sales = sales.sort_values("Date")
    test = test.sort_values("Date")
    biz = biz.sort_values("Date")

    print("=== Data Overview ===")
    print(f"Train rows: {len(sales)}  ({sales['Date'].min().date()} -> {sales['Date'].max().date()})")
    print(f"Test rows:  {len(test)}  ({test['Date'].min().date()} -> {test['Date'].max().date()})")

    all_index = pd.DatetimeIndex(
        sorted(set(pd.DatetimeIndex(sales["Date"])) | set(pd.DatetimeIndex(test["Date"])))
    )
    calendar = build_calendar_features(all_index)

    biz_cols = [
        c
        for c in biz.columns
        if c
        not in {
            "Date",
            "Revenue",
            "COGS",
            "east_rev",
            "total_rev_reg",
        }
    ]
    biz_lookup = {
        col: {
            pd.Timestamp(d): float(v)
            for d, v in zip(pd.DatetimeIndex(biz["Date"]), biz[col].values)
        }
        for col in biz_cols
    }

    feature_sets = build_feature_sets()
    fold_results = []
    summary_results = []

    print("\n=== Feature Experiments (Long-Horizon, Recursive) ===")
    for cfg in feature_sets:
        print(f"Running: {cfg.name}")
        fold_df, summary_df = run_backtest_for_set(sales, calendar, biz_lookup, biz_cols, cfg)
        if not fold_df.empty:
            fold_results.append(fold_df)
            summary_results.append(summary_df)
            row = summary_df.iloc[0]
            print(
                f"  rev_MAE={row['rev_mae']:,.0f}  rev_RMSE={row['rev_rmse']:,.0f}  rev_R2={row['rev_r2']:.4f} | "
                f"cogs_MAE={row['cogs_mae']:,.0f}  cogs_RMSE={row['cogs_rmse']:,.0f}  cogs_R2={row['cogs_r2']:.4f}"
            )
        else:
            print("  skipped (insufficient folds)")

    fold_all = pd.concat(fold_results, ignore_index=True)
    summary_all = pd.concat(summary_results, ignore_index=True)

    summary_all = summary_all.sort_values(
        by=["rev_mae", "rev_rmse", "cogs_mae", "cogs_rmse"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    best_name = summary_all.iloc[0]["feature_set"]
    summary_all["selected"] = summary_all["feature_set"].eq(best_name)

    print("\n=== Experiment Summary (sorted) ===")
    print(summary_all.to_string(index=False))
    print(f"\nSelected feature set: {best_name}")

    fold_all.to_csv(DATA_DIR / "feature_experiments_folds.csv", index=False)
    summary_all.to_csv(DATA_DIR / "feature_experiments_summary.csv", index=False)

    selected_cfg = next(fs for fs in feature_sets if fs.name == best_name)
    blend_cfg = calibrate_blend_for_feature_set(sales, calendar, biz_lookup, biz_cols, selected_cfg)
    use_blend = bool(int(blend_cfg["use_blend"]))
    w_rev_lgb = float(blend_cfg["w_rev_lgb"])
    w_margin_lgb = float(blend_cfg["w_margin_lgb"])
    print("\n=== Blend Calibration (selected feature set) ===")
    print(
        f"use_blend={use_blend}  w_rev_lgb={w_rev_lgb:.2f}  w_margin_lgb={w_margin_lgb:.2f} | "
        f"rev_MAE: {blend_cfg['baseline_rev_mae']:,.0f} -> {blend_cfg['selected_rev_mae']:,.0f} | "
        f"cogs_MAE: {blend_cfg['baseline_cogs_mae']:,.0f} -> {blend_cfg['selected_cogs_mae']:,.0f}"
    )

    sales = sales.set_index("Date")
    sales_idx = pd.DatetimeIndex(sales.index)
    margin = (sales["COGS"] / sales["Revenue"]).clip(0.40, 0.98)
    biz_index_set = set(sales_idx)

    # Final training on full history
    rev_trend_lookup_final = fit_trend_lookup(
        sales["Revenue"], pd.DatetimeIndex(sales_idx.union(pd.DatetimeIndex(test["Date"]))), clip_min=100.0
    )
    X_rev, y_rev = make_train_matrix(
        sales_idx,
        sales["Revenue"],
        calendar,
        biz_lookup,
        biz_cols,
        selected_cfg,
        biz_index_set,
        trend_lookup=rev_trend_lookup_final,
    )
    rev_lgb_model, rev_lgb_med = fit_lgbm(X_rev, y_rev, seed=SEED + 999, log_target=True)
    rev_ridge_model, rev_ridge_med = fit_ridge(X_rev, y_rev, log_target=True)
    rev_pred_lgb = recursive_forecast(
        rev_lgb_model,
        rev_lgb_med,
        pd.DatetimeIndex(test["Date"]),
        {pd.Timestamp(d): float(v) for d, v in sales["Revenue"].items()},
        calendar,
        biz_lookup,
        biz_cols,
        selected_cfg,
        trend_lookup=rev_trend_lookup_final,
        log_target=True,
    ).clip(lower=100.0)
    rev_pred_ridge = recursive_forecast(
        rev_ridge_model,
        rev_ridge_med,
        pd.DatetimeIndex(test["Date"]),
        {pd.Timestamp(d): float(v) for d, v in sales["Revenue"].items()},
        calendar,
        biz_lookup,
        biz_cols,
        selected_cfg,
        trend_lookup=rev_trend_lookup_final,
        log_target=True,
    ).clip(lower=100.0)
    rev_pred = (w_rev_lgb * rev_pred_lgb + (1.0 - w_rev_lgb) * rev_pred_ridge).clip(lower=100.0)

    margin_trend_lookup_final = fit_trend_lookup(
        margin, pd.DatetimeIndex(sales_idx.union(pd.DatetimeIndex(test["Date"]))), clip_min=0.01
    )
    X_margin, y_margin = make_train_matrix(
        sales_idx,
        margin,
        calendar,
        biz_lookup,
        biz_cols,
        selected_cfg,
        biz_index_set,
        trend_lookup=margin_trend_lookup_final,
    )
    margin_lgb_model, margin_lgb_med = fit_lgbm(X_margin, y_margin, seed=SEED + 1999)
    margin_ridge_model, margin_ridge_med = fit_ridge(X_margin, y_margin)
    margin_pred_lgb = recursive_forecast(
        margin_lgb_model,
        margin_lgb_med,
        pd.DatetimeIndex(test["Date"]),
        {pd.Timestamp(d): float(v) for d, v in margin.items()},
        calendar,
        biz_lookup,
        biz_cols,
        selected_cfg,
        trend_lookup=margin_trend_lookup_final,
    )
    margin_pred_ridge = recursive_forecast(
        margin_ridge_model,
        margin_ridge_med,
        pd.DatetimeIndex(test["Date"]),
        {pd.Timestamp(d): float(v) for d, v in margin.items()},
        calendar,
        biz_lookup,
        biz_cols,
        selected_cfg,
        trend_lookup=margin_trend_lookup_final,
    )
    margin_pred = (
        w_margin_lgb * margin_pred_lgb + (1.0 - w_margin_lgb) * margin_pred_ridge
    ).clip(lower=0.40, upper=0.98)

    cogs_pred = (rev_pred * margin_pred).clip(lower=50.0)
    cogs_pred = np.minimum(cogs_pred, rev_pred * 0.99)
    cogs_pred = pd.Series(cogs_pred, index=test["Date"])

    sub = pd.DataFrame(
        {
            "Date": pd.DatetimeIndex(test["Date"]).strftime("%Y-%m-%d"),
            "Revenue": np.round(rev_pred.values, 2),
            "COGS": np.round(cogs_pred.values, 2),
        }
    )
    assert len(sub) == len(test)
    assert (sub["Revenue"] > 0).all()
    assert (sub["COGS"] < sub["Revenue"]).all()
    sub.to_csv(DATA_DIR / "submission.csv", index=False)
    print("\nSaved: dataset/submission.csv")

    # Feature importances
    rev_imp = pd.DataFrame(
        {"feature": X_rev.columns, "importance_gain": rev_lgb_model.feature_importances_}
    ).sort_values("importance_gain", ascending=False)
    margin_imp = pd.DataFrame(
        {"feature": X_margin.columns, "importance_gain": margin_lgb_model.feature_importances_}
    ).sort_values("importance_gain", ascending=False)
    rev_imp.to_csv(DATA_DIR / "feature_importance_revenue.csv", index=False)
    margin_imp.to_csv(DATA_DIR / "feature_importance_margin.csv", index=False)

    grp = (
        rev_imp.assign(group=rev_imp["feature"].map(feature_group))
        .groupby("group", as_index=False)["importance_gain"]
        .sum()
        .sort_values("importance_gain", ascending=False)
    )

    importance_report = {
        "method": "lightgbm_gain_importance",
        "selected_feature_set": best_name,
        "blend": {
            "enabled": use_blend,
            "w_rev_lgb": w_rev_lgb,
            "w_margin_lgb": w_margin_lgb,
            "calibration": blend_cfg,
        },
        "feature_groups": {r["feature"]: feature_group(r["feature"]) for _, r in rev_imp.iterrows()},
        "group_importance": {
            str(r["group"]): float(r["importance_gain"]) for _, r in grp.iterrows()
        },
        "top_features": {
            str(r["feature"]): float(r["importance_gain"])
            for _, r in rev_imp.head(25).iterrows()
        },
    }
    with (DATA_DIR / "shap_report.json").open("w", encoding="utf-8") as f:
        json.dump(importance_report, f, indent=2)

    print("Saved: dataset/feature_experiments_folds.csv")
    print("Saved: dataset/feature_experiments_summary.csv")
    print("Saved: dataset/feature_importance_revenue.csv")
    print("Saved: dataset/feature_importance_margin.csv")
    print("Saved: dataset/shap_report.json")


if __name__ == "__main__":
    main()

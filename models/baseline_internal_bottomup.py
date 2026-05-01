from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import (
    baseline_submission_path,
    build_daily_frame,
    load_dataframes,
    sample_dates,
)


PROFILE_COLUMNS = [
    "rev_norm_mean",
    "rev_norm_std",
    "cogs_norm_mean",
    "cogs_norm_std",
]


def add_core_calendar_features(frame: pd.DataFrame, date_column: str = "Date") -> pd.DataFrame:
    data = frame.copy()
    data[date_column] = pd.to_datetime(data[date_column])
    data["year"] = data[date_column].dt.year
    data["month"] = data[date_column].dt.month
    data["day"] = data[date_column].dt.day
    data["dayofweek"] = data[date_column].dt.dayofweek
    data["dayofyear"] = data[date_column].dt.dayofyear
    data["weekofyear"] = data[date_column].dt.isocalendar().week.astype(int)
    data["quarter"] = data[date_column].dt.quarter
    data["is_weekend"] = (data["dayofweek"] >= 5).astype(int)
    data["days_in_month"] = data[date_column].dt.days_in_month
    data["days_from_mend"] = data["days_in_month"] - data["day"]
    data["days_from_mstart"] = data["day"] - 1
    data["is_last1"] = (data["days_from_mend"] == 0).astype(int)
    data["is_last3"] = (data["days_from_mend"] <= 2).astype(int)
    data["is_last7"] = (data["days_from_mend"] <= 6).astype(int)
    data["is_first3"] = (data["day"] <= 3).astype(int)
    data["is_first7"] = (data["day"] <= 7).astype(int)
    data["month_sin"] = np.sin(2.0 * np.pi * data["month"] / 12.0)
    data["month_cos"] = np.cos(2.0 * np.pi * data["month"] / 12.0)
    data["dow_sin"] = np.sin(2.0 * np.pi * data["dayofweek"] / 7.0)
    data["dow_cos"] = np.cos(2.0 * np.pi * data["dayofweek"] / 7.0)
    data["doy_sin"] = np.sin(2.0 * np.pi * data["dayofyear"] / 366.0)
    data["doy_cos"] = np.cos(2.0 * np.pi * data["dayofyear"] / 366.0)
    return data


def build_cross_year_profile(history: pd.DataFrame, date_column: str = "Date") -> pd.DataFrame:
    data = add_core_calendar_features(history, date_column=date_column)
    annual_means = (
        data.groupby("year")
        .agg(
            revenue_year_mean=("Revenue", "mean"),
            cogs_year_mean=("COGS", "mean"),
        )
        .reset_index()
    )
    data = data.merge(annual_means, on="year", how="left")
    data["rev_norm"] = data["Revenue"] / data["revenue_year_mean"].replace(0, np.nan)
    data["cogs_norm"] = data["COGS"] / data["cogs_year_mean"].replace(0, np.nan)

    profile = (
        data.groupby(["month", "day"])
        .agg(
            rev_norm_mean=("rev_norm", "mean"),
            rev_norm_std=("rev_norm", "std"),
            cogs_norm_mean=("cogs_norm", "mean"),
            cogs_norm_std=("cogs_norm", "std"),
        )
        .reset_index()
    )
    profile["rev_norm_std"] = profile["rev_norm_std"].fillna(0.05)
    profile["cogs_norm_std"] = profile["cogs_norm_std"].fillna(0.03)
    profile["rev_norm_mean"] = profile["rev_norm_mean"].clip(0.20, 3.50)
    profile["cogs_norm_mean"] = profile["cogs_norm_mean"].clip(0.65, 1.20)
    return profile


def attach_profile(frame: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    data = add_core_calendar_features(frame)
    merged = data.drop(columns=[c for c in PROFILE_COLUMNS if c in data.columns]).merge(
        profile[["month", "day"] + PROFILE_COLUMNS],
        on=["month", "day"],
        how="left",
    )
    for column in PROFILE_COLUMNS:
        default = 1.0 if column.endswith("mean") else 0.05
        merged[column] = merged[column].fillna(default)
    return merged


def _geometric_growth(
    annual_series: pd.Series,
    base_year: int,
    recent_years: int = 4,
    clip_low: float = 0.88,
    clip_high: float = 1.22,
) -> float:
    data = annual_series.dropna().sort_index()
    if len(data) <= 1:
        return 1.0

    recent = data.loc[data.index <= base_year].tail(recent_years)
    if len(recent) <= 1:
        return 1.0

    ratios = recent.iloc[1:].to_numpy(dtype=float) / np.maximum(
        recent.iloc[:-1].to_numpy(dtype=float),
        1e-9,
    )
    growth = float(np.exp(np.mean(np.log(np.clip(ratios, 1e-9, None)))))
    return float(np.clip(growth, clip_low, clip_high))


def build_repo_style_baseline_frame(
    history_daily: pd.DataFrame,
    future_dates: pd.DatetimeIndex,
    base_year: int | None = None,
    recent_years: int = 4,
) -> pd.DataFrame:
    history = history_daily[["date", "Revenue", "COGS"]].copy()
    history = history.rename(columns={"date": "Date"})
    history = add_core_calendar_features(history)

    profile = build_cross_year_profile(history)
    annual = (
        history.groupby("year")
        .agg(
            revenue_mean=("Revenue", "mean"),
            cogs_mean=("COGS", "mean"),
        )
        .sort_index()
    )
    if base_year is None:
        base_year = int(annual.index.max())

    revenue_growth = _geometric_growth(annual["revenue_mean"], base_year, recent_years, 0.90, 1.18)
    cogs_growth = _geometric_growth(annual["cogs_mean"], base_year, recent_years, 0.90, 1.16)

    base_revenue = float(annual.loc[base_year, "revenue_mean"])
    base_cogs = float(annual.loc[base_year, "cogs_mean"])

    future = pd.DataFrame({"Date": pd.DatetimeIndex(future_dates)})
    future = attach_profile(future, profile)
    year_gap = future["year"].to_numpy(dtype=float) - float(base_year)

    revenue_scale = base_revenue * (revenue_growth**year_gap)
    cogs_scale = base_cogs * (cogs_growth**year_gap)

    future["Revenue"] = np.maximum(revenue_scale * future["rev_norm_mean"].to_numpy(dtype=float), 0.0)
    future["COGS"] = np.maximum(cogs_scale * future["cogs_norm_mean"].to_numpy(dtype=float), 0.0)

    ratio = (future["COGS"] / future["Revenue"].replace(0, np.nan)).clip(0.78, 0.98).fillna(0.88)
    future["COGS"] = future["Revenue"] * ratio
    return future[["Date", "Revenue", "COGS"]].copy()


def _write_submission(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.2f")
    return path


def build_repo_baseline_submission() -> dict[str, Path]:
    data = load_dataframes()
    daily = build_daily_frame(data)
    future_dates = sample_dates()
    baseline = build_repo_style_baseline_frame(daily, future_dates)

    baseline_path = _write_submission(baseline, baseline_submission_path())
    return {
        "baseline_submission": baseline_path,
    }


def build_internal_bottomup_baseline() -> dict[str, Path]:
    # Keep the public function name stable for the existing scripts.
    return build_repo_baseline_submission()


def main() -> None:
    outputs = build_repo_baseline_submission()
    print(str(outputs["baseline_submission"]))


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"
OUT_DIR = ROOT / "eda_results" / "team_story"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)


def save_table(df: pd.DataFrame, name: str) -> None:
    df.to_csv(TAB_DIR / f"{name}.csv", index=False)


def set_plot_defaults() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9


def load_data() -> dict[str, pd.DataFrame]:
    return {
        "sales": pd.read_csv(DATA_DIR / "sales.csv", parse_dates=["Date"]).sort_values("Date"),
        "orders": pd.read_csv(DATA_DIR / "orders.csv", parse_dates=["order_date"]).sort_values("order_date"),
        "order_items": pd.read_csv(DATA_DIR / "order_items.csv", low_memory=False),
        "products": pd.read_csv(DATA_DIR / "products.csv"),
        "returns": pd.read_csv(DATA_DIR / "returns.csv", parse_dates=["return_date"]),
        "customers": pd.read_csv(DATA_DIR / "customers.csv", parse_dates=["signup_date"]),
        "geography": pd.read_csv(DATA_DIR / "geography.csv"),
        "payments": pd.read_csv(DATA_DIR / "payments.csv"),
        "web_traffic": pd.read_csv(DATA_DIR / "web_traffic.csv", parse_dates=["date"]).sort_values("date"),
        "inventory": pd.read_csv(DATA_DIR / "inventory.csv", parse_dates=["snapshot_date"]),
        "promotions": pd.read_csv(DATA_DIR / "promotions.csv", parse_dates=["start_date", "end_date"]),
        "feature_exp": pd.read_csv(DATA_DIR / "feature_experiments_summary.csv"),
        "feat_imp_rev": pd.read_csv(DATA_DIR / "feature_importance_revenue.csv"),
        "candidate_vs_2022": pd.read_csv(
            ROOT / "eda_results" / "cross_repo_audit" / "candidate_vs_2022_reference.csv"
        ),
        "candidate_fix_summary": pd.read_csv(
            ROOT / "eda_results" / "cross_repo_audit" / "candidate_fix_and_blend_summary.csv"
        ),
    }


def build_line_fact(d: dict[str, pd.DataFrame]) -> pd.DataFrame:
    oi = d["order_items"].copy()
    p = d["products"][["product_id", "category", "segment", "size", "cogs"]].copy()
    o = d["orders"][["order_id", "order_date", "payment_method", "order_status", "order_source", "zip"]].copy()
    g = d["geography"][["zip", "region"]].copy()
    promo = d["promotions"][
        ["promo_id", "promo_name", "promo_type", "discount_value", "promo_channel", "stackable_flag"]
    ].copy()

    f = (
        oi.merge(p, on="product_id", how="left")
        .merge(o, on="order_id", how="left")
        .merge(g, on="zip", how="left")
        .merge(promo, on="promo_id", how="left")
    )
    f["order_year"] = f["order_date"].dt.year
    f["order_month"] = f["order_date"].dt.month
    f["order_dow"] = f["order_date"].dt.dayofweek
    f["gross_revenue"] = f["quantity"] * f["unit_price"]
    f["discount_amount"] = f["discount_amount"].fillna(0.0)
    f["net_revenue"] = f["gross_revenue"] - f["discount_amount"]
    f["line_cogs"] = f["quantity"] * f["cogs"]
    f["gross_profit"] = f["net_revenue"] - f["line_cogs"]
    f["discount_rate"] = np.where(f["gross_revenue"] > 0, f["discount_amount"] / f["gross_revenue"], 0.0)
    f["has_promo"] = f["promo_id"].notna().astype(int)
    return f


def fig01_yearly_financials(sales: pd.DataFrame) -> pd.DataFrame:
    s = sales.copy()
    s["year"] = s["Date"].dt.year
    y = (
        s.groupby("year", as_index=False)
        .agg(Revenue=("Revenue", "sum"), COGS=("COGS", "sum"))
        .sort_values("year")
    )
    y["GrossProfit"] = y["Revenue"] - y["COGS"]
    y["MarginRate"] = np.where(y["Revenue"] > 0, y["GrossProfit"] / y["Revenue"], np.nan)
    y["RevenueYoY"] = y["Revenue"].pct_change()
    y["GrossProfitYoY"] = y["GrossProfit"].pct_change()
    save_table(y, "fig01_yearly_financials")

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(y["year"], y["Revenue"] / 1e9, marker="o", label="Revenue (B)")
    ax1.plot(y["year"], y["COGS"] / 1e9, marker="o", label="COGS (B)")
    ax1.plot(y["year"], y["GrossProfit"] / 1e9, marker="o", label="Gross Profit (B)")
    ax1.set_ylabel("VND (Billion)")
    ax1.set_xlabel("Year")
    ax1.set_title("Yearly Revenue / COGS / Gross Profit")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(y["year"], y["MarginRate"] * 100, color="black", linestyle="--", marker="s", label="Margin Rate")
    ax2.set_ylabel("Margin Rate (%)")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig01_yearly_financials.png")
    plt.close(fig)
    return y


def fig02_monthly_heatmap(sales: pd.DataFrame) -> pd.DataFrame:
    s = sales.copy()
    s["year"] = s["Date"].dt.year
    s["month"] = s["Date"].dt.month
    mm = s.groupby(["year", "month"], as_index=False)["Revenue"].sum()
    pivot = mm.pivot(index="year", columns="month", values="Revenue").sort_index()
    save_table(mm, "fig02_monthly_revenue")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    im = ax.imshow(pivot.values / 1e9, aspect="auto", cmap="YlGnBu")
    ax.set_title("Monthly Revenue Heatmap (Billion VND)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(range(1, 13))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index.astype(str))
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Revenue (Billion VND)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig02_monthly_heatmap.png")
    plt.close(fig)
    return mm


def fig03_weekday_pattern(sales: pd.DataFrame) -> pd.DataFrame:
    s = sales.copy()
    s["dow"] = s["Date"].dt.dayofweek
    s["gross_profit"] = s["Revenue"] - s["COGS"]
    s["margin_rate"] = np.where(s["Revenue"] > 0, s["gross_profit"] / s["Revenue"], np.nan)
    w = s.groupby("dow", as_index=False).agg(
        avg_revenue=("Revenue", "mean"),
        avg_cogs=("COGS", "mean"),
        avg_margin_rate=("margin_rate", "mean"),
    )
    w["dow_name"] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    save_table(w, "fig03_weekday_pattern")

    fig, ax1 = plt.subplots(figsize=(8.5, 4.5))
    ax1.bar(w["dow_name"], w["avg_revenue"] / 1e6, color="#3A7CA5", alpha=0.85, label="Avg Revenue (M)")
    ax1.set_ylabel("Avg Revenue (Million VND)")
    ax1.set_xlabel("Day of Week")
    ax1.set_title("Daily Demand Pattern by Day of Week")

    ax2 = ax1.twinx()
    ax2.plot(w["dow_name"], w["avg_margin_rate"] * 100, color="#222222", marker="o", label="Avg Margin Rate")
    ax2.set_ylabel("Avg Margin Rate (%)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig03_weekday_pattern.png")
    plt.close(fig)
    return w


def fig04_category_scatter(line_fact: pd.DataFrame) -> pd.DataFrame:
    c = (
        line_fact.groupby("category", as_index=False)
        .agg(
            net_revenue=("net_revenue", "sum"),
            gross_profit=("gross_profit", "sum"),
            order_lines=("order_id", "count"),
            avg_discount_rate=("discount_rate", "mean"),
        )
        .sort_values("net_revenue", ascending=False)
    )
    c["margin_rate"] = np.where(c["net_revenue"] > 0, c["gross_profit"] / c["net_revenue"], np.nan)
    save_table(c, "fig04_category_profitability")

    fig, ax = plt.subplots(figsize=(8.5, 5))
    sizes = np.maximum(40, c["order_lines"] / 1500)
    sc = ax.scatter(
        c["net_revenue"] / 1e9,
        c["margin_rate"] * 100,
        s=sizes,
        c=c["avg_discount_rate"] * 100,
        cmap="viridis",
        alpha=0.9,
    )
    for _, r in c.iterrows():
        ax.text(r["net_revenue"] / 1e9, r["margin_rate"] * 100, f" {r['category']}", fontsize=8, va="center")
    ax.set_xlabel("Net Revenue (Billion VND)")
    ax.set_ylabel("Margin Rate (%)")
    ax.set_title("Category Revenue vs Margin (color = avg discount rate)")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Avg Discount Rate (%)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig04_category_scatter.png")
    plt.close(fig)
    return c


def fig05_segment_profitability(line_fact: pd.DataFrame) -> pd.DataFrame:
    s = (
        line_fact.groupby("segment", as_index=False)
        .agg(net_revenue=("net_revenue", "sum"), gross_profit=("gross_profit", "sum"), lines=("order_id", "count"))
        .sort_values("net_revenue", ascending=False)
    )
    s["margin_rate"] = np.where(s["net_revenue"] > 0, s["gross_profit"] / s["net_revenue"], np.nan)
    save_table(s, "fig05_segment_profitability")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(s["segment"], s["margin_rate"] * 100, color="#2A9D8F")
    ax.set_ylabel("Margin Rate (%)")
    ax.set_xlabel("Segment")
    ax.set_title("Gross Margin Rate by Segment")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig05_segment_profitability.png")
    plt.close(fig)
    return s


def fig06_discount_buckets(line_fact: pd.DataFrame) -> pd.DataFrame:
    f = line_fact.copy()
    bins = [-0.001, 0.0, 0.05, 0.10, 0.20, 1.0]
    labels = ["0%", "0-5%", "5-10%", "10-20%", "20%+"]
    f["discount_bucket"] = pd.cut(f["discount_rate"], bins=bins, labels=labels)
    b = (
        f.groupby("discount_bucket", as_index=False)
        .agg(
            lines=("order_id", "count"),
            net_revenue=("net_revenue", "sum"),
            gross_profit=("gross_profit", "sum"),
            avg_discount_rate=("discount_rate", "mean"),
        )
        .sort_values("discount_bucket")
    )
    b["margin_rate"] = np.where(b["net_revenue"] > 0, b["gross_profit"] / b["net_revenue"], np.nan)
    save_table(b, "fig06_discount_buckets")

    fig, ax1 = plt.subplots(figsize=(8.5, 4.5))
    ax1.bar(b["discount_bucket"].astype(str), b["net_revenue"] / 1e9, color="#457B9D", label="Net Revenue")
    ax1.set_ylabel("Net Revenue (Billion VND)")
    ax1.set_xlabel("Discount Bucket")
    ax1.set_title("Discount Intensity vs Revenue and Margin")
    ax2 = ax1.twinx()
    ax2.plot(b["discount_bucket"].astype(str), b["margin_rate"] * 100, color="#E63946", marker="o")
    ax2.set_ylabel("Margin Rate (%)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig06_discount_buckets.png")
    plt.close(fig)
    return b


def fig07_promo_type(line_fact: pd.DataFrame) -> pd.DataFrame:
    p = line_fact[line_fact["has_promo"] == 1].copy()
    g = (
        p.groupby("promo_type", as_index=False)
        .agg(lines=("order_id", "count"), net_revenue=("net_revenue", "sum"), gross_profit=("gross_profit", "sum"))
        .sort_values("lines", ascending=False)
    )
    g["margin_rate"] = np.where(g["net_revenue"] > 0, g["gross_profit"] / g["net_revenue"], np.nan)
    save_table(g, "fig07_promo_type")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(g["promo_type"].fillna("Unknown"), g["gross_profit"] / 1e9, color="#8D99AE")
    ax.set_ylabel("Gross Profit (Billion VND)")
    ax.set_xlabel("Promo Type")
    ax.set_title("Gross Profit Contribution by Promo Type")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig07_promo_type.png")
    plt.close(fig)
    return g


def fig08_returns(returns: pd.DataFrame, products: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    r = returns.merge(products[["product_id", "category"]], on="product_id", how="left")
    top_reason = r.groupby("return_reason", as_index=False).agg(count=("return_id", "count")).sort_values(
        "count", ascending=False
    )
    cat_reason = (
        r.groupby(["category", "return_reason"], as_index=False)
        .agg(count=("return_id", "count"))
        .sort_values(["category", "count"], ascending=[True, False])
    )
    save_table(top_reason, "fig08_returns_top_reason")
    save_table(cat_reason, "fig08_returns_by_category_reason")

    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    ax.bar(top_reason["return_reason"], top_reason["count"], color="#A8DADC")
    ax.set_title("Return Reasons (All Categories)")
    ax.set_xlabel("Return Reason")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig08_returns_reasons.png")
    plt.close(fig)
    return top_reason, cat_reason


def fig09_cancellation(orders: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    o = orders.copy()
    o["is_cancelled"] = o["order_status"].eq("cancelled").astype(int)
    by_pay = (
        o.groupby("payment_method", as_index=False)
        .agg(orders=("order_id", "count"), cancelled=("is_cancelled", "sum"))
        .sort_values("orders", ascending=False)
    )
    by_pay["cancel_rate"] = np.where(by_pay["orders"] > 0, by_pay["cancelled"] / by_pay["orders"], np.nan)

    by_src = (
        o.groupby("order_source", as_index=False)
        .agg(orders=("order_id", "count"), cancelled=("is_cancelled", "sum"))
        .sort_values("orders", ascending=False)
    )
    by_src["cancel_rate"] = np.where(by_src["orders"] > 0, by_src["cancelled"] / by_src["orders"], np.nan)
    save_table(by_pay, "fig09_cancel_by_payment")
    save_table(by_src, "fig09_cancel_by_source")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].bar(by_pay["payment_method"], by_pay["cancel_rate"] * 100, color="#F4A261")
    axes[0].set_title("Cancellation Rate by Payment Method")
    axes[0].set_ylabel("Cancellation Rate (%)")
    axes[0].tick_params(axis="x", rotation=30)
    axes[1].bar(by_src["order_source"], by_src["cancel_rate"] * 100, color="#E9C46A")
    axes[1].set_title("Cancellation Rate by Order Source")
    axes[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig09_cancellation_rates.png")
    plt.close(fig)
    return by_pay, by_src


def fig10_customer_cohorts(customers: pd.DataFrame) -> pd.DataFrame:
    c = customers.copy()
    c["cohort_month"] = c["signup_date"].dt.to_period("M").dt.to_timestamp()
    m = (
        c.groupby("cohort_month", as_index=False)
        .agg(new_customers=("customer_id", "nunique"))
        .sort_values("cohort_month")
    )
    m["cumulative_customers"] = m["new_customers"].cumsum()
    save_table(m, "fig10_customer_cohorts")

    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.bar(m["cohort_month"], m["new_customers"], width=25, color="#6D597A", alpha=0.85)
    ax1.set_ylabel("New Customers")
    ax1.set_xlabel("Signup Month")
    ax1.set_title("Customer Acquisition Cohorts")
    ax2 = ax1.twinx()
    ax2.plot(m["cohort_month"], m["cumulative_customers"], color="#2A9D8F", linewidth=1.7)
    ax2.set_ylabel("Cumulative Customers")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig10_customer_cohorts.png")
    plt.close(fig)
    return m


def fig11_age_group_orders(orders: pd.DataFrame, customers: pd.DataFrame, line_fact: pd.DataFrame) -> pd.DataFrame:
    oa = orders.merge(customers[["customer_id", "age_group"]], on="customer_id", how="left")
    order_value = line_fact.groupby("order_id", as_index=False)["net_revenue"].sum()
    oa = oa.merge(order_value, on="order_id", how="left")
    g = (
        oa.groupby("age_group", as_index=False)
        .agg(
            orders=("order_id", "count"),
            customers=("customer_id", "nunique"),
            avg_order_value=("net_revenue", "mean"),
        )
        .sort_values("orders", ascending=False)
    )
    g["orders_per_customer"] = np.where(g["customers"] > 0, g["orders"] / g["customers"], np.nan)
    save_table(g, "fig11_age_group_orders")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    axes[0].bar(g["age_group"].astype(str), g["orders_per_customer"], color="#457B9D")
    axes[0].set_title("Orders per Customer by Age Group")
    axes[0].set_ylabel("Orders / Customer")
    axes[1].bar(g["age_group"].astype(str), g["avg_order_value"] / 1e3, color="#1D3557")
    axes[1].set_title("Average Order Value by Age Group")
    axes[1].set_ylabel("AOV (Thousand VND)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig11_age_group_orders.png")
    plt.close(fig)
    return g


def fig12_region_trend(line_fact: pd.DataFrame) -> pd.DataFrame:
    f = line_fact.copy()
    f["year"] = f["order_date"].dt.year
    r = (
        f.groupby(["year", "region"], as_index=False)
        .agg(net_revenue=("net_revenue", "sum"))
        .sort_values(["year", "net_revenue"], ascending=[True, False])
    )
    save_table(r, "fig12_region_trend")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for region, grp in r.groupby("region"):
        ax.plot(grp["year"], grp["net_revenue"] / 1e9, marker="o", label=str(region))
    ax.set_title("Regional Revenue Trend by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Net Revenue (Billion VND)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig12_region_trend.png")
    plt.close(fig)
    return r


def fig13_inventory(inventory: pd.DataFrame) -> pd.DataFrame:
    inv = inventory.copy()
    inv["month"] = inv["snapshot_date"].dt.to_period("M").dt.to_timestamp()
    m = (
        inv.groupby("month", as_index=False)
        .agg(
            stockout_rate=("stockout_flag", "mean"),
            avg_stockout_days=("stockout_days", "mean"),
            avg_fill_rate=("fill_rate", "mean"),
            avg_days_supply=("days_of_supply", "mean"),
        )
        .sort_values("month")
    )
    save_table(m, "fig13_inventory_monthly")

    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.plot(m["month"], m["stockout_rate"] * 100, color="#D62828", label="Stockout Rate (%)")
    ax1.plot(m["month"], m["avg_fill_rate"] * 100, color="#2A9D8F", label="Fill Rate (%)")
    ax1.set_ylabel("Rate (%)")
    ax1.set_xlabel("Month")
    ax1.set_title("Inventory Stress: Stockout and Fill Rate Over Time")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(m["month"], m["avg_stockout_days"], color="#264653", linestyle="--", label="Avg Stockout Days")
    ax2.set_ylabel("Avg Stockout Days")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig13_inventory_stress.png")
    plt.close(fig)
    return m


def fig14_web_traffic(web: pd.DataFrame, orders: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    oday = orders.groupby("order_date", as_index=False).agg(orders=("order_id", "count"))
    s = sales.rename(columns={"Date": "date"})
    m = (
        web.merge(oday, left_on="date", right_on="order_date", how="left")
        .merge(s[["date", "Revenue"]], on="date", how="left")
        .drop(columns=["order_date"])
        .sort_values("date")
    )
    m["orders"] = m["orders"].fillna(0)
    save_table(m, "fig14_web_traffic_joined")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].scatter(m["sessions"], m["Revenue"], alpha=0.5, s=10, color="#3A86FF")
    corr_sr = np.corrcoef(m["sessions"], m["Revenue"])[0, 1]
    axes[0].set_title(f"Sessions vs Revenue (corr={corr_sr:.3f})")
    axes[0].set_xlabel("Sessions")
    axes[0].set_ylabel("Revenue")

    axes[1].scatter(m["sessions"], m["orders"], alpha=0.5, s=10, color="#8338EC")
    corr_so = np.corrcoef(m["sessions"], m["orders"])[0, 1]
    axes[1].set_title(f"Sessions vs Orders (corr={corr_so:.3f})")
    axes[1].set_xlabel("Sessions")
    axes[1].set_ylabel("Orders")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig14_web_traffic_corr.png")
    plt.close(fig)
    return m


def fig15_feature_set_performance(feature_exp: pd.DataFrame) -> pd.DataFrame:
    f = feature_exp.sort_values("rev_mae").copy()
    save_table(f, "fig15_feature_set_performance")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    axes[0].barh(f["feature_set"], f["rev_mae"] / 1e6, color="#118AB2")
    axes[0].set_title("Revenue MAE by Feature Set")
    axes[0].set_xlabel("MAE (Million VND)")
    axes[1].barh(f["feature_set"], f["cogs_mae"] / 1e6, color="#06D6A0")
    axes[1].set_title("COGS MAE by Feature Set")
    axes[1].set_xlabel("MAE (Million VND)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig15_feature_set_performance.png")
    plt.close(fig)
    return f


def fig16_feature_importance(feat_imp: pd.DataFrame) -> pd.DataFrame:
    f = feat_imp.head(20).copy()
    save_table(f, "fig16_top20_feature_importance_revenue")

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.barh(f["feature"][::-1], f["importance_gain"][::-1], color="#EF476F")
    ax.set_title("Top 20 Revenue Feature Importance (LightGBM gain)")
    ax.set_xlabel("Importance Gain")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig16_feature_importance_top20.png")
    plt.close(fig)
    return f


def fig17_candidate_comparison(candidate_vs_2022: pd.DataFrame) -> pd.DataFrame:
    c = candidate_vs_2022.copy().sort_values("rev_vs_2022_pct")
    save_table(c, "fig17_candidate_vs_2022")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    axes[0].barh(c["candidate"], c["rev_vs_2022_pct"], color="#3A0CA3")
    axes[0].set_title("Candidate Revenue Mean vs 2022 Baseline")
    axes[0].set_xlabel("Delta vs 2022 (%)")
    axes[1].barh(c["candidate"], c["margin_delta_vs_2022_pp"], color="#F72585")
    axes[1].set_title("Candidate Margin vs 2022 Baseline")
    axes[1].set_xlabel("Delta vs 2022 (percentage points)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig17_candidate_comparison.png")
    plt.close(fig)
    return c


def fig18_candidate_fix_impact(candidate_fix_summary: pd.DataFrame) -> pd.DataFrame:
    c = candidate_fix_summary.copy().sort_values("violations_cogs_ge_revenue", ascending=False)
    save_table(c, "fig18_candidate_fix_impact")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    axes[0].barh(c["name"], c["violations_cogs_ge_revenue"], color="#E76F51")
    axes[0].set_title("Constraint Violations in Raw External Submissions")
    axes[0].set_xlabel("Rows with COGS >= Revenue")
    axes[1].barh(c["name"], c["total_cogs_adjustment_abs"] / 1e6, color="#F4A261")
    axes[1].set_title("Fix Magnitude Needed for Compliance")
    axes[1].set_xlabel("Absolute COGS Adjustment (Million VND)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig18_candidate_fix_impact.png")
    plt.close(fig)
    return c


def build_story_md(metrics: dict[str, float], refs: dict[str, pd.DataFrame]) -> str:
    y = refs["yearly"]
    top_year = y.sort_values("Revenue", ascending=False).iloc[0]
    low_year = y.sort_values("Revenue", ascending=True).iloc[0]
    cagr = (y.iloc[-1]["Revenue"] / y.iloc[0]["Revenue"]) ** (1 / max(1, (len(y) - 1))) - 1

    weekday = refs["weekday"]
    best_dow = weekday.sort_values("avg_revenue", ascending=False).iloc[0]
    worst_dow = weekday.sort_values("avg_revenue", ascending=True).iloc[0]

    category = refs["category"]
    top_cat = category.iloc[0]
    worst_cat_margin = category.sort_values("margin_rate", ascending=True).iloc[0]

    discount = refs["discount"]
    d_high = discount.sort_values("avg_discount_rate", ascending=False).iloc[0]
    d_low = discount.sort_values("avg_discount_rate", ascending=True).iloc[0]

    cancel_pay = refs["cancel_pay"].sort_values("cancel_rate", ascending=False).iloc[0]
    cancel_src = refs["cancel_src"].sort_values("cancel_rate", ascending=False).iloc[0]

    cohorts = refs["cohorts"]
    peak_acq = cohorts.sort_values("new_customers", ascending=False).iloc[0]

    age = refs["age"]
    top_age = age.sort_values("orders_per_customer", ascending=False).iloc[0]

    region = refs["region"]
    region_latest = region[region["year"] == region["year"].max()].sort_values("net_revenue", ascending=False)
    top_region_latest = region_latest.iloc[0]

    inventory = refs["inventory"]
    inv_peak = inventory.sort_values("stockout_rate", ascending=False).iloc[0]
    inv_low_fill = inventory.sort_values("avg_fill_rate", ascending=True).iloc[0]

    feat_perf = refs["feat_perf"]
    best_fs = feat_perf.sort_values("rev_mae", ascending=True).iloc[0]
    worst_fs = feat_perf.sort_values("rev_mae", ascending=False).iloc[0]

    candidates = refs["candidate"]
    safe = candidates[candidates["candidate"].eq("ours_current_fixed.csv")].iloc[0]
    balanced = candidates[candidates["candidate"].eq("r1_residual_ensemble_fixed.csv")].iloc[0]
    aggressive = candidates.sort_values("rev_vs_2022_pct", ascending=False).iloc[0]

    return f"""# Team Dataset + Model Story (Detailed)

Generated on: {pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M UTC")}

## 1. Executive Summary

1. Revenue concentration and profitability moved unevenly over time: peak revenue year was {int(top_year["year"])} ({top_year["Revenue"]:,.0f}), lowest was {int(low_year["year"])} ({low_year["Revenue"]:,.0f}); full-period CAGR is {cagr*100:.2f}%.
2. Demand is strongly cyclical by month and weekday: highest weekday revenue is {best_dow["dow_name"]} ({best_dow["avg_revenue"]:,.0f}/day) vs lowest {worst_dow["dow_name"]} ({worst_dow["avg_revenue"]:,.0f}/day).
3. Promotion pressure and return pressure can erode margin quickly: highest-discount bucket ({d_high["discount_bucket"]}) has margin {d_high["margin_rate"]*100:.2f}% vs {d_low["discount_bucket"]} at {d_low["margin_rate"]*100:.2f}%.
4. Operational leakage is measurable: highest cancellation payment method is {cancel_pay["payment_method"]} ({cancel_pay["cancel_rate"]*100:.2f}% cancel rate), and highest-risk order source is {cancel_src["order_source"]} ({cancel_src["cancel_rate"]*100:.2f}%).
5. Model side: `core_long` remains best (Rev MAE {best_fs["rev_mae"]:,.0f}); external submissions often fail constraints and require correction before use.

## 2. Demand + Financial Story

- See figures:
  - `fig01_yearly_financials.png`
  - `fig02_monthly_heatmap.png`
  - `fig03_weekday_pattern.png`

- What the numbers say:
  - Total revenue over whole history: {y["Revenue"].sum():,.0f}
  - Total COGS over whole history: {y["COGS"].sum():,.0f}
  - Total gross profit over whole history: {y["GrossProfit"].sum():,.0f}
  - Best year gross margin: {y["MarginRate"].max()*100:.2f}% ; worst year gross margin: {y["MarginRate"].min()*100:.2f}%
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
  - Top revenue category: {top_cat["category"]} ({top_cat["net_revenue"]:,.0f} net revenue, {top_cat["margin_rate"]*100:.2f}% margin).
  - Weakest margin category: {worst_cat_margin["category"]} ({worst_cat_margin["margin_rate"]*100:.2f}% margin).
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
  - Peak stockout-rate month: {inv_peak["month"].strftime("%Y-%m")} at {inv_peak["stockout_rate"]*100:.2f}%.
  - Lowest fill-rate month: {inv_low_fill["month"].strftime("%Y-%m")} at {inv_low_fill["avg_fill_rate"]*100:.2f}%.

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
  - Peak acquisition month: {peak_acq["cohort_month"].strftime("%Y-%m")} with {int(peak_acq["new_customers"]):,} new customers.
  - Highest orders/customer segment: {top_age["age_group"]} at {top_age["orders_per_customer"]:.2f}.
  - Latest-year top region: {top_region_latest["region"]} ({top_region_latest["net_revenue"]:,.0f}).
  - Sessions to revenue correlation: {metrics["sessions_rev_corr"]:.3f}; sessions to orders correlation: {metrics["sessions_orders_corr"]:.3f}.

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
  - Best feature set: `{best_fs["feature_set"]}` with Rev MAE {best_fs["rev_mae"]:,.0f}, COGS MAE {best_fs["cogs_mae"]:,.0f}.
  - Worst tested set (`{worst_fs["feature_set"]}`) is +{(worst_fs["rev_mae"]/best_fs["rev_mae"]-1)*100:.2f}% worse on Revenue MAE.
  - External candidates often required substantial correction for `COGS < Revenue` before becoming valid.
  - Safe candidate profile (ours): revenue mean delta vs 2022 = {safe["rev_vs_2022_pct"]:.2f}%.
  - Balanced external candidate (`r1_residual_ensemble_fixed.csv`) delta vs 2022 = {balanced["rev_vs_2022_pct"]:.2f}%.
  - Most aggressive candidate in pack: `{aggressive["candidate"]}` at +{aggressive["rev_vs_2022_pct"]:.2f}% vs 2022 mean.

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
"""


def main() -> None:
    ensure_dirs()
    set_plot_defaults()
    d = load_data()
    line_fact = build_line_fact(d)
    save_table(line_fact.head(10000), "sample_line_fact_10k")

    refs: dict[str, pd.DataFrame] = {}
    refs["yearly"] = fig01_yearly_financials(d["sales"])
    fig02_monthly_heatmap(d["sales"])
    refs["weekday"] = fig03_weekday_pattern(d["sales"])
    refs["category"] = fig04_category_scatter(line_fact)
    fig05_segment_profitability(line_fact)
    refs["discount"] = fig06_discount_buckets(line_fact)
    fig07_promo_type(line_fact)
    fig08_returns(d["returns"], d["products"])
    refs["cancel_pay"], refs["cancel_src"] = fig09_cancellation(d["orders"])
    refs["cohorts"] = fig10_customer_cohorts(d["customers"])
    refs["age"] = fig11_age_group_orders(d["orders"], d["customers"], line_fact)
    refs["region"] = fig12_region_trend(line_fact)
    refs["inventory"] = fig13_inventory(d["inventory"])
    joined_web = fig14_web_traffic(d["web_traffic"], d["orders"], d["sales"])
    refs["feat_perf"] = fig15_feature_set_performance(d["feature_exp"])
    fig16_feature_importance(d["feat_imp_rev"])
    refs["candidate"] = fig17_candidate_comparison(d["candidate_vs_2022"])
    fig18_candidate_fix_impact(d["candidate_fix_summary"])

    metrics = {
        "sessions_rev_corr": float(np.corrcoef(joined_web["sessions"], joined_web["Revenue"])[0, 1]),
        "sessions_orders_corr": float(np.corrcoef(joined_web["sessions"], joined_web["orders"])[0, 1]),
    }
    story = build_story_md(metrics, refs)
    (OUT_DIR / "team_story_detailed.md").write_text(story, encoding="utf-8")

    manifest = {
        "generated_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "figure_count": len(list(FIG_DIR.glob("*.png"))),
        "table_count": len(list(TAB_DIR.glob("*.csv"))),
        "main_report": str((OUT_DIR / "team_story_detailed.md").relative_to(ROOT)),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

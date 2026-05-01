from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"
SUBMISSIONS_DIR = DATASET_DIR / "submissions"


def ensure_submission_dir() -> Path:
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    return SUBMISSIONS_DIR


def canonical_submission_path() -> Path:
    return DATASET_DIR / "submission.csv"


def final_submission_path() -> Path:
    return ensure_submission_dir() / "final_internal_bottomup_shape_medium.csv"


def baseline_submission_path() -> Path:
    return ensure_submission_dir() / "internal_bottomup_baseline.csv"


def copy_file(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == target.resolve():
        return target
    shutil.copy2(source, target)
    return target


def load_dataframes() -> dict[str, pd.DataFrame]:
    return {
        "sales": pd.read_csv(DATASET_DIR / "sales.csv", parse_dates=["Date"]),
        "orders": pd.read_csv(DATASET_DIR / "orders.csv", parse_dates=["order_date"]),
        "order_items": pd.read_csv(DATASET_DIR / "order_items.csv", low_memory=False),
        "products": pd.read_csv(DATASET_DIR / "products.csv"),
        "promotions": pd.read_csv(DATASET_DIR / "promotions.csv", parse_dates=["start_date", "end_date"]),
        "inventory": pd.read_csv(DATASET_DIR / "inventory.csv", parse_dates=["snapshot_date"]),
        "web_traffic": pd.read_csv(DATASET_DIR / "web_traffic.csv", parse_dates=["date"]),
        "returns": pd.read_csv(DATASET_DIR / "returns.csv", parse_dates=["return_date"]),
        "reviews": pd.read_csv(DATASET_DIR / "reviews.csv", parse_dates=["review_date"]),
        "customers": pd.read_csv(DATASET_DIR / "customers.csv", parse_dates=["signup_date"]),
        "shipments": pd.read_csv(DATASET_DIR / "shipments.csv", parse_dates=["ship_date", "delivery_date"]),
        "payments": pd.read_csv(DATASET_DIR / "payments.csv"),
        "sample_submission": pd.read_csv(DATASET_DIR / "sample_submission.csv", parse_dates=["Date"]),
    }


def build_daily_frame(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    sales = data["sales"].copy().rename(columns={"Date": "date"})
    orders = data["orders"].copy()
    order_items = data["order_items"].copy()
    products = data["products"].copy()
    traffic = data["web_traffic"].copy()
    shipments = data["shipments"].copy()
    returns = data["returns"].copy()
    reviews = data["reviews"].copy()

    orders_daily = (
        orders.groupby("order_date")
        .agg(
            orders=("order_id", "count"),
            delivered_orders=("order_status", lambda s: (s == "delivered").sum()),
            returned_orders=("order_status", lambda s: (s == "returned").sum()),
            cancelled_orders=("order_status", lambda s: (s == "cancelled").sum()),
        )
        .reset_index()
        .rename(columns={"order_date": "date"})
    )

    order_items["promo_flag"] = order_items[["promo_id", "promo_id_2"]].notna().any(axis=1).astype(int)
    order_items["gross_revenue"] = order_items["quantity"] * order_items["unit_price"]
    order_items["net_revenue"] = order_items["gross_revenue"] - order_items["discount_amount"].fillna(0.0)
    product_cost = products[["product_id", "cogs"]].rename(columns={"cogs": "unit_cogs"})
    order_items = order_items.merge(product_cost, on="product_id", how="left")
    order_items["line_cogs"] = order_items["quantity"] * order_items["unit_cogs"].fillna(0.0)

    promo_orders = (
        order_items.groupby("order_id")
        .agg(
            promo_flag=("promo_flag", "max"),
            discount_amount=("discount_amount", "sum"),
            item_revenue=("net_revenue", "sum"),
            item_cogs=("line_cogs", "sum"),
        )
        .reset_index()
    )
    promo_orders = promo_orders.merge(orders[["order_id", "order_date"]], on="order_id", how="left")
    promo_daily = (
        promo_orders.groupby("order_date")
        .agg(
            promo_order_share=("promo_flag", "mean"),
            discount_amount=("discount_amount", "sum"),
            item_revenue=("item_revenue", "sum"),
            item_cogs=("item_cogs", "sum"),
        )
        .reset_index()
        .rename(columns={"order_date": "date"})
    )

    traffic_daily = (
        traffic.groupby("date")
        .agg(
            sessions=("sessions", "sum"),
            page_views=("page_views", "sum"),
            unique_visitors=("unique_visitors", "sum"),
            bounce_rate=("bounce_rate", "mean"),
            avg_session_duration_sec=("avg_session_duration_sec", "mean"),
        )
        .reset_index()
    )

    shipment_daily = shipments.copy()
    shipment_daily["delivery_days"] = (
        shipment_daily["delivery_date"] - shipment_daily["ship_date"]
    ).dt.days.clip(lower=0)
    shipment_daily = (
        shipment_daily.groupby("ship_date")
        .agg(
            avg_delivery_days=("delivery_days", "mean"),
            shipping_fee=("shipping_fee", "mean"),
        )
        .reset_index()
        .rename(columns={"ship_date": "date"})
    )

    return_daily = (
        returns.groupby("return_date")
        .agg(
            return_orders=("order_id", "nunique"),
            refund_amount=("refund_amount", "sum"),
        )
        .reset_index()
        .rename(columns={"return_date": "date"})
    )

    review_daily = (
        reviews.groupby("review_date")
        .agg(
            avg_rating=("rating", "mean"),
            review_count=("review_id", "count"),
        )
        .reset_index()
        .rename(columns={"review_date": "date"})
    )

    daily = sales.merge(orders_daily, on="date", how="left")
    for frame in [promo_daily, traffic_daily, shipment_daily, return_daily, review_daily]:
        daily = daily.merge(frame, on="date", how="left")

    daily = daily.sort_values("date").reset_index(drop=True)
    fill_zero = [
        "orders",
        "delivered_orders",
        "returned_orders",
        "cancelled_orders",
        "promo_order_share",
        "discount_amount",
        "item_revenue",
        "item_cogs",
        "sessions",
        "page_views",
        "unique_visitors",
        "return_orders",
        "refund_amount",
        "review_count",
    ]
    for column in fill_zero:
        if column in daily:
            daily[column] = daily[column].fillna(0.0)

    daily["orders"] = daily["orders"].replace(0, np.nan)
    daily["sessions"] = daily["sessions"].replace(0, np.nan)
    daily["revenue_per_order"] = daily["Revenue"] / daily["orders"]
    daily["cogs_ratio"] = daily["COGS"] / daily["Revenue"].replace(0, np.nan)
    daily["conversion"] = daily["orders"] / daily["sessions"]
    daily["page_views_per_session"] = daily["page_views"] / daily["sessions"]
    daily["discount_rate"] = daily["discount_amount"] / (daily["item_revenue"] + daily["discount_amount"]).replace(0, np.nan)
    daily["gross_profit"] = daily["Revenue"] - daily["COGS"]
    daily["gross_margin"] = daily["gross_profit"] / daily["Revenue"].replace(0, np.nan)
    daily["date"] = pd.to_datetime(daily["date"])
    daily["year"] = daily["date"].dt.year
    daily["month"] = daily["date"].dt.month
    daily["day"] = daily["date"].dt.day
    daily["day_of_week"] = daily["date"].dt.dayofweek
    return daily


def build_monthly_inventory_frame(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    inventory = data["inventory"].copy()
    inventory["date"] = inventory["snapshot_date"]
    inventory["month"] = inventory["date"].dt.month
    inventory["year"] = inventory["date"].dt.year
    inventory["turnover_ratio"] = inventory["units_sold"] / (
        inventory["stock_on_hand"] + inventory["units_received"]
    ).replace(0, np.nan)
    monthly = (
        inventory.groupby(["year", "month", "date"])
        .agg(
            avg_dos=("days_of_supply", "mean"),
            overstock_rate=("overstock_flag", "mean"),
            stockout_rate=("stockout_flag", "mean"),
            fill_rate=("fill_rate", "mean"),
            turnover_ratio=("turnover_ratio", "mean"),
            stock_on_hand=("stock_on_hand", "sum"),
            units_sold=("units_sold", "sum"),
        )
        .reset_index()
        .sort_values("date")
    )
    return monthly


def build_segment_inventory_frame(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    inventory = data["inventory"].copy()
    inventory["revenue_proxy"] = inventory["units_sold"] * inventory["stock_on_hand"].clip(lower=1)
    frame = (
        inventory.groupby(["category", "segment"])
        .agg(
            overstock_rate=("overstock_flag", "mean"),
            stockout_rate=("stockout_flag", "mean"),
            avg_dos=("days_of_supply", "mean"),
            stock_on_hand=("stock_on_hand", "sum"),
            units_sold=("units_sold", "sum"),
        )
        .reset_index()
    )
    frame["revenue_share_proxy"] = frame["units_sold"] / frame["units_sold"].sum()
    return frame.sort_values("revenue_share_proxy", ascending=False)


def build_sku_inventory_frame(data: dict[str, pd.DataFrame], top_n: int = 12) -> pd.DataFrame:
    inventory = data["inventory"].copy()
    products = data["products"].copy()
    product_cost = products[["product_id", "product_name", "category", "segment", "cogs"]]
    inventory = inventory.merge(product_cost, on=["product_id", "product_name", "category", "segment"], how="left")
    inventory["capital_locked"] = inventory["stock_on_hand"] * inventory["cogs"].fillna(0.0)
    sku = (
        inventory.groupby(["product_id", "product_name", "category", "segment"])
        .agg(
            avg_dos=("days_of_supply", "mean"),
            stock_on_hand=("stock_on_hand", "sum"),
            capital_locked=("capital_locked", "sum"),
            sell_through_rate=("sell_through_rate", "mean"),
        )
        .reset_index()
    )
    sku["label"] = sku["product_name"] + " | " + sku["category"]
    return sku.sort_values("capital_locked", ascending=False).head(top_n)


def sample_dates() -> pd.DatetimeIndex:
    sample = pd.read_csv(DATASET_DIR / "sample_submission.csv", parse_dates=["Date"])
    return pd.DatetimeIndex(sample["Date"])

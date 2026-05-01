from __future__ import annotations

import csv
import importlib.util
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks" / "part2_eda"
FIGURES_DIR = ROOT / "figures"
FINAL_DIR = FIGURES_DIR / "final_figures"
EDA_STAGES = ["descriptive", "diagnostic", "predictive", "prescriptive"]

NOTEBOOKS = [
    NOTEBOOK_DIR / "01_problem_overview.ipynb",
    NOTEBOOK_DIR / "02_promotion_and_demand.ipynb",
    NOTEBOOK_DIR / "03_inventory_and_operations.ipynb",
    NOTEBOOK_DIR / "04_customer_and_returns.ipynb",
    NOTEBOOK_DIR / "05_final_figures.ipynb",
]

APPROVED_EXPORTS = [
    {
        "index": 1,
        "title": "Chất lượng cầu suy giảm mạnh sau năm 2019 dù lưu lượng truy cập tiếp tục tăng",
        "category": "problem_overview",
        "filename": "01_demand_quality_break.png",
        "notebook": "01_problem_overview.ipynb",
        "stage": "descriptive",
    },
    {
        "index": 2,
        "title": "Margin theo category khi có và không có promo",
        "category": "promotion_and_demand",
        "filename": "02_margin_by_category.png",
        "notebook": "02_promotion_and_demand.ipynb",
        "stage": "diagnostic",
    },
    {
        "index": 3,
        "title": "Average order value trước và sau promo theo category",
        "category": "promotion_and_demand",
        "filename": "03_aov_before_after_promo.png",
        "notebook": "02_promotion_and_demand.ipynb",
        "stage": "diagnostic",
    },
    {
        "index": 4,
        "title": "Tỷ trọng traffic và tỷ trọng đơn hàng theo kênh",
        "category": "promotion_and_demand",
        "filename": "04_channel_share_efficiency.png",
        "notebook": "02_promotion_and_demand.ipynb",
        "stage": "diagnostic",
    },
    {
        "index": 5,
        "title": "Conversion theo kênh trước và sau 2019",
        "category": "promotion_and_demand",
        "filename": "05_channel_conversion_before_after_2019.png",
        "notebook": "02_promotion_and_demand.ipynb",
        "stage": "diagnostic",
    },
    {
        "index": 6,
        "title": "Promo đang đi cùng mùa, nhưng không tạo incremental GP",
        "category": "promotion_and_demand",
        "filename": "04_promo_no_incremental_gp.png",
        "notebook": "02_promotion_and_demand.ipynb",
        "stage": "diagnostic",
    },
    {
        "index": 7,
        "title": "Nguyên nhân chính của cú rơi 2019: inventory stress tăng lên từ 2018 H2 và kéo conversion đi xuống",
        "category": "inventory_and_operations",
        "filename": "05_root_cause_2019_inventory_stress.png",
        "notebook": "03_inventory_and_operations.ipynb",
        "stage": "diagnostic",
    },
    {
        "index": 8,
        "title": "Khủng hoảng tồn kho có tính cấu trúc: các segment lớn nhất đang vừa overstock vừa stockout",
        "category": "inventory_and_operations",
        "filename": "06_inventory_stress_matrix.png",
        "notebook": "03_inventory_and_operations.ipynb",
        "stage": "diagnostic",
    },
    {
        "index": 9,
        "title": "Timeline 2018-2020: dip 2019 là hệ quả của inventory health xấu đi, không phải giao hàng chậm",
        "category": "inventory_and_operations",
        "filename": "08_timeline_2018_2020_inventory_vs_logistics.png",
        "notebook": "03_inventory_and_operations.ipynb",
        "stage": "diagnostic",
    },
    {
        "index": 10,
        "title": "Giá trị refund theo lý do trả hàng",
        "category": "customer_and_returns",
        "filename": "11_refund_amount_by_reason.png",
        "notebook": "04_customer_and_returns.ipynb",
        "stage": "diagnostic",
    },
    {
        "index": 11,
        "title": "Cơ cấu lý do trả hàng giữa đơn promo và không promo",
        "category": "customer_and_returns",
        "filename": "12_return_reason_mix.png",
        "notebook": "04_customer_and_returns.ipynb",
        "stage": "diagnostic",
    },
    {
        "index": 12,
        "title": "Xu hướng sức khỏe tồn kho theo DOS qua thời gian",
        "category": "inventory_and_operations",
        "filename": "10_dos_trend_over_time.png",
        "notebook": "03_inventory_and_operations.ipynb",
        "stage": "descriptive",
    },
    {
        "index": 13,
        "title": "Giá trị chưa ghi nhận từ hủy đơn và hoàn tiền",
        "category": "promotion_and_demand",
        "filename": "14_leakage_unrealized_value.png",
        "notebook": "02_promotion_and_demand.ipynb",
        "stage": "prescriptive",
    },
    {
        "index": 14,
        "title": "Dải phục hồi doanh thu và GP theo kịch bản kiểm soát leakage",
        "category": "promotion_and_demand",
        "filename": "15_leakage_recovery_scenarios.png",
        "notebook": "02_promotion_and_demand.ipynb",
        "stage": "prescriptive",
    },
    {
        "index": 15,
        "title": "Residual COGS ratio theo thời gian và August anomaly",
        "category": "inventory_and_operations",
        "filename": "16_cogs_anomaly_timeline.png",
        "notebook": "03_inventory_and_operations.ipynb",
        "stage": "predictive",
    },
    {
        "index": 16,
        "title": "Tác động sang tháng sau theo regime COGS hiện tại",
        "category": "inventory_and_operations",
        "filename": "17_cogs_regime_next_month.png",
        "notebook": "03_inventory_and_operations.ipynb",
        "stage": "predictive",
    },
    {
        "index": 17,
        "title": "Tỷ trọng doanh thu theo customer-value decile",
        "category": "customer_and_returns",
        "filename": "18_revenue_concentration_by_decile.png",
        "notebook": "04_customer_and_returns.ipynb",
        "stage": "predictive",
    },
    {
        "index": 18,
        "title": "Độ nhạy doanh thu khi churn tăng ở nhóm khách giá trị cao",
        "category": "customer_and_returns",
        "filename": "19_retention_sensitivity_high_value.png",
        "notebook": "04_customer_and_returns.ipynb",
        "stage": "predictive",
    },
    {
        "index": 19,
        "title": "Trade-off doanh thu và GP khi cắt promo âm economics",
        "category": "inventory_and_operations",
        "filename": "20_action_tradeoff_scenarios.png",
        "notebook": "03_inventory_and_operations.ipynb",
        "stage": "prescriptive",
    },
    {
        "index": 20,
        "title": "Thứ tự 5 can thiệp cần làm trước",
        "category": "inventory_and_operations",
        "filename": "21_action_priority_ladder.png",
        "notebook": "03_inventory_and_operations.ipynb",
        "stage": "prescriptive",
    },
]


def ensure_runtime() -> None:
    required = ["matplotlib", "seaborn", "nbformat", "nbclient"]
    if all(importlib.util.find_spec(module) for module in required):
        return
    raise RuntimeError(
        "A Python runtime with matplotlib, seaborn, nbformat, and nbclient is required. "
        "Install project dependencies first, for example with `uv sync`."
    )


ensure_runtime()

import nbformat
from nbclient import NotebookClient


def ensure_directories() -> None:
    for folder in [
        FIGURES_DIR / "problem_overview",
        FIGURES_DIR / "promotion_and_demand",
        FIGURES_DIR / "inventory_and_operations",
        FIGURES_DIR / "customer_and_returns",
        FINAL_DIR,
    ]:
        folder.mkdir(parents=True, exist_ok=True)
    for stage in EDA_STAGES:
        (FINAL_DIR / stage).mkdir(parents=True, exist_ok=True)


def clear_release_outputs() -> None:
    for folder in [
        FIGURES_DIR / "problem_overview",
        FIGURES_DIR / "promotion_and_demand",
        FIGURES_DIR / "inventory_and_operations",
        FIGURES_DIR / "customer_and_returns",
        FINAL_DIR,
    ]:
        for path in folder.glob("*.png"):
            path.unlink()
    for stage in EDA_STAGES:
        stage_dir = FINAL_DIR / stage
        for path in stage_dir.glob("*.png"):
            path.unlink()


def execute_notebook(path: Path) -> None:
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=1200,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.execute()
    nbformat.write(notebook, path)


def verify_and_copy_exports() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in APPROVED_EXPORTS:
        source = FIGURES_DIR / item["category"] / item["filename"]
        if not source.exists():
            raise FileNotFoundError(f"Missing expected export: {source}")
        target = FINAL_DIR / item["filename"]
        stage_target = FINAL_DIR / item["stage"] / item["filename"]
        shutil.copy2(source, target)
        shutil.copy2(source, stage_target)
        rows.append(
            {
                "index": str(item["index"]),
                "title": item["title"],
                "filename": item["filename"],
                "notebook": item["notebook"],
                "category": item["category"],
                "eda_stage": item["stage"],
                "category_copy": str(source),
                "final_copy": str(target),
                "stage_copy": str(stage_target),
            }
        )
    return rows


def write_manifest(rows: list[dict[str, str]]) -> None:
    manifest_path = FINAL_DIR / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "index",
                "title",
                "filename",
                "notebook",
                "category",
                "eda_stage",
                "category_copy",
                "final_copy",
                "stage_copy",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ensure_directories()
    clear_release_outputs()
    for notebook in NOTEBOOKS[:4]:
        execute_notebook(notebook)
    rows = verify_and_copy_exports()
    write_manifest(rows)
    execute_notebook(NOTEBOOKS[4])
    print(f"Rebuilt {len(rows)} approved Part 2 figures.")
    print(f"Release set: {FINAL_DIR}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.data import canonical_submission_path, copy_file
from models.final_meta_regime_ensemble import build_final_meta_regime_ensemble


APPENDIX_DIR = ROOT / "paper" / "neurips" / "appendix"
MODELING_FIGURES_DIR = ROOT / "figures" / "modeling"


def ensure_output_dirs() -> None:
    APPENDIX_DIR.mkdir(parents=True, exist_ok=True)
    MODELING_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _ordered_columns(columns: list[str]) -> list[str]:
    order = ["global", "early", "mid", "mid_early", "mid_late", "late", "normal", "sale_window", "odd_august"]
    head = [name for name in order if name in columns]
    tail = [name for name in columns if name not in head]
    return head + tail


def export_appendix_tables(outputs: dict[str, object]) -> None:
    candidate_table = outputs["candidate_table"].copy()
    candidate_table.to_csv(APPENDIX_DIR / "meta_candidate_results.csv", index=False)

    final_weights = outputs["final_weights_table"].copy()
    final_weights.to_csv(APPENDIX_DIR / "meta_final_weights.csv", index=False)

    for name, frame in outputs["explainability"].items():
        frame.to_csv(APPENDIX_DIR / f"{name}.csv", index=False)


def _plot_weight_heatmap(weights: pd.DataFrame, target: str, ax: plt.Axes, title: str) -> None:
    frame = weights[(weights["target"] == target) & (weights["component"] != "intercept")].copy()
    pivot = frame.pivot_table(index="component", columns="segment", values="weight", aggfunc="first").fillna(0.0)
    pivot = pivot.reindex(columns=_ordered_columns(list(pivot.columns)))
    sns.heatmap(pivot, cmap="RdBu_r", center=0.0, annot=True, fmt=".2f", cbar=True, ax=ax)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Segment")
    ax.set_ylabel("Component")


def _plot_shap_summary(summary: pd.DataFrame, ax: plt.Axes, title: str, color: str) -> None:
    top = summary.head(10).iloc[::-1]
    ax.barh(top["feature"], top["mean_abs_shap"], color=color)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Mean |SHAP|")
    ax.set_ylabel("")


def export_modeling_figures(outputs: dict[str, object]) -> None:
    sns.set_theme(style="whitegrid", context="talk")

    weights = outputs["final_weights_table"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    _plot_weight_heatmap(weights, "revenue", axes[0], "Revenue meta weights theo horizon")
    _plot_weight_heatmap(weights, "ratio", axes[1], "Ratio meta weights theo regime")
    plt.tight_layout()
    fig.savefig(MODELING_FIGURES_DIR / "21_meta_horizon_weight_heatmap.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    explain = outputs["explainability"]
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    _plot_shap_summary(explain["revenue_recursive"], axes[0, 0], "SHAP: Revenue recursive head", "#1f77b4")
    _plot_shap_summary(explain["revenue_regime"], axes[0, 1], "SHAP: Revenue regime head", "#ff7f0e")
    _plot_shap_summary(explain["ratio_recursive"], axes[1, 0], "SHAP: Ratio recursive head", "#2ca02c")
    _plot_shap_summary(explain["ratio_regime"], axes[1, 1], "SHAP: Ratio regime head", "#d62728")
    plt.tight_layout()
    fig.savefig(MODELING_FIGURES_DIR / "22_meta_horizon_shap_summary.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    stability = explain["feature_stability_summary"].copy()
    models_to_plot = ["revenue_recursive", "revenue_regime", "ratio_recursive", "ratio_regime"]
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()
    for ax, model_name in zip(axes, models_to_plot):
        top = stability[stability["model"] == model_name].head(8).iloc[::-1]
        ax.barh(
            top["feature"],
            top["mean_abs_shap"],
            xerr=top["std_abs_shap"].fillna(0.0),
            color="#4c78a8",
        )
        ax.set_title(f"Stability by fold: {model_name}", fontweight="bold")
        ax.set_xlabel("Mean |SHAP| ± std")
        ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(MODELING_FIGURES_DIR / "23_meta_horizon_feature_stability.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    ensure_output_dirs()
    meta_outputs = build_final_meta_regime_ensemble()
    export_appendix_tables(meta_outputs)
    export_modeling_figures(meta_outputs)
    canonical_path = copy_file(meta_outputs["final_submission"], canonical_submission_path())
    print("Final submission model:", meta_outputs["final_candidate_name"])
    print("Internal CV winner:", meta_outputs["selected_config"]["internal_cv_winner_name"])
    print("Final submission objective:", float(meta_outputs["practical_objective"]))
    print("Final submission file:", meta_outputs["final_submission"])
    print("Canonical submission:", canonical_path)


if __name__ == "__main__":
    main()

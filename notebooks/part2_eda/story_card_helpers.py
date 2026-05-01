from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT / "figures"
FINAL_FIGURES_DIR = FIGURES_DIR / "final_figures"

NAVY = "#1F2A44"
MUTED = "#5B6B7A"
GRID = "#D7DEE7"
PANEL = "#F7FAFC"
BORDER = "#E2E8F0"
BLUE = "#2563EB"
ORANGE = "#EA580C"
RED = "#DC2626"
GREEN = "#16A34A"
GRAY = "#94A3B8"


def setup_story_theme() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#CBD5E1",
            "axes.linewidth": 0.8,
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "font.family": "DejaVu Sans",
            "grid.color": GRID,
            "grid.alpha": 0.55,
            "grid.linestyle": "-",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
        }
    )


def create_story_card(
    *,
    top_shape: tuple[int, int],
    figsize: tuple[float, float],
    include_footer: bool = False,
) -> tuple[plt.Figure, list[plt.Axes], list[plt.Axes]]:
    fig = plt.figure(figsize=figsize, facecolor="white")
    chart_axes: list[plt.Axes] = []
    footer_axes: list[plt.Axes] = []

    if include_footer:
        outer = fig.add_gridspec(2, 1, height_ratios=[7.4, 2.6], hspace=0.12)
        top = outer[0].subgridspec(top_shape[0], top_shape[1], hspace=0.30, wspace=0.18)
        for row in range(top_shape[0]):
            for col in range(top_shape[1]):
                chart_axes.append(fig.add_subplot(top[row, col]))

        footer = outer[1].subgridspec(1, 3, wspace=0.08)
        for idx in range(3):
            ax = fig.add_subplot(footer[0, idx])
            ax.set_axis_off()
            ax.add_patch(
                Rectangle(
                    (0, 0),
                    1,
                    1,
                    transform=ax.transAxes,
                    facecolor=PANEL,
                    edgecolor=BORDER,
                    linewidth=1.0,
                )
            )
            footer_axes.append(ax)
    else:
        top = fig.add_gridspec(
            top_shape[0],
            top_shape[1],
            left=0.06,
            right=0.96,
            bottom=0.10,
            top=0.86,
            hspace=0.32,
            wspace=0.20,
        )
        for row in range(top_shape[0]):
            for col in range(top_shape[1]):
                chart_axes.append(fig.add_subplot(top[row, col]))

    return fig, chart_axes, footer_axes


def apply_story_header(
    fig: plt.Figure,
    top_axis: plt.Axes,
    *,
    figure_title: str,
    subtitle: str,
    encoding_line: str,
    show_subtitle: bool = False,
    show_encoding_line: bool = False,
) -> None:
    # Final notebook/export figures are chart-only.
    # Narrative titles and explanations live in markdown cells and in the paper.
    top_axis.set_title("")
    fig.subplots_adjust(top=0.96, left=0.06, right=0.96, bottom=0.08)


def fill_footer(footer_axes: list[plt.Axes], sections: list[tuple[str, str]]) -> None:
    if not footer_axes:
        return
    for ax, (title, body) in zip(footer_axes, sections):
        ax.text(
            0.04,
            0.86,
            title,
            ha="left",
            va="top",
            fontsize=15,
            fontweight="bold",
            color="#111827",
            transform=ax.transAxes,
        )
        wrapped = textwrap.fill(body, width=40)
        ax.text(
            0.04,
            0.68,
            wrapped,
            ha="left",
            va="top",
            fontsize=11.5,
            color="#475569",
            linespacing=1.45,
            transform=ax.transAxes,
        )


def save_story_card(fig: plt.Figure, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=220,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.15,
    )
    plt.close(fig)
    return output_path


def wrap_text(text: str, width: int = 60) -> str:
    return textwrap.fill(text, width=width)

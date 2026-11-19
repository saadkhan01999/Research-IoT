"""Figure style for the paper: one look for every figure.

Rules (author's brief): vector PDF plus 300-dpi PNG; single column 3.5 in, double column
7.2 in; serif font; no text smaller than 8 pt; no title inside the figure (the caption is in
LaTeX); labelled axes with units; the palette below in this order; hatches or marker shapes
on every series so a black-and-white print still reads (three colours have under 3:1
contrast on white: checked with the dataviz validator, 2026-11-11); one-hue blue for
heatmaps; no rainbow maps; never two y-axes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # fixed order, never cycled
HATCHES = ["", "///", "\\\\\\", "xx", ".."]
MARKERS = ["o", "s", "^", "D", "v"]
INK, MUTED, GRID = "#1f2328", "#59636e", "#d0d7de"
SINGLE, DOUBLE = 3.5, 7.2
BLUES = LinearSegmentedColormap.from_list("one_hue_blue", ["#f4f8fd", "#9cc3ef", "#2a78d6",
                                                            "#0b3d7a"])


def setup() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8, "axes.edgecolor": MUTED,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02, "legend.frameon": False,
    })


def style(i: int) -> dict[str, str]:
    """Colour, hatch and marker of the i-th series (fixed order)."""
    if i >= len(PALETTE):
        raise ValueError("more than five series: fold into 'other' or use small multiples")
    return {"color": PALETTE[i], "hatch": HATCHES[i], "marker": MARKERS[i]}


def save(fig: plt.Figure, out_dir: Path, name: str) -> list[Path]:
    """Write <name>.pdf (vector) and <name>.png (300 dpi); close the figure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.pdf", out_dir / f"{name}.png"]
    fig.savefig(paths[0])
    fig.savefig(paths[1], dpi=300)
    plt.close(fig)
    return paths


def box(ax: plt.Axes, x: float, y: float, w: float, h: float, text: str,
        fill: str = "#ffffff", edge: str = INK, size: float = 8, bold: bool = False) -> None:
    """A labelled rectangle for schematic figures (axes in data units 0..1)."""
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.01",
                                facecolor=fill, edgecolor=edge, linewidth=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax: plt.Axes, xy_from: tuple[float, float], xy_to: tuple[float, float],
          style_: str = "-|>", color: str = MUTED, dashed: bool = False) -> None:
    ax.annotate("", xy=xy_to, xytext=xy_from,
                arrowprops={"arrowstyle": style_, "color": color, "lw": 0.8,
                            "linestyle": "--" if dashed else "-", "shrinkA": 1, "shrinkB": 1})


def blank(ax: plt.Axes) -> None:
    """Turn an axes into a drawing canvas (no ticks, grid or frame)."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

"""Step 9: every figure of the paper, drawn from saved results.

Usage: python scripts/09_figures.py --profile quick|full [--only NAME ...]

Each figure reads only files written by Steps 2-8 (results/...). A figure whose inputs do not
exist yet is skipped with a note, so the script can run before every experiment has finished.
Output: paper/figures/<name>.pdf and .png (the quick profile writes to paper/figures_quick/ so
draft figures never mix with the final ones) and a manifest listing the source files of every
figure. Style rules are in src/iotml/plots.py.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from iotml import plots
from iotml.config import project_path
from iotml.logs import get_logger
from iotml.results import load_per_class, load_runs, seen_family_mix

plots.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

MODELS = ["decision_tree", "random_forest", "extra_trees", "xgboost", "lightgbm"]
MODEL_LABEL = {"decision_tree": "DT", "random_forest": "RF", "extra_trees": "ET",
               "xgboost": "XGBoost", "lightgbm": "LightGBM"}
PROTOCOLS = ["random", "random_dedup", "file_grouped", "order_within_file"]
PROTOCOL_LABEL = {"random": "Random","random_dedup": "Random,\ndeduplicated",
                  "file_grouped": "File-\ngrouped", "order_within_file": "Ordered\nwithin file"}
PROTOCOL_SHORT = {"random": "Random", "random_dedup": "Dedup.", "file_grouped": "File",
                  "order_within_file": "Ordered"}   # same short names as Table 6
SCHEME_LABEL = {"L34": "34 labels", "L8": "8 families", "LM": "30 merged labels",
                "L2": "Benign vs attack"}
FAMILY_ORDER = ["DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web", "BruteForce", "Benign"]
TREATMENTS = ["none", "weights", "under", "smotetomek"]
TREATMENT_LABEL = {"none": "None", "weights": "Class\nweights", "under": "Under-\nsampling",
                   "smotetomek": "SMOTE-\nTomek"}


class MissingInputError(Exception):
    """An input file of a figure does not exist yet."""


def need(path: Path) -> Path:
    if not path.exists():
        raise MissingInputError(rel(path))
    return path


def rel(path: Path) -> str:
    return str(path.relative_to(project_path("."))).replace("\\", "/")


@dataclass
class Ctx:
    profile: str
    out: Path
    analysis: Path
    _runs: pd.DataFrame | None = None
    _per_class: pd.DataFrame | None = None
    sources: list[str] = field(default_factory=list)

    def use(self, path: Path) -> Path:
        self.sources.append(rel(need(path)))
        return path

    def runs(self) -> pd.DataFrame:
        if self._runs is None:
            self._runs = load_runs(self.profile)
        if self._runs.empty:
            raise MissingInputError(f"results/experiments/**/*_{self.profile}.json")
        self.sources.append(f"results/experiments/**/*_{self.profile}.json")
        return self._runs

    def main_runs(self) -> pd.DataFrame:
        r = self.runs()
        keep = (r["train_mode"] == "capped") & r["params_mode"].isin(
            ["tuned", "tuned_from_random"]) & r["protocol"].isin(PROTOCOLS) \
            & r["model"].isin(MODELS)
        return r[keep]

    def per_class(self) -> pd.DataFrame:
        if self._per_class is None:
            self._per_class = load_per_class(self.profile)
        if self._per_class.empty:
            raise MissingInputError(f"results/experiments/**/*_{self.profile}.json")
        return self._per_class

    def best_model(self, scheme: str) -> str:
        """Model with the highest mean macro-F1 on the deduplicated split (random if absent)."""
        r = self.main_runs()
        for protocol in ("random_dedup", "random"):
            sub = r[(r["protocol"] == protocol) & (r["scheme"] == scheme)]
            if not sub.empty:
                return str(sub.groupby("model")["macro_f1"].mean().idxmax())
        raise MissingInputError(f"main runs for scheme {scheme}")


FIGURES: dict[str, Callable[[Ctx], plt.Figure]] = {}


def figure(name: str) -> Callable[[Callable[[Ctx], plt.Figure]], Callable[[Ctx], plt.Figure]]:
    def register(fn: Callable[[Ctx], plt.Figure]) -> Callable[[Ctx], plt.Figure]:
        FIGURES[name] = fn
        return fn
    return register


def millions(n: float) -> str:
    return f"{n / 1e6:.2f} M"


def audit(ctx: Ctx) -> dict:
    return json.loads(ctx.use(project_path("results/audit/audit_summary.json")).read_text(
        encoding="utf-8"))


# ======================================================================= schematic figures
@figure("fig01_testbed")
def fig_testbed(ctx: Ctx) -> plt.Figure:
    """CICIoT2023 testbed, redrawn from the text of the dataset paper (Section III of [A02])."""
    ctx.use(project_path("refs/fulltext/A02/text.txt"))
    fig, ax = plt.subplots(figsize=(plots.DOUBLE, 2.9))
    plots.blank(ax)
    ax.add_patch(Rectangle((0.0, 0.0), 0.605, 1.0, facecolor="#eef4fc", edgecolor="none"))
    ax.add_patch(Rectangle((0.615, 0.0), 0.385, 1.0, facecolor="#fdf1ec", edgecolor="none"))
    ax.text(0.30, 0.935, "Part 1: Internet access and attacking devices", ha="center",
            color=plots.MUTED)
    ax.text(0.807, 0.935, "Part 2: gateways and victim IoT devices", ha="center",
            color=plots.MUTED)
    w, h = 0.125, 0.15
    nodes = {
        "net": (0.015, 0.74, "Internet"), "asus": (0.165, 0.74, "ASUS\nrouter"),
        "win": (0.315, 0.74, "Windows 10\ndesktop"), "cisco": (0.465, 0.74, "Cisco\nswitch"),
        "ap": (0.465, 0.44, "VeraPlus\naccess point"),
        "rpi": (0.465, 0.14, "7 Raspberry Pi\nattackers"),
        "tap": (0.64, 0.74, "Gigamon\nnetwork tap"),
        "mon": (0.81, 0.74, "2 monitors\n(Wireshark,\nmergecap)"),
        "netgear": (0.64, 0.44, "Netgear\nunmanaged\nswitch"),
        "gw": (0.81, 0.44, "5 gateways and\nbase stations\n(Zigbee, Z-Wave)"),
        "vera": (0.64, 0.14, "VeraPlus\ncontroller"),
        "victims": (0.81, 0.14, "2 Zigbee/Z-Wave\nhubs and victim\nIoT devices"),
    }
    for key, (x, y, text) in nodes.items():
        plots.box(ax, x, y, 0.175 if x > 0.8 else w, h, text,
                  fill="#fff7e0" if key in ("tap", "mon") else "#ffffff")

    def c(key: str, side: str) -> tuple[float, float]:
        x, y, _ = nodes[key]
        ww = 0.175 if x > 0.8 else w
        return {"l": (x, y + h / 2), "r": (x + ww, y + h / 2), "t": (x + ww / 2, y + h),
                "b": (x + ww / 2, y)}[side]

    for a, b in [("net", "asus"), ("asus", "win"), ("win", "cisco"), ("cisco", "tap"),
                 ("tap", "netgear"), ("netgear", "gw")]:
        sa, sb = ("r", "l") if nodes[a][1] == nodes[b][1] else ("b", "t")
        plots.arrow(ax, c(a, sa), c(b, sb), style_="-")
    for a, b in [("cisco", "ap"), ("ap", "rpi"), ("netgear", "vera")]:
        plots.arrow(ax, c(a, "b"), c(b, "t"), style_="-")
    plots.arrow(ax, c("vera", "r"), c("victims", "l"), style_="-")
    plots.arrow(ax, c("tap", "r"), c("mon", "l"), dashed=True)
    ax.annotate("", xy=(0.8975, 0.135), xytext=(0.5275, 0.135),
                arrowprops={"arrowstyle": "-|>", "color": plots.PALETTE[1], "lw": 1.0,
                            "linestyle": "--", "connectionstyle": "arc3,rad=0.22"})
    ax.text(0.71, 0.075, "attack traffic", ha="center", color=plots.INK)
    return fig


@figure("fig02_pipeline")
def fig_pipeline(ctx: Ctx) -> plt.Figure:
    a = audit(ctx)
    files, rows = a["counts"]["source_files"], a["counts"]["rows"]
    fig, ax = plt.subplots(figsize=(plots.DOUBLE, 2.05))
    plots.blank(ax)
    top = [f"Per-capture CSV\nrelease: {files} files,\n{millions(rows)} rows",
           "Columnar cache\n(label checked\nagainst folder)",
           "Audit: duplicates,\nlabel conflicts,\nceilings, row order",
           "Label schemes\nL34, L8, LM, L2",
           "Split protocols\n(random, dedup, file,\norder, unseen type)"]
    bottom = ["Statistics: bootstrap\nby capture file,\nFriedman, Holm",
              "Natural-mix test\nscores (inverse\nsampling weights)",
              "Models: DT, RF, ET,\nXGBoost, LightGBM,\nMLP, 1D-CNN, GRU",
              "Tuning on the\nvalidation part\n(macro-F1)",
              "Training sample\ncapped per label,\ntraining part only"]
    w, h, gap = 0.18, 0.36, 0.0225
    xs = [0.005 + i * (w + gap) for i in range(5)]
    for i, text in enumerate(top):
        plots.box(ax, xs[i], 0.58, w, h, text, fill="#eef4fc" if i in (2, 4) else "#ffffff")
        if i < 4:
            plots.arrow(ax, (xs[i] + w, 0.76), (xs[i + 1], 0.76))
    for i, text in enumerate(bottom):
        plots.box(ax, xs[i], 0.04, w, h, text, fill="#eef4fc" if i in (0, 1) else "#ffffff")
        if i < 4:
            plots.arrow(ax, (xs[i + 1], 0.22), (xs[i] + w, 0.22))
    plots.arrow(ax, (xs[4] + w / 2, 0.58), (xs[4] + w / 2, 0.40))
    return fig


def _toy_protocols(seed: int = 7) -> tuple[list[str], list[str], np.ndarray, dict]:
    """Toy rows of four capture files (three of class A, one of class B) and their parts under
    each protocol, following the rules of src/iotml/splits.py. Values: 0 train, 1 validation,
    2 test, -1 not used. Also returns, per protocol, which evaluation rows repeat a training
    vector (leaked copies)."""
    rng = np.random.default_rng(seed)
    files = ["A1", "A2", "A3", "B1"]
    n = 24
    vec = np.zeros((4, n), dtype=int)
    next_id = 0
    for f in range(4):
        for i in range(n):
            pool = vec[:f + 1].ravel()[: f * n + i] if f < 3 else vec[3, :i]
            pool = pool[pool > 0]
            if pool.size and rng.random() < 0.35:
                vec[f, i] = rng.choice(pool)
            else:
                next_id += 1
                vec[f, i] = next_id
    shares = [0.7, 0.1, 0.2]

    def block(m: int) -> np.ndarray:
        n_tr, n_va = round(shares[0] * m), round(shares[1] * m)
        out = np.full(m, -1)
        out[:n_tr - 1] = 0
        out[n_tr:n_tr + n_va] = 1
        out[n_tr + n_va + 1:] = 2
        return out

    def purge(p: np.ndarray) -> np.ndarray:
        train_vec = set(vec[p == 0].ravel())
        p = p.copy()
        p[(p > 0) & np.isin(vec, list(train_vec))] = -1
        return p

    parts: dict[str, np.ndarray] = {}
    parts["random"] = rng.choice(3, size=(4, n), p=shares)
    vec_part = {v: rng.choice(3, p=shares) for v in np.unique(vec)}
    first = np.zeros_like(vec, dtype=bool)
    seen: set[int] = set()
    for f in range(4):
        for i in range(n):
            if vec[f, i] not in seen:
                first[f, i] = True
                seen.add(vec[f, i])
    dedup = np.vectorize(vec_part.get)(vec)
    dedup[~first] = -1
    parts["random_dedup"] = dedup
    fg = np.vstack([np.zeros(n, int), np.ones(n, int), np.full(n, 2), block(n)])
    parts["file_grouped"] = purge(fg)
    parts["order_within_file"] = purge(np.vstack([block(n)] * 4))
    unseen = dedup.copy()
    unseen[3][unseen[3] < 2] = -1
    parts["unseen_type"] = unseen
    leaked = {k: (p > 0) & np.isin(vec, list(set(vec[p == 0].ravel()))) for k, p in
              parts.items()}
    return files, list(parts), np.stack([parts[k] for k in parts]), leaked


@figure("fig03_protocols")
def fig_protocols(ctx: Ctx) -> plt.Figure:
    files, names, parts, leaked = _toy_protocols()
    labels = {"random": "Random", "random_dedup": "Random, deduplicated",
              "file_grouped": "File-grouped", "order_within_file": "Ordered within file",
              "unseen_type": "Unseen type (B held out)"}
    colors = {0: plots.PALETTE[0], 1: plots.PALETTE[1], 2: plots.PALETTE[2], -1: "#e6e8eb"}
    hatch = {0: "", 1: "////", 2: "xxxx", -1: ""}
    # One column wide (page budget): 24 toy rows per file still give cells of about 0.12 in.
    fig, ax = plt.subplots(figsize=(plots.SINGLE, 4.0))
    ax.set_xlim(-1.4, 24.1)
    panel = 5.3
    n_rows = len(names) * panel
    ax.set_ylim(n_rows + 0.3, -0.1)
    ax.axis("off")
    plt.rcParams["hatch.linewidth"] = 0.4
    for p, name in enumerate(names):
        y0 = p * panel + 1.1
        ax.text(-1.3, y0 - 0.15, labels[name], va="bottom", ha="left", color=plots.INK,
                fontweight="bold", fontsize=6.5)
        for f, fname in enumerate(files):
            y = y0 + f
            ax.text(-0.3, y + 0.42, fname, va="center", ha="right", color=plots.MUTED,
                    fontsize=6)
            for i in range(parts.shape[2]):
                v = int(parts[p, f, i])
                ax.add_patch(Rectangle((i + 0.06, y + 0.06), 0.88, 0.76, facecolor=colors[v],
                                       edgecolor="white", hatch=hatch[v], linewidth=0.2))
                if leaked[name][f, i]:
                    ax.plot(i + 0.5, y + 0.44, "o", ms=1.8, color=plots.INK)
    handles = [Patch(facecolor=colors[k], hatch=hatch[k], edgecolor="white", label=t)
               for k, t in [(0, "Training"), (1, "Validation"), (2, "Test"),
                            (-1, "Not used (copy, gap, purged, held out)")]]
    handles.append(Line2D([], [], marker="o", ls="", color=plots.INK, ms=2.2,
                          label="Evaluation row repeating a training vector"))
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              handlelength=1.0, columnspacing=0.8, fontsize=6, borderaxespad=0.1)
    ax.text(24.0, n_rows + 0.15, "rows in capture order →", ha="right", va="top",
            color=plots.MUTED, fontsize=6)
    ctx.sources.append("toy data; rules of src/iotml/splits.py and src/iotml/experiment.py")
    return fig


# ======================================================================= dataset and audit
def _ordered_labels(counts: pd.DataFrame) -> pd.DataFrame:
    counts = counts.copy()
    counts["fam_rank"] = counts["category"].map({f: i for i, f in enumerate(FAMILY_ORDER)})
    return counts.sort_values(["fam_rank", "rows"], ascending=[True, False]).reset_index(drop=True)


def _family_bands(ax: plt.Axes, table: pd.DataFrame, x_text: float | None,
                  fontsize: float = 8) -> None:
    for fam, grp in table.groupby("category", sort=False):
        lo, hi = grp.index.min(), grp.index.max()
        if lo > 0:
            ax.axhline(lo - 0.5, color=plots.GRID, lw=0.8)
        if x_text is not None:
            ax.text(x_text, (lo + hi) / 2, fam, transform=ax.get_yaxis_transform(),
                    va="center", ha="left", color=plots.MUTED, fontsize=fontsize)


@figure("fig04_class_distribution")
def fig_class_distribution(ctx: Ctx) -> plt.Figure:
    counts = pd.read_csv(ctx.use(project_path("results/audit/class_counts_34.csv")))
    ceil = json.loads(ctx.use(project_path("results/labels/ceilings.json")).read_text(
        encoding="utf-8"))["L34"]["recall_upper_bound"]
    t = _ordered_labels(counts)
    t["bound"] = t["label"].map(ceil)
    y = np.arange(len(t))
    # One column wide (page budget); small type, 34 rows.
    fig, (a, b) = plt.subplots(1, 2, figsize=(plots.SINGLE, 4.6), sharey=True,
                               gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.06})
    a.barh(y, t["rows"], height=0.7, color=plots.PALETTE[0])
    a.set_xscale("log")
    a.set_xlim(500, 2e8)
    a.set_xlabel("Rows (log scale)", fontsize=6.5)
    a.set_yticks(y, t["label"], fontsize=6)
    a.tick_params(axis="x", labelsize=6)
    a.invert_yaxis()
    a.grid(axis="y", visible=False)
    for yi, v in zip(y, t["rows"], strict=True):
        if v < 5000 or v > 4e6:
            text = f"{v / 1e6:.1f} M" if v > 4e6 else f"{v:,}"
            a.text(v * 1.2, yi, text, va="center", color=plots.INK, fontsize=6)
    b.hlines(y, 0, t["bound"], color=plots.GRID, lw=0.6)
    b.plot(t["bound"], y, "o", color=plots.PALETTE[0], ms=2.8)
    b.set_xlim(0.4, 1.03)
    b.set_xticks([0.5, 0.75, 1.0])
    b.tick_params(axis="x", labelsize=6)
    b.set_xlabel("Highest possible recall", fontsize=6.5)
    b.grid(axis="y", visible=False)
    for yi, v in zip(y, t["bound"], strict=True):
        if v < 0.95:
            b.text(v - 0.015, yi, f"{v:.3f}", va="center", ha="right", color=plots.INK,
                   fontsize=6)
    _family_bands(a, t, None)
    _family_bands(b, t, 1.03, fontsize=6)
    return fig


@figure("fig05_audit_waterfall")
def fig_audit_waterfall(ctx: Ctx) -> plt.Figure:
    a = audit(ctx)["duplicates"]
    split = json.loads(ctx.use(project_path("results/splits/random_dedup_s11.json")).read_text(
        encoding="utf-8"))
    rows = a["rows"]
    pairs = rows - split["duplicate_copies_dropped"]
    single = a["unique_vectors"] - a["vectors_multi_label"]
    steps = [("All\nrows", rows, 0, "total"),
             ("Exact\ncopies", rows - pairs, pairs, "drop"),
             ("Distinct\nvector-label\npairs", pairs, 0, "total"),
             ("Pairs of\nmulti-label\nvectors", pairs - single, single, "drop"),
             ("Vectors\nwith one\nlabel", single, 0, "total")]
    fig, ax = plt.subplots(figsize=(plots.SINGLE, 2.1))      # one column (page budget)
    for i, (_name, value, base, kind) in enumerate(steps):
        st = plots.style(0 if kind == "total" else 1)
        ax.bar(i, value / 1e6, bottom=base / 1e6, width=0.6, color=st["color"],
               hatch=st["hatch"], edgecolor="white", linewidth=0.5)
        top = (base + value) / 1e6
        sign = "−" if kind == "drop" else ""
        ax.text(i, top + 0.8, f"{sign}{value / 1e6:.2f} M", ha="center", va="bottom",
                color=plots.INK, fontsize=6)
        if i < len(steps) - 1:
            level = (base + value) / 1e6 if kind == "total" else base / 1e6
            ax.plot([i + 0.3, i + 0.7], [level, level], color=plots.MUTED, lw=0.6, ls=":")
    ax.set_xticks(range(len(steps)), [s[0] for s in steps])
    ax.tick_params(axis="x", labelsize=6)
    ax.tick_params(axis="y", labelsize=6)
    ax.set_ylabel("Rows (millions)", fontsize=6.5)
    ax.set_ylim(0, rows / 1e6 * 1.12)
    ax.grid(axis="x", visible=False)
    return fig


@figure("fig06_ceilings")
def fig_ceilings(ctx: Ctx) -> plt.Figure:
    ceil = json.loads(ctx.use(project_path("results/labels/ceilings.json")).read_text(
        encoding="utf-8"))
    r = ctx.main_runs()
    order = ["L34", "LM", "L8", "L2"]
    fig, ax = plt.subplots(figsize=(plots.SINGLE, 1.9))
    lowest = 100.0
    for yi, s in enumerate(order):
        c = 100 * ceil[s]["accuracy_ceiling"]
        sub = r[(r["protocol"] == "random") & (r["scheme"] == s)]
        ax.plot(c, yi, "D", color=plots.INK, ms=5, zorder=3)
        ax.text(c + 0.8, yi, f"{c:.2f}", ha="left", va="center", color=plots.INK)
        if not sub.empty:
            means = 100 * sub.groupby("model")["accuracy"].mean()
            best = means.idxmax()
            # every model that prints the same value as the best one (a tie is not one winner)
            tied = [MODEL_LABEL[m] for m in means.index
                    if f"{means[m]:.2f}" == f"{means[best]:.2f}"]
            lowest = min(lowest, means[best])
            ax.hlines(yi, means[best], c, color=plots.GRID, lw=2)
            ax.plot(means[best], yi, "o", color=plots.PALETTE[0], ms=5, zorder=3)
            ax.text(means[best] - 0.8, yi, f"{means[best]:.2f} {' and '.join(tied)}",
                    ha="right", va="center", color=plots.INK)
    lo = 5 * np.floor((min(lowest, 89.0) - 15) / 5)
    ax.set_yticks(range(len(order)), [SCHEME_LABEL[s] for s in order])
    ax.set_xticks(np.arange(lo, 100.01, 5))
    ax.set_xlim(lo, 104.5)
    ax.set_ylim(len(order) - 0.5, -0.6)
    ax.set_xlabel("Accuracy on the natural class mix (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Line2D([], [], marker="D", ls="", color=plots.INK, label="Ceiling"),
                       Line2D([], [], marker="o", ls="", color=plots.PALETTE[0],
                              label="Best model, random split")],
              loc="lower left", ncol=2, bbox_to_anchor=(0.0, 1.0), handletextpad=0.3)
    return fig


# ======================================================================= main results
@figure("fig07_main_results")
def fig_main_results(ctx: Ctx) -> plt.Figure:
    r = ctx.main_runs()
    schemes = [s for s in ["L8", "L34", "LM"] if (r["scheme"] == s).any()]
    if not schemes:
        raise MissingInputError("main runs")
    fig, axes = plt.subplots(1, len(schemes), figsize=(plots.DOUBLE, 2.2), sharey=True,
                             squeeze=False)
    agg = r.groupby(["scheme", "protocol", "model"])["macro_f1"].agg(["mean", "std"])
    for ax, s in zip(axes[0], schemes, strict=True):
        for i, m in enumerate(MODELS):
            st = plots.style(i)
            xs, ys, es = [], [], []
            for j, p in enumerate(PROTOCOLS):
                if (s, p, m) in agg.index:
                    xs.append(j + (i - 2) * 0.12)
                    ys.append(agg.loc[(s, p, m), "mean"])
                    es.append(np.nan_to_num(agg.loc[(s, p, m), "std"]))
            ax.errorbar(xs, ys, yerr=es, fmt=st["marker"], color=st["color"], ms=4.5,
                        elinewidth=0.8, capsize=1.5, label=MODEL_LABEL[m])
        ax.set_xticks(range(len(PROTOCOLS)), [PROTOCOL_SHORT[p] for p in PROTOCOLS], fontsize=7)
        ax.set_title(SCHEME_LABEL[s], fontsize=7.5, color=plots.INK)
        ax.grid(axis="x", visible=False)
        ax.set_xlim(-0.5, len(PROTOCOLS) - 0.5)
    axes[0][0].set_ylabel("Macro-F1 (natural class mix)")
    axes[0][0].legend(loc="lower left", bbox_to_anchor=(0, 1.08), ncol=5, handletextpad=0.2,
                      columnspacing=1.0)
    return fig


@figure("fig08_per_class_recall")
def fig_per_class_recall(ctx: Ctx) -> plt.Figure:
    best = ctx.best_model("L34")
    pc = ctx.per_class()
    pc = pc[(pc["scheme"] == "L34") & (pc["model"] == best) & (pc["train_mode"] == "capped")
            & pc["params_mode"].isin(["tuned", "tuned_from_random"])
            & pc["protocol"].isin(PROTOCOLS)]
    counts = _ordered_labels(pd.read_csv(ctx.use(
        project_path("results/audit/class_counts_34.csv"))))
    bound = json.loads(ctx.use(project_path("results/labels/ceilings.json")).read_text(
        encoding="utf-8"))["L34"]["recall_upper_bound"]
    recall = pc.groupby(["class", "protocol"])["recall"].mean().unstack("protocol")
    cols = [p for p in PROTOCOLS if p in recall.columns]
    mat = np.column_stack([counts["label"].map(bound).to_numpy()]
                          + [recall.reindex(counts["label"])[p].to_numpy() for p in cols])
    names = ["Bound"] + [PROTOCOL_SHORT[p] for p in cols]
    fig, ax = plt.subplots(figsize=(plots.SINGLE, 4.7))       # 34 rows at small type
    ax.imshow(mat, cmap=plots.BLUES, vmin=0, vmax=1, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v > 0.6 else plots.INK)
    ax.set_xticks(range(len(names)), names, fontsize=6)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(counts)), counts["label"], fontsize=6)
    ax.grid(False)
    ax.axvline(0.5, color="white", lw=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    for lo in counts.groupby("category", sort=False).apply(lambda g: g.index.min()):
        if lo > 0:
            ax.axhline(lo - 0.5, color="white", lw=1.5)
    ctx.sources.append(f"best L34 model: {best}")
    return fig


@figure("fig09_confusion")
def fig_confusion(ctx: Ctx) -> plt.Figure:
    best = ctx.best_model("L8")
    # All four protocols in one row: the same space as two larger panels (page budget).
    panels = [p for p in PROTOCOLS
              if (ctx.analysis / "confusion" / f"{p}_L8_{best}.csv").exists()]
    if not panels:
        raise MissingInputError(rel(ctx.analysis / "confusion" / f"random_dedup_L8_{best}.csv"))
    width = plots.DOUBLE if len(panels) > 2 else plots.SINGLE * len(panels)
    fig, axes = plt.subplots(1, len(panels), figsize=(width, 2.25), squeeze=False,
                             sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, p in zip(axes[0], panels, strict=True):
        cm = pd.read_csv(ctx.use(ctx.analysis / "confusion" / f"{p}_L8_{best}.csv"),
                         index_col=0)
        ax.imshow(cm.to_numpy(), cmap=plots.BLUES, vmin=0, vmax=1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                v = cm.iat[i, j]
                if v >= 0.005:
                    ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=6,
                            color="white" if v > 0.6 else plots.INK)
        ax.set_xticks(range(cm.shape[1]), cm.columns, rotation=60, ha="right", fontsize=6)
        ax.set_yticks(range(cm.shape[0]), cm.index, fontsize=6)
        ax.set_xlabel("Predicted family", fontsize=6)
        ax.set_title(PROTOCOL_LABEL[p].replace(chr(10), " "), fontsize=6.5, color=plots.INK)
        ax.grid(False)
        ax.tick_params(length=0)
    axes[0][0].set_ylabel("True family", fontsize=6)
    ctx.sources.append(f"best L8 model: {best}")
    return fig


@figure("fig10_imbalance")
def fig_imbalance(ctx: Ctx) -> plt.Figure:
    r = ctx.runs()
    r = r[r["train_mode"].str.startswith("imb_") & (r["protocol"] == "random_dedup")]
    if r.empty:
        raise MissingInputError("imbalance runs (06_imbalance.py)")
    r = r.assign(treatment=r["train_mode"].str.removeprefix("imb_"))
    schemes = [s for s in ["L8", "L34"] if (r["scheme"] == s).any()]
    models = [m for m in MODELS if (r["model"] == m).any()]
    fig, axes = plt.subplots(1, len(schemes), figsize=(plots.SINGLE, 2.3), sharey=True,
                             squeeze=False, gridspec_kw={"wspace": 0.08})
    agg = r.groupby(["scheme", "treatment", "model"])["macro_f1"].agg(["mean", "std"])
    width = 0.8 / len(models)
    for ax, s in zip(axes[0], schemes, strict=True):
        for k, m in enumerate(models):
            st = plots.style(MODELS.index(m))
            for j, t in enumerate(TREATMENTS):
                if (s, t, m) not in agg.index:
                    continue
                mean, sd = agg.loc[(s, t, m)]
                x = j + (k - (len(models) - 1) / 2) * width
                ax.bar(x, mean, width=width * 0.92, color=st["color"], hatch=st["hatch"],
                       edgecolor="white", linewidth=0.5, yerr=np.nan_to_num(sd),
                       error_kw={"elinewidth": 0.8, "capsize": 1.5},
                       label=MODEL_LABEL[m] if j == 0 else None)
                ax.text(x, mean + np.nan_to_num(sd) + 0.01, f"{mean:.3f}", ha="center",
                        va="bottom", rotation=90, color=plots.INK, fontsize=6)
        ax.set_xticks(range(len(TREATMENTS)), [TREATMENT_LABEL[t] for t in TREATMENTS],
                      rotation=35, ha="right", fontsize=6)
        ax.tick_params(axis="y", labelsize=6)
        ax.set_title(SCHEME_LABEL[s], fontsize=6.5, color=plots.INK)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, 1.18)
    axes[0][0].set_ylabel("Macro-F1 (natural mix)", fontsize=6.5)
    axes[0][0].legend(loc="upper left", ncol=len(models), fontsize=6, handlelength=1.0)
    return fig


@figure("fig11_cost")
def fig_cost(ctx: Ctx) -> plt.Figure:
    cost = pd.read_csv(ctx.use(ctx.analysis / "cost.csv"))
    r = ctx.main_runs()
    f1 = r[(r["protocol"] == "random_dedup") & (r["scheme"] == "L8")].groupby("model")[
        "macro_f1"].mean()
    if f1.empty:
        raise MissingInputError("random_dedup L8 runs")
    cost = cost[cost["model"].isin(f1.index)].assign(f1=lambda d: d["model"].map(f1))
    fig, axes = plt.subplots(1, 2, figsize=(plots.SINGLE, 2.1), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    for ax, col, label in [(axes[0], "single_row_median_ms_1thr",
                            "One-row latency, 1 thread (ms)"),
                           (axes[1], "file_mb", "Model file size (MB)")]:
        for _, row in cost.iterrows():
            st = plots.style(MODELS.index(row["model"]))
            ax.plot(row[col], row["f1"], st["marker"], color=st["color"], ms=4)
            ax.annotate(MODEL_LABEL[row["model"]], (row[col], row["f1"]), xytext=(3, 2),
                        textcoords="offset points", color=plots.INK, fontsize=6)
        ax.set_xscale("log")
        ax.set_xlabel(label + ", log scale", fontsize=6)
        ax.tick_params(labelsize=6)
    axes[0].set_ylabel("Macro-F1, deduplicated split", fontsize=6)
    return fig


@figure("fig12_mean_ranks")
def fig_mean_ranks(ctx: Ctx) -> plt.Figure:
    comp = json.loads(ctx.use(ctx.analysis / "model_comparison.json").read_text(
        encoding="utf-8"))["all"]
    ranks = pd.Series(comp["mean_ranks"]).sort_values()
    p_holm = {w["model"]: w["p_holm"] for w in comp["wilcoxon_vs_best"]}
    fig, ax = plt.subplots(figsize=(plots.SINGLE, 1.8))
    y = np.arange(len(ranks))
    ax.hlines(y, 1, ranks.to_numpy(), color=plots.GRID, lw=2)
    for yi, (m, v) in zip(y, ranks.items(), strict=True):
        st = plots.style(MODELS.index(m))
        ax.plot(v, yi, st["marker"], color=st["color"], ms=6)
        note = "best" if m == comp["best"] else f"Holm p = {p_holm[m]:.3f}"
        ax.text(v + 0.2, yi, f"{v:.2f}  ({note})", va="center", color=plots.INK)
    ax.set_yticks(y, [MODEL_LABEL[m] for m in ranks.index])
    ax.invert_yaxis()
    ax.set_xlim(0.8, len(ranks) + 2.3)
    ax.set_xlabel(f"Mean rank over {comp['blocks']} protocol-scheme blocks (1 = best)")
    ax.grid(axis="y", visible=False)
    return fig


@figure("fig13_unseen_type")
def fig_unseen(ctx: Ctx) -> plt.Figure:
    r = ctx.runs()
    r = r[r["protocol"].str.startswith("unseen_type__") & (r["scheme"] == "L8")]
    if r.empty or "held_out" not in r:
        raise MissingInputError("unseen-type runs (05_strict.py --unseen)")
    model = "xgboost" if (r["model"] == "xgboost").any() else r["model"].iloc[0]
    r = r[r["model"] == model]
    counts = pd.read_csv(ctx.use(project_path("results/audit/class_counts_34.csv")))
    fam = dict(zip(counts["label"], counts["category"], strict=True))
    t = r.groupby("held_out")["held_out_recall_own_class"].agg(["mean", "count"]).sort_values(
        "mean")
    # 32 held-out types in the full run: 0.12 in per row keeps the column figure under 4.6 in
    fig, ax = plt.subplots(figsize=(plots.SINGLE, 0.12 * len(t) + 0.7))
    y = np.arange(len(t))
    ax.barh(y, t["mean"], height=0.7, color=plots.PALETTE[0], label="Type held out")
    # reference: the same type when it is in training (deduplicated split, same model and seeds)
    seen_mix = seen_family_mix(ctx.profile, model, sorted(int(s) for s in r["seed"].unique()))
    ctx.use(project_path("data/interim/predictions/random_dedup/L8") / model)
    seen = [seen_mix.loc[lab, fam[lab]] for lab in t.index]
    ax.plot(seen, y, ls="none", marker="|", ms=4.5, mew=1.0, color=plots.INK,
            label="Type in training")
    # the values in one column at the right, clear of bars and markers
    for yi, v in zip(y, t["mean"], strict=True):
        ax.text(1.14, yi, f"{v:.2f}", va="center", ha="right", color=plots.INK, fontsize=6)
    ax.set_yticks(y, [f"{lab} ({fam.get(lab, '?')})" for lab in t.index], fontsize=6)
    ax.tick_params(axis="x", labelsize=6)
    ax.set_xlim(0, 1.15)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(-0.7, len(t) - 0.3)       # no empty band above and below the bars
    ax.set_xlabel(f"Share given to its own family ({MODEL_LABEL[model]})", fontsize=6.5)
    ax.grid(axis="y", visible=False)
    handles, names = ax.get_legend_handles_labels()
    order = [names.index("Type held out"), names.index("Type in training")]
    ax.legend([handles[i] for i in order], [names[i] for i in order], loc="lower center",
              bbox_to_anchor=(0.45, 1.0), ncol=2, fontsize=6, frameon=False, handlelength=1.2,
              columnspacing=1.2, borderaxespad=0.2)
    return fig


@figure("fig14_early_detection")
def fig_early(ctx: Ctx) -> plt.Figure:
    curve = pd.read_csv(ctx.use(project_path(f"results/early_detection/curve_{ctx.profile}.csv")))
    fig, axes = plt.subplots(1, 2, figsize=(plots.SINGLE, 2.0), sharey=True,
                             gridspec_kw={"wspace": 0.3})
    for ax, col, label in [(axes[0], "block_accuracy", "Captures decided correctly"),
                           (axes[1], "macro_family_recall", "Mean recall over families")]:
        for m in [m for m in MODELS if (curve["model"] == m).any()]:
            st = plots.style(MODELS.index(m))
            g = curve[curve["model"] == m].groupby("n_rows")[col].agg(["mean", "std"])
            ax.errorbar(g.index, g["mean"], yerr=np.nan_to_num(g["std"]), marker=st["marker"],
                        color=st["color"], ms=3, capsize=1.2, elinewidth=0.6,
                        label=MODEL_LABEL[m])
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10, 20, 50], ["1", "2", "5", "10", "20", "50"])
        ax.set_xlabel("First $n$ test rows", fontsize=6)
        ax.set_ylabel(label, fontsize=6)
        ax.tick_params(labelsize=6)
        ax.set_ylim(0, 1.02)
    axes[1].yaxis.set_tick_params(labelleft=True)
    axes[0].legend(loc="lower right", fontsize=6, handlelength=1.2)
    return fig


@figure("fig15_calibration")
def fig_calibration(ctx: Ctx) -> plt.Figure:
    cal = pd.read_csv(ctx.use(ctx.analysis / "calibration.csv"))
    protocol = "random_dedup" if (cal["protocol"] == "random_dedup").any() else "random"
    cal = cal[(cal["protocol"] == protocol) & (cal["scheme"] == "L8")]
    if cal.empty:
        raise MissingInputError("calibration rows for L8")
    fig, ax = plt.subplots(figsize=(plots.SINGLE, 2.3))
    ax.plot([0, 1], [0, 1], color=plots.MUTED, lw=0.8, ls="--", label="Perfect calibration")
    for m in [m for m in MODELS if (cal["model"] == m).any()]:
        path = ctx.use(ctx.analysis / "reliability" / f"{protocol}_L8_{m}.csv")
        rt = pd.read_csv(path)
        rt = rt[rt["weight_share"] >= 0.002]
        st = plots.style(MODELS.index(m))
        ece = float(cal.loc[cal["model"] == m, "ece"].iloc[0])
        ax.plot(rt["mean_confidence"], rt["accuracy"], marker=st["marker"], color=st["color"],
                ms=3.5, lw=1.0, label=f"{MODEL_LABEL[m]} (ECE {ece:.3f})")
    ax.set_xlim(0.2, 1.01)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("Mean top-class confidence in bin")
    ax.set_ylabel("Accuracy in bin")
    ax.legend(loc="upper left", handlelength=1.5)
    return fig


@figure("fig16_shap")
def fig_shap(ctx: Ctx) -> plt.Figure:
    s = pd.read_csv(ctx.use(ctx.analysis / "shap_overall.csv"), index_col=0).iloc[:, 0]
    s = s.sort_values(ascending=False).head(12)[::-1]
    fig, ax = plt.subplots(figsize=(plots.SINGLE, 2.6))
    ax.barh(range(len(s)), s.to_numpy(), height=0.65, color=plots.PALETTE[0])
    ax.set_yticks(range(len(s)), s.index)
    ax.set_xlabel("Mean |SHAP value| over classes (XGBoost, 8 families)")
    ax.grid(axis="y", visible=False)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", choices=["quick", "full"], required=True)
    parser.add_argument("--only", nargs="+")
    args = parser.parse_args()
    log = get_logger(f"09_figures_{args.profile}", project_path("logs"))
    out = project_path("paper/figures" if args.profile == "full" else "paper/figures_quick")
    analysis = project_path("results/analysis" if args.profile == "full"
                            else "results/analysis_quick")
    manifest = []
    shared: dict[str, pd.DataFrame | None] = {"runs": None, "per_class": None}
    for name, fn in FIGURES.items():
        if args.only and name not in args.only:
            continue
        ctx = Ctx(args.profile, out, analysis, shared["runs"], shared["per_class"])
        try:
            fig = fn(ctx)
        except MissingInputError as err:
            plt.close("all")
            log.info("skip %s: missing %s", name, err)
            manifest.append({"figure": name, "status": f"skipped, missing {err}",
                             "sources": ""})
            continue
        except Exception as err:   # keep going; the manifest records the failure
            plt.close("all")
            log.exception("FAILED %s", name)
            manifest.append({"figure": name, "status": f"FAILED: {err!r}", "sources": ""})
            continue
        finally:
            shared["runs"], shared["per_class"] = ctx._runs, ctx._per_class
        plots.save(fig, out, name)
        log.info("drew %s", name)
        manifest.append({"figure": name, "status": "drawn",
                         "sources": "; ".join(dict.fromkeys(ctx.sources))})
    pd.DataFrame(manifest).to_csv(out / "figures_manifest.csv", index=False)


if __name__ == "__main__":
    main()

"""Step 9: every table of the paper, built from saved results.

Usage: python scripts/09_tables.py --profile quick|full [--only NAME ...]

Writes tables/<name>.csv (all digits) and paper/tables/<name>.tex (a booktabs tabular that
paper/main.tex includes; captions stay in main.tex). The quick profile writes to
tables/quick/ and paper/tables_quick/. A table whose inputs do not exist yet is skipped with
a note, as in 09_figures.py.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from iotml.config import project_path, read_yaml
from iotml.logs import get_logger
from iotml.results import load_runs

MODELS = ["decision_tree", "random_forest", "extra_trees", "xgboost", "lightgbm"]
MODEL_LABEL = {"decision_tree": "DT", "random_forest": "RF", "extra_trees": "ET",
               "xgboost": "XGBoost", "lightgbm": "LightGBM", "mlp_ce": "MLP (cross-entropy)",
               "mlp_focal": "MLP (focal loss)", "cnn1d_ce": "1D-CNN on windows",
               "gru_ce": "GRU on windows", "gru_focal": "GRU on windows (focal loss)",
               "xgboost_last_row": "XGBoost, last row of window",
               "xgboost_window_stats": "XGBoost, window statistics"}
PROTOCOLS = ["random", "random_dedup", "file_grouped", "order_within_file"]
PROTOCOL_LABEL = {"random": "Random", "random_dedup": "Random, dedup.",
                  "file_grouped": "File-grouped", "order_within_file": "Ordered in file",
                  "order_windows": "Ordered in file (windows)"}
SCHEME_LABEL = {"L34": "34 labels", "L8": "8 families", "LM": "30 merged labels",
                "L2": "Benign vs attack"}
TREATMENT_LABEL = {"none": "None", "weights": "Class weights", "under": "Undersampling",
                   "smotetomek": "SMOTE-Tomek"}


class MissingInputError(Exception):
    pass


def need(path: Path) -> Path:
    if not path.exists():
        raise MissingInputError(str(path.relative_to(project_path("."))).replace("\\", "/"))
    return path


def load_json(path: Path) -> dict:
    return json.loads(need(path).read_text(encoding="utf-8"))


# ------------------------------------------------------------------ LaTeX helpers
def esc(text: object) -> str:
    s = str(text)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#"),
                 ("_", r"\_"), ("$", r"\$"), ("~", r"\textasciitilde{}"), ("^", r"\^{}")]:
        s = s.replace(a, b)
    return s


def prose(text: object) -> str:
    """esc() plus typeset symbols that the evidence table writes in plain text."""
    s = esc(text).replace("+/-", r"$\pm$").replace("us/sample", r"$\mu$s per sample")
    return re.sub(r"(\d) x (\d)", r"\1 $\\times$ \2", s)


def num(n: float) -> str:
    """Integer with LaTeX thousands separators."""
    return f"{int(round(n)):,}".replace(",", "{,}")


def pm(mean: float, sd: float, digits: int = 3, bold: bool = False) -> str:
    body = f"{mean:.{digits}f}"
    if sd is not None and np.isfinite(sd):
        body += f" \\pm {sd:.{digits}f}"
    return f"$\\mathbf{{{body}}}$" if bold else f"${body}$"


def ceiling_pct(share: float) -> str:
    """A ceiling in percent; a value below 100 is never printed as 100.00."""
    return "$>99.99$" if 0.99995 <= share < 1 else f"${100 * share:.2f}$"


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(need(path))
    except pd.errors.EmptyDataError as err:
        raise MissingInputError(f"rows in {path.name}") from err


def tabular(colspec: str, header: list[str], rows: list[list[str]],
            groups: dict[int, str] | None = None) -> str:
    """booktabs tabular; `groups` maps a row index to a full-width group heading."""
    ncol = len(header)
    out = [f"\\begin{{tabular}}{{{colspec}}}", r"\toprule", " & ".join(header) + r" \\",
           r"\midrule"]
    for i, row in enumerate(rows):
        if groups and i in groups:
            if i > 0:
                out.append(r"\midrule")
            out.append(f"\\multicolumn{{{ncol}}}{{l}}{{\\textit{{{groups[i]}}}}} \\\\")
        out.append(" & ".join(row) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(out)


class Ctx:
    def __init__(self, profile: str) -> None:
        self.profile = profile
        self.analysis = project_path("results/analysis" if profile == "full"
                                     else "results/analysis_quick")
        self._runs: pd.DataFrame | None = None

    def runs(self) -> pd.DataFrame:
        if self._runs is None:
            self._runs = load_runs(self.profile)
        if self._runs.empty:
            raise MissingInputError(f"results/experiments/**/*_{self.profile}.json")
        return self._runs

    def main_runs(self) -> pd.DataFrame:
        r = self.runs()
        return r[(r["train_mode"] == "capped") & r["params_mode"].isin(
            ["tuned", "tuned_from_random"]) & r["protocol"].isin(PROTOCOLS)
            & r["model"].isin(MODELS)]


TABLES: dict[str, Callable[[Ctx], tuple[pd.DataFrame, str]]] = {}


def table(name: str) -> Callable:
    def register(fn: Callable) -> Callable:
        TABLES[name] = fn
        return fn
    return register


# ------------------------------------------------------------------ dataset and audit
@table("tab01_dataset")
def tab_dataset(ctx: Ctx) -> tuple[pd.DataFrame, str]:
    a = load_json(project_path("results/audit/audit_summary.json"))
    c, p, s = a["counts"], a["profile"], a["schema"]
    drop = read_yaml(project_path("config/models.yaml"))["features"]["drop"]
    rows = [
        ("Capture files (one attack or benign capture each)", num(c["source_files"])),
        ("Rows (flow-window records)", num(c["rows"])),
        ("Labels / families", f"{c['labels']} / {c['categories']} (benign counted in both)"),
        ("Benign rows", f"{num(c['benign_rows'])} "
                        f"({100 * c['benign_rows'] / c['rows']:.2f}\\%)"),
        ("Largest label", f"{esc(c['largest_label'])} ({num(c['largest_label_rows'])})"),
        ("Smallest label", f"{esc(c['smallest_label'])} ({num(c['smallest_label_rows'])})"),
        ("Largest-to-smallest ratio", num(c["imbalance_ratio"])),
        ("Labels recorded in one capture file only", num(c["labels_with_one_file"])),
        ("Feature columns in this release",
         f"{s['our_features']} ({s['shared']} of the paper's {s['paper_features_excl_timestamp']}"
         f" plus {', '.join(s['ours_only'])})"),
        ("Features used for learning",
         f"{s['our_features'] - len(drop)} ({esc(' and '.join(drop))} dropped as exact copies)"),
        ("Missing cells",
         f"{num(p['missing_cells'])} ({esc(', '.join(p['columns_with_missing']))})"),
        ("Infinite cells", f"{num(p['inf_cells'])} ({esc(', '.join(p['columns_with_inf']))})"),
        ("Time stamp column", "present" if s["timestamp_column_present"] else "absent"),
    ]
    df = pd.DataFrame(rows, columns=["quantity", "value"])
    return df, tabular("p{0.44\\linewidth}p{0.45\\linewidth}", ["Quantity", "Value"],
                       [[esc(q), v] for q, v in rows])


@table("tab02_audit")
def tab_audit(ctx: Ctx) -> tuple[pd.DataFrame, str]:
    d = load_json(project_path("results/audit/audit_summary.json"))["duplicates"]
    n = d["rows"]
    items = [
        ("Rows identical to an earlier row", None, d["duplicate_rows"]),
        ("Distinct feature vectors", d["unique_vectors"], None),
        ("Vectors carrying more than one label", d["vectors_multi_label"],
         d["rows_multi_label"]),
        ("Vectors carrying more than one family", d["vectors_multi_category"],
         d["rows_multi_category"]),
        ("Vectors carrying both benign and attack labels", d["vectors_multi_binary"],
         d["rows_multi_binary"]),
        ("Vectors found in more than one capture file", d["vectors_multi_file"],
         d["rows_multi_file"]),
    ]
    df = pd.DataFrame([{"finding": f, "vectors": v, "rows": r,
                        "row_share": None if r is None else r / n} for f, v, r in items])
    rows = [[esc(f), "" if v is None else num(v), "" if r is None else num(r),
             "" if r is None else f"{100 * r / n:.1f}"] for f, v, r in items]
    return df, tabular("lrrr", ["Finding", "Vectors", "Rows", r"Rows (\%)"], rows)


@table("tab03_schemes")
def tab_schemes(ctx: Ctx) -> tuple[pd.DataFrame, str]:
    c = load_json(project_path("results/labels/ceilings.json"))
    m = load_json(project_path("results/labels/merge_sensitivity.json"))
    groups = "; ".join(" + ".join(g) for g in m["groups"])
    built = {"L34": "Original attack labels plus benign",
             "LM": "L34 with the three conflict groups merged",
             "L8": "Seven attack families plus benign",
             "L2": "Benign versus attack",
             "L7 (analysis only: DDoS and DoS as one family)": "L8 with DDoS and DoS joined"}
    rows, recs = [], []
    for key in ["L34", "LM", "L8", "L7 (analysis only: DDoS and DoS as one family)", "L2"]:
        v = c[key]
        name = "L7$^{*}$" if key.startswith("L7") else key
        recs.append({"scheme": key, "classes": v["classes"],
                     "accuracy_ceiling": v["accuracy_ceiling"],
                     "macro_recall_bound": v["macro_recall_upper_bound"]})
        rows.append([name, str(v["classes"]), esc(built[key]),
                     f"{100 * v['accuracy_ceiling']:.2f}",
                     f"{100 * v['macro_recall_upper_bound']:.2f}"])
    df = pd.DataFrame(recs)
    df.attrs["merged_groups"] = groups
    # one column wide in the paper (page budget): short headings, wrapped description
    return df, tabular("lrp{0.33\\linewidth}rr",
                       ["Scheme", "Classes", "Built from", r"Ceiling (\%)",
                        r"Recall bound (\%)"], rows)


@table("tab04_protocols")
def tab_protocols(ctx: Ctx) -> tuple[pd.DataFrame, str]:
    rules = {"random": "Rows drawn at random, stratified by label",
             "random_dedup": "Identical vectors kept in one part; repeated copies removed",
             "file_grouped": "Whole capture files per part (labels with 3 or more files); "
                             "ordered blocks with gaps otherwise; purged",
             "order_within_file": "Earlier rows of every capture train, later rows test, "
                                  "with gaps; purged"}
    recs, rows = [], []
    for p in PROTOCOLS:
        found = sorted(project_path("results/splits").glob(f"{p}_s*.json"),
                       key=lambda f: int(f.stem.rsplit("_s", 1)[1]))
        if not found:
            continue
        s = json.loads(found[0].read_text(encoding="utf-8"))
        r = s["rows"]
        purged = sum(s.get("purged_rows_seen_in_train", {}).values()) if isinstance(
            s.get("purged_rows_seen_in_train"), dict) else 0
        recs.append({"protocol": p, "seed": s["seed"], **r, "purged": purged})
        rows.append([PROTOCOL_LABEL[p], esc(rules[p]), num(r["train"]), num(r["val"]),
                     num(r["test"]), num(r["unused"])])
    if not recs:
        raise MissingInputError("results/splits/*.json")
    return pd.DataFrame(recs), tabular(
        "lp{0.34\\linewidth}rrrr",
        ["Protocol", "Rule", "Training", "Validation", "Test", "Not used"], rows)


@table("tab05_hyperparameters")
def tab_hyper(ctx: Ctx) -> tuple[pd.DataFrame, str]:
    cfg = read_yaml(project_path("config/models.yaml"))["models"]
    recs, rows = [], []
    for m in MODELS:
        space = cfg[m]["search"]
        tuned = project_path(f"results/experiments/random/tuning/{m}_capped_{ctx.profile}.json")
        best = json.loads(tuned.read_text(encoding="utf-8"))["best_params"] \
            if tuned.exists() else {}
        space_txt = "; ".join(f"{k}: {', '.join('none' if v is None else str(v) for v in vals)}"
                              for k, vals in space.items() if len(vals) > 1)
        best_txt = "; ".join(f"{k} {'none' if v is None else v}" for k, v in best.items()
                             if len(space.get(k, [])) > 1) or "not yet tuned"
        recs.append({"model": m, "search_space": space_txt, "chosen": best_txt})
        rows.append([MODEL_LABEL[m], esc(space_txt), esc(best_txt)])
    return pd.DataFrame(recs), tabular("lp{0.5\\linewidth}p{0.3\\linewidth}",
                                       ["Model", "Search space", "Chosen setting"], rows)


# ------------------------------------------------------------------ results
PROTOCOL_HEAD = {"random": "Random", "random_dedup": "Dedup.", "file_grouped": "File",
                 "order_within_file": "Ordered"}


@table("tab06_main_results")
def tab_main_results(ctx: Ctx) -> tuple[pd.DataFrame, str]:
    """Macro-F1 and accuracy (%) side by side for every protocol: one page-wide table instead
    of two column tables (page budget). Bold: every value that prints the same as the best of
    its column and label scheme."""
    r = ctx.main_runs()
    metrics = [("macro_f1", 1.0, 3), ("accuracy", 100.0, 2)]
    aggs = {m: r.groupby(["scheme", "protocol", "model"])[m].agg(["mean", "std", "count"])
            for m, _, _ in metrics}
    first = aggs["macro_f1"]
    schemes = [s for s in ["L8", "L34", "LM", "L2"] if s in first.index.get_level_values(0)]
    protocols = [p for p in PROTOCOLS if p in first.index.get_level_values(1)]
    ncol = 1 + len(metrics) * len(protocols)
    ceil_path = project_path("results/splits/test_ceilings"
                             + ("" if ctx.profile == "full" else f"_{ctx.profile}") + ".csv")
    ceil = (pd.read_csv(ceil_path).groupby(["scheme", "protocol"])["accuracy_ceiling"].mean()
            if ceil_path.exists() else None)
    out = [f"\\begin{{tabular}}{{l{'c' * (ncol - 1)}}}", r"\toprule",
           " & " + " & ".join(f"\\multicolumn{{{len(protocols)}}}{{c}}{{{head}}}"
                              for head in ["Macro-F1", "Accuracy (\\%)"]) + r" \\",
           "".join(f"\\cmidrule(lr){{{2 + k * len(protocols)}-{1 + (k + 1) * len(protocols)}}}"
                   for k in range(len(metrics))),
           "Model & " + " & ".join(PROTOCOL_HEAD[p] for _ in metrics for p in protocols)
           + r" \\", r"\midrule"]
    for si, s in enumerate(schemes):
        if si:
            out.append(r"\midrule")
        out.append(f"\\multicolumn{{{ncol}}}{{l}}{{\\textit{{{SCHEME_LABEL[s]}}}}} \\\\")
        for m in MODELS:
            cells = [MODEL_LABEL[m]]
            for metric, scale, digits in metrics:
                agg = aggs[metric]
                for p in protocols:
                    if (s, p, m) in agg.index:
                        best = round(scale * agg.loc[(s, p)]["mean"].max(), digits)
                        mean, sd, _ = agg.loc[(s, p, m)]
                        cells.append(pm(scale * mean, scale * sd, digits=digits,
                                        bold=round(scale * mean, digits) == best))
                    else:
                        cells.append("--")
            out.append(" & ".join(cells) + r" \\")
        if ceil is not None and s in ceil.index.get_level_values(0):
            # Label-conflict ceiling of the rows each protocol tests on (05_split_ceilings.py):
            # an accuracy bound, so the macro-F1 columns stay empty.
            cells = ["Ceiling"] + ["--"] * len(protocols)
            tested = set(first.loc[s].index.get_level_values(0))
            cells += [ceiling_pct(ceil.loc[(s, p)]) if (s, p) in ceil.index and p in tested
                      else "--" for p in protocols]
            out.append(" & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", ""]
    seeds = int(first["count"].max())
    data = pd.concat({m: a for m, a in aggs.items()}, names=["metric"]).reset_index()
    return data, (f"% mean +/- standard deviation over up to {seeds} seeds\n" + "\n".join(out))


# As-published cells whose fit failed (D-025: training loss diverged, 04_diagnose_lightgbm.py).
FAILED_FIT = {("L34", "lightgbm")}


@table("tab15_random_split")
def tab_random_split(ctx: Ctx) -> tuple[pd.DataFrame, str]:
    """Random split: as published (natural mix, defaults) next to capped and tuned."""
    r = ctx.runs()
    rnd = r[(r["protocol"] == "random") & r["model"].isin(MODELS)]
    settings = {
        "published": rnd[(rnd["train_mode"] == "natural") & (rnd["params_mode"] == "default")],
        "careful": rnd[(rnd["train_mode"] == "capped")
                       & rnd["params_mode"].isin(["tuned", "tuned_from_random"])],
    }
    if settings["published"].empty:
        raise MissingInputError("as-published runs (04_baseline.py)")
    metrics = [("accuracy", 100.0, 2), ("macro_f1", 1.0, 3)]
    rows: list[list[str]] = []
    groups: dict[int, str] = {}
    recs: list[dict] = []
    seeds: dict[str, int] = {}
    for s in [s for s in ["L8", "L34"] if (settings["published"]["scheme"] == s).any()]:
        groups[len(rows)] = SCHEME_LABEL[s]
        agg = {k: d[d["scheme"] == s].groupby("model")[[m for m, _, _ in metrics]]
               .agg(["mean", "std", "count"]) for k, d in settings.items()}
        for m in MODELS:
            cells = [MODEL_LABEL[m]]
            for k, g in agg.items():
                for met, scale, digits in metrics:
                    if m not in g.index:
                        cells.append("--")
                        continue
                    mean, sd, n = (g.loc[m, (met, stat)] for stat in ("mean", "std", "count"))
                    # bold the best value as printed, so that printed ties are all bold
                    best = round(mean * scale, digits) == round(g[(met, "mean")].max() * scale,
                                                                digits)
                    cell = pm(mean * scale, sd * scale, digits, bold=best)
                    if k == "published" and (s, m) in FAILED_FIT:
                        cell = cell[:-1] + r"^{\dagger}$"
                    cells.append(cell)
                    seeds[k] = max(seeds.get(k, 0), int(n))
                    recs.append({"scheme": s, "setting": k, "model": m, "metric": met,
                                 "mean": mean, "std": sd, "seeds": int(n)})
            rows.append(cells)
    tex = tabular("lcccc", ["Model", r"Accuracy (\%)", "Macro-F1", r"Accuracy (\%)", "Macro-F1"],
                  rows, groups)
    top = (r" & \multicolumn{2}{c}{As published} & \multicolumn{2}{c}{Capped, tuned} \\"
           "\n" r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}" "\n")
    tex = tex.replace("\\toprule\n", "\\toprule\n" + top, 1)
    tex = (f"% mean +/- standard deviation; as published over {seeds.get('published', 0)} "
           f"seeds, capped and tuned over {seeds.get('careful', 0)}\n" + tex)
    return pd.DataFrame(recs), tex


@table("tab08_bootstrap")
def tab_bootstrap(ctx: Ctx) -> tuple[pd.DataFrame, str]:
    b = read_csv(ctx.analysis / "bootstrap_ci.csv")
    r = ctx.main_runs()
    rows, recs = [], []
    for s in [s for s in ["L8", "L34", "LM", "L2"] if (b["scheme"] == s).any()]:
        for p in PROTOCOLS:
            sub = b[(b["scheme"] == s) & (b["protocol"] == p)]
            if sub.empty:
                continue
            best = r[(r["scheme"] == s) & (r["protocol"] == p)].groupby("model")[
                "macro_f1"].mean().idxmax()
            if not (sub["model"] == best).any():
                continue                      # interval of the best model not computed yet
            row = sub[sub["model"] == best].iloc[0]
            recs.append(row.to_dict())
            # short scheme and protocol names (defined in the paper): one column wide
            rows.append([s, PROTOCOL_HEAD[p], MODEL_LABEL[best],
                         f"{row['macro_f1']:.3f}", f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]",
                         f"{row['ci_high'] - row['ci_low']:.3f}", str(int(row["groups"]))])
    return pd.DataFrame(recs), tabular("lllcccr", ["Scheme", "Protocol", "Best model",
                                                   "Macro-F1", "95\\% interval", "Width",
                                                   "Files"], rows)


@table("tab09_imbalance")
def tab_imbalance(ctx: Ctx) -> tuple[pd.DataFrame, str]:
    r = ctx.runs()
    r = r[r["train_mode"].str.startswith("imb_")]
    if r.empty:
        raise MissingInputError("imbalance runs (06_imbalance.py)")
    r = r.assign(treatment=r["train_mode"].str.removeprefix("imb_"))
    agg = r.groupby(["scheme", "model", "treatment"])[
        ["macro_f1", "macro_precision", "balanced_accuracy", "fit_seconds"]].agg(["mean", "std"])
    rows, groups = [], {}
    for s in [s for s in ["L8", "L34"] if s in agg.index.get_level_values(0)]:
        groups[len(rows)] = SCHEME_LABEL[s]
        for m in [m for m in MODELS if (s, m) in agg.index.droplevel(2)]:
            for t in TREATMENT_LABEL:
                if (s, m, t) not in agg.index:
                    continue
                v = agg.loc[(s, m, t)]
                rows.append([MODEL_LABEL[m], TREATMENT_LABEL[t],
                             pm(v[("macro_f1", "mean")], v[("macro_f1", "std")]),
                             pm(v[("macro_precision", "mean")], v[("macro_precision", "std")]),
                             pm(v[("balanced_accuracy", "mean")], v[("balanced_accuracy", "std")]),
                             f"{v[('fit_seconds', 'mean')]:.0f}"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index(), tabular("llcccr", ["Model", "Treatment", "Macro-F1",
                                                 "Macro precision", "Macro recall",
                                                 "Fit (s)"], rows, groups)


@table("tab10_neural")
def tab_neural(ctx: Ctx) -> tuple[pd.DataFrame, str]:
    r = ctx.runs()
    neural = r[r["model"].str.startswith(("mlp_", "cnn1d", "gru", "xgboost_"))]
    if neural.empty:
        raise MissingInputError("neural runs (06_neural.py)")
    ref = ctx.main_runs()
    ref = ref[(ref["model"] == "xgboost") & (ref["scheme"] == "L8")
              & ref["protocol"].isin(neural["protocol"].unique())]
    both = pd.concat([neural[neural["scheme"] == "L8"], ref])
    agg = both.groupby(["protocol", "model"])[["macro_f1", "accuracy", "fit_seconds"]].agg(
        ["mean", "std"])
    rows, groups = [], {}
    order = ["random_dedup", "file_grouped", "order_within_file", "order_windows"]
    for p in [p for p in order if p in agg.index.get_level_values(0)]:
        groups[len(rows)] = PROTOCOL_LABEL[p]
        for m in agg.loc[p].index:
            v = agg.loc[(p, m)]
            rows.append([esc(MODEL_LABEL.get(m, m)),
                         pm(v[("macro_f1", "mean")], v[("macro_f1", "std")]),
                         pm(v[("accuracy", "mean")], v[("accuracy", "std")]),
                         f"{v[('fit_seconds', 'mean')]:.0f}"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index(), tabular("lccr", ["Model (8 families)", "Macro-F1", "Accuracy",
                                               "Fit (s)"], rows, groups)


@table("tab11_cost")
def tab_cost(ctx: Ctx) -> tuple[pd.DataFrame, str]:
    c = read_csv(ctx.analysis / "cost.csv")
    thr = [col for col in c.columns if col.startswith("single_row_median_ms_")]
    many = max(thr, key=lambda s: int(s.rsplit("_", 1)[1].removesuffix("thr")))
    n_many = many.rsplit("_", 1)[1].removesuffix("thr")
    rows = [[MODEL_LABEL[r["model"]], f"{r['fit_seconds']:.0f}", f"{r['file_mb']:.1f}",
             f"{r['single_row_median_ms_1thr']:.2f}", f"{r['single_row_p95_ms_1thr']:.2f}",
             f"{r['batch_us_per_row_1thr']:.1f}", f"{r[f'batch_us_per_row_{n_many}thr']:.1f}"]
            for _, r in c.iterrows()]
    note = f"% machine: {c['machine'].iloc[0]}; {c['threads_note'].iloc[0]}\n"
    return c, note + tabular("lrrrrrr", ["Model", "Fit (s)", "File (MB)",
                                         "1 row, median (ms)", "1 row, p95 (ms)",
                                         r"Batch, 1 thr. ($\mu$s/row)",
                                         f"Batch, {n_many} thr. ($\\mu$s/row)"], rows)


@table("tab12_stats")
def tab_stats(ctx: Ctx) -> tuple[pd.DataFrame, str]:
    comp = load_json(ctx.analysis / "model_comparison.json")
    recs, rows = [], []
    for key, label in [("L8", "8 families"), ("L34", "34 labels"), ("LM", "30 merged labels"),
                       ("all", "All blocks")]:
        if key not in comp:
            continue
        c = comp[key]
        fr = (f"$\\chi^2 = {c['friedman_chi2']:.2f}$, $p = {c['friedman_p']:.3g}$"
              if "friedman_p" in c else "too few blocks")
        holm = ", ".join(f"{MODEL_LABEL[w['model']]} {w['p_holm']:.3g}"
                         for w in c["wilcoxon_vs_best"])
        recs.append({"group": key, "blocks": c["blocks"], "best": c["best"],
                     "friedman_chi2": c.get("friedman_chi2"), "friedman_p": c.get("friedman_p"),
                     "holm": holm})
        rows.append([label, str(c["blocks"]), MODEL_LABEL[c["best"]], fr, esc(holm)])
    return pd.DataFrame(recs), tabular("lrlp{0.22\\linewidth}p{0.27\\linewidth}",
                                       ["Group", "Blocks", "Best (mean rank)", "Friedman test",
                                        "Holm-adjusted p vs best"], rows)


@table("tab13_unseen")
def tab_unseen(ctx: Ctx) -> tuple[pd.DataFrame, str]:
    r = ctx.runs()
    r = r[r["protocol"].str.startswith("unseen_type__") & (r["scheme"] == "L8")]
    if r.empty:
        raise MissingInputError("unseen-type runs (05_strict.py --unseen)")
    # one model, the same as Figure 13 and the \val{unseen-*} keys (XGBoost in the paper)
    model = "xgboost" if (r["model"] == "xgboost").any() else r["model"].iloc[0]
    r = r[r["model"] == model]
    recs, mixes = [], {}
    for _, row in r.iterrows():
        u = json.loads(Path(row["path"]).read_text(encoding="utf-8"))["metrics"]["unseen_type"]
        recs.append({"model": row["model"], "seed": row["seed"], "held_out": u["held_out"],
                     "family": u["own_class"], "test_rows": u["test_rows"],
                     "own_family_share": u["recall_own_class"]})
        mixes.setdefault(u["held_out"], []).append(pd.Series(u["predicted_as"], dtype=float))
    df = pd.DataFrame(recs)
    g = df.groupby(["held_out", "family"], as_index=False).agg(
        own=("own_family_share", "mean"), rows=("test_rows", "mean")).sort_values("own")
    rows = []
    for x in g.itertuples():
        # mean share per predicted family over seeds, then the largest family other than its own
        mix = pd.concat(mixes[x.held_out], axis=1).fillna(0.0).mean(axis=1).drop(
            x.family, errors="ignore")
        other = (f"{esc(mix.idxmax())} ({mix.max():.3f})" if len(mix) and mix.max() > 0
                 else "none")
        rows.append([esc(x.held_out), esc(x.family), num(x.rows), f"{x.own:.3f}", other])
    # one column wide in the paper (page budget): short headings
    return df, tabular("llrcl", ["Held-out type", "Family", "Test rows", "Own family",
                                 "Top other family"], rows)


@table("tab14_literature")
def tab_literature(ctx: Ctx) -> tuple[pd.DataFrame, str]:
    e = pd.read_csv(need(project_path("refs/evidence_table.csv")))
    keep = e["theme"].isin(["CICIoT2023 reported results", "CICIoT2023 careful evaluation"]) \
        & e["reported_result"].notna()
    e = e[keep].sort_values(["theme", "year"])
    rows = [[f"\\cite{{{x.bibkey}}}", str(x.year),
             prose(x.task if isinstance(x.task, str) else "n/s"),
             prose(x.split_protocol if isinstance(x.split_protocol, str) else "not stated"),
             prose(x.reported_result)] for x in e.itertuples()]
    # full page width with fewer wrapped lines (page budget)
    return e, tabular("llp{0.14\\linewidth}p{0.27\\linewidth}p{0.40\\linewidth}",
                      ["Study", "Year", "Task", "Split", "Reported result"], rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", choices=["quick", "full"], required=True)
    parser.add_argument("--only", nargs="+")
    args = parser.parse_args()
    log = get_logger(f"09_tables_{args.profile}", project_path("logs"))
    csv_dir = project_path("tables" if args.profile == "full" else "tables/quick")
    tex_dir = project_path("paper/tables" if args.profile == "full" else "paper/tables_quick")
    csv_dir.mkdir(parents=True, exist_ok=True)
    tex_dir.mkdir(parents=True, exist_ok=True)
    ctx = Ctx(args.profile)
    status = []
    for name, fn in TABLES.items():
        if args.only and name not in args.only:
            continue
        try:
            df, tex = fn(ctx)
        except MissingInputError as err:
            log.info("skip %s: missing %s", name, err)
            status.append({"table": name, "status": f"skipped, missing {err}"})
            continue
        except Exception as err:   # keep going; the manifest records the failure
            log.exception("FAILED %s", name)
            status.append({"table": name, "status": f"FAILED: {err!r}"})
            continue
        df.to_csv(csv_dir / f"{name}.csv", index=False)
        (tex_dir / f"{name}.tex").write_text(
            f"% generated by scripts/09_tables.py --profile {args.profile}; do not edit\n" + tex,
            encoding="utf-8")
        log.info("wrote %s (%d rows)", name, len(df))
        status.append({"table": name, "status": "written"})
    manifest = tex_dir / "tables_manifest.csv"
    table = pd.DataFrame(status)
    if args.only and manifest.exists():      # keep the rows of the tables not rebuilt now
        old = pd.read_csv(manifest)
        table = pd.concat([old[~old["table"].isin(table["table"])], table])
        rank = {n: i for i, n in enumerate(TABLES)}
        table = table.sort_values("table", key=lambda c: c.map(rank))
    table.to_csv(manifest, index=False)


if __name__ == "__main__":
    main()

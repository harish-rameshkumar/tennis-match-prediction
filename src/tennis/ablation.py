"""Feature-group ablation (owner: Jayant).

Groups follow the column names in the README. For one model (default: the
paper's logistic regression) we report, on the test years:
  * all        : every feature
  * drop_<g>   : everything except group g
  * only_<g>   : group g alone
  * add_<g>    : groups added cumulatively in ADD_ORDER
Each row has match-level accuracy (with a 95% binomial interval), log-loss and
Brier. With ~2,600 test matches the interval is about +/-1.9 points, so a
difference smaller than that is noise: say so in the write-up.

Run:  python -m tennis.ablation
      python -m tennis.ablation --model hgb --test-years 2023 2024
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tennis.dataset import get_train_test
from tennis.evaluate import match_metrics, score_matches
from tennis.models import MODEL_NAMES, build_model, feature_columns, get_xy

# First matching rule wins, so more specific patterns come first
# (diff_surface_elo must be Elo, not surface form).
GROUP_RULES = [
    ("elo",            r"^diff_(surface_)?elo$"),
    ("surface_form",   r"^diff_surface_"),
    ("head_to_head",   r"^(diff_h2h_|h2h_)"),
    ("common_opp",     r"common_opp"),
    ("serve_stats",    r"^(has_serve_stats_|diff_(first|second|aces|dfs|bp|sv)_)"),
    ("tournament",     r"^diff_tourney_"),
    ("form",           r"^(has_history_|diff_(career|last\d+|recent30)_)"),
    ("rank_attrs",     r"^(rank|rank_points|height|hand|age)_[ab]$|missing"),
]
ADD_ORDER = ["rank_attrs", "form", "surface_form", "tournament", "head_to_head",
             "common_opp", "serve_stats", "elo"]


def assign_groups(columns) -> dict[str, list[str]]:
    groups = {g: [] for g, _ in GROUP_RULES}
    unassigned = []
    for c in columns:
        for g, pat in GROUP_RULES:
            if re.search(pat, c):
                groups[g].append(c)
                break
        else:
            unassigned.append(c)
    if unassigned:
        raise ValueError(f"columns not in any group (update GROUP_RULES): {unassigned}")
    return {g: cols for g, cols in groups.items() if cols}


def _eval(name, features, train, test, Xtr_full=None):
    Xtr, ytr = get_xy(train, features)
    Xte, _ = get_xy(test, features)
    model = build_model(name, Xtr).fit(Xtr, ytr)
    m = match_metrics(score_matches(model, test, Xte)["p_win"])
    se = (m["accuracy"] * (1 - m["accuracy"]) / m["n_matches"]) ** 0.5
    return {"n_features": len(features), "accuracy": m["accuracy"],
            "acc_ci95": 1.96 * se, "log_loss": m["log_loss"], "brier": m["brier"]}


def run_ablation(model="logreg", test_years=None, out_dir="results", fig_dir="reports/figures"):
    train, test = get_train_test(*test_years) if test_years else get_train_test()
    tag = "-".join(map(str, test_years)) if test_years else "2019"
    cols = feature_columns()
    groups = assign_groups(cols)
    print("groups:", {g: len(c) for g, c in groups.items()})

    rows = []

    def add(kind, group, feats):
        r = _eval(model, feats, train, test)
        rows.append({"model": model, "test_year": tag, "kind": kind, "group": group, **r})
        print(f"{kind:9s} {group:13s} n={r['n_features']:3d} acc {r['accuracy']:.3%} "
              f"(+/-{r['acc_ci95']:.1%}) log-loss {r['log_loss']:.4f}")

    add("all", "all", cols)
    for g, gc in groups.items():
        add("drop", g, [c for c in cols if c not in gc])
    for g, gc in groups.items():
        add("only", g, gc)
    acc = []
    for g in [g for g in ADD_ORDER if g in groups]:
        acc += groups[g]
        add("add", g, list(acc))

    df = pd.DataFrame(rows)
    full = df.loc[df["kind"] == "all", "accuracy"].iloc[0]
    full_ll = df.loc[df["kind"] == "all", "log_loss"].iloc[0]
    df["acc_vs_all"] = df["accuracy"] - full
    df["logloss_vs_all"] = df["log_loss"] - full_ll
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"ablation_{model}_{tag}.csv", index=False)
    _plot(df, Path(fig_dir) / f"ablation_{model}_{tag}.png", model, tag)
    return df


def _plot(df, path, model, tag):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    full = df.loc[df["kind"] == "all", "accuracy"].iloc[0]
    for ax, kind, title in ((axes[0], "drop", "all features minus group"),
                            (axes[1], "only", "group alone")):
        d = df[df["kind"] == kind].sort_values("accuracy")
        ax.barh(d["group"], d["accuracy"] * 100, xerr=d["acc_ci95"] * 100, color="#4a7")
        ax.axvline(full * 100, color="k", ls="--", lw=1, label="all features")
        ax.set_title(title)
        ax.set_xlabel("match accuracy (%)")
        lo = max(0, (d["accuracy"].min() - d["acc_ci95"].max()) * 100 - 1)
        ax.set_xlim(lo, (d["accuracy"].max() + d["acc_ci95"].max()) * 100 + 1)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Feature-group ablation ({model}, test {tag})")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="logreg", choices=[m for m in MODEL_NAMES if m != "rank_baseline"])
    ap.add_argument("--test-years", nargs=2, type=int, default=None, metavar=("START", "END"))
    a = ap.parse_args()
    run_ablation(a.model, a.test_years)

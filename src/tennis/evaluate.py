"""Match-level evaluation (owner: Jayant).

Every match has two rows. The model scores both, and inference.combine()
merges them into ONE probability that the match winner wins:
    p = (p_ab + (1 - p_ba)) / 2
Accuracy, log-loss and Brier are then computed per MATCH, not per row.

Run:  python -m tennis.evaluate --models rank_baseline logreg rf linear_svc hgb
      python -m tennis.evaluate --test-years 2023 2024 --tune
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix

from tennis.dataset import get_train_test
from tennis.inference import combine
from tennis.models import GRIDS, MODEL_NAMES, build_model, get_xy

EXTRA_COLS = ["match_date", "year", "tournament", "round", "surface", "player_a_id",
              "player_b_id", "player_a", "player_b", "player_a_name", "player_b_name"]
EPS = 1e-15


# --------------------------------------------------------------------------
# Row level -> match level
# --------------------------------------------------------------------------
def score_matches(model, df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """One row per match with p_win = P(actual winner wins), from combine().

    The label-1 row is the winner-as-A row, the label-0 row is the mirrored
    one. combine() is symmetric, so this ordering leaks nothing: swapping the
    rows just maps p -> 1 - p, and we always read p from the winner's side.
    """
    d = df[["match_id", "label", "rank_a", "rank_b"] +
           [c for c in EXTRA_COLS if c in df.columns]].copy()
    d["p_row"] = model.predict_proba(X)[:, 1]          # P(row's player A wins)
    d = d.sort_values(["match_id", "label"], ascending=[True, False], kind="mergesort")
    if len(d) % 2:
        raise ValueError("odd number of rows: every match needs both mirrored rows")
    win, lose = d.iloc[0::2], d.iloc[1::2]
    assert (win["match_id"].to_numpy() == lose["match_id"].to_numpy()).all()
    assert (win["label"] == 1).all() and (lose["label"] == 0).all()
    out = win.drop(columns=["p_row", "label"]).reset_index(drop=True)
    out["p_ab"] = win["p_row"].to_numpy()                # row (winner, loser)
    out["p_ba"] = lose["p_row"].to_numpy()               # row (loser, winner)
    out["p_win"] = np.asarray(combine(out["p_ab"].to_numpy(), out["p_ba"].to_numpy()))
    # rank_a/rank_b come from the winner-as-A row: A = winner, B = loser
    out["upset"] = out["rank_a"] > out["rank_b"]          # worse-ranked player won
    out["rank_gap"] = (out["rank_a"] - out["rank_b"]).abs()
    return out


def match_metrics(p_win, probabilistic: bool = True) -> dict:
    """Metrics from P(winner wins). A 0.5 tie counts as half a correct call.

    Log-loss/Brier are identical whichever player we call "the positive
    class", so the winner-oriented form is exact.
    """
    p_win = np.asarray(p_win, float)
    tie = np.isclose(p_win, 0.5, rtol=0, atol=1e-9)
    acc = float(np.mean(((p_win > 0.5) & ~tie) + 0.5 * tie))
    out = {"n_matches": int(len(p_win)), "accuracy": acc}
    if probabilistic:
        p = np.clip(p_win, EPS, 1 - EPS)
        out["log_loss"] = float(-np.mean(np.log(p)))
        out["brier"] = float(np.mean((1 - p_win) ** 2))
    else:  # hard 0/1 predictions: log-loss is meaningless
        out["log_loss"] = float("nan")
        out["brier"] = float(np.mean((1 - p_win) ** 2))
    return out


def confusion(p_win) -> np.ndarray:
    """Both orientations of each match: (p, truth=1) and (1-p, truth=0).
    Symmetric by construction ([[c, w], [w, c]]); shown for completeness."""
    p_win = np.asarray(p_win, float)
    y = np.r_[np.ones_like(p_win), np.zeros_like(p_win)]
    pred = np.r_[p_win > 0.5, (1 - p_win) > 0.5]
    return confusion_matrix(y, pred, labels=[0, 1])


# --------------------------------------------------------------------------
# Tables / plots
# --------------------------------------------------------------------------
GAP_BINS = [0, 10, 25, 50, 100, 200, np.inf]


def rank_gap_table(matches: pd.DataFrame) -> pd.DataFrame:
    g = pd.cut(matches["rank_gap"], GAP_BINS, right=False)
    rows = []
    for k, grp in matches.groupby(g, observed=True):
        m = match_metrics(grp["p_win"])
        rows.append({"rank_gap": str(k), "n": m["n_matches"], "accuracy": m["accuracy"],
                     "upset_rate": float(grp["upset"].mean())})
    return pd.DataFrame(rows)


def worst_misses(matches: pd.DataFrame, k: int = 15) -> pd.DataFrame:
    """Matches where the model was most confident in the wrong player."""
    return matches.nsmallest(k, "p_win")


def calibration_plot(curves: dict, path, title: str):
    """Reliability diagram. Each match is used in both orientations (p, 1) and
    (1-p, 0) so the curve is symmetric around 0.5."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for name, p_win in curves.items():
        p = np.r_[p_win, 1 - p_win]
        y = np.r_[np.ones_like(p_win), np.zeros_like(p_win)]
        frac, mean_p = calibration_curve(y, p, n_bins=10, strategy="uniform")
        ax.plot(mean_p, frac, marker="o", ms=3, label=name)
    ax.set_xlabel("predicted P(player wins)")
    ax.set_ylabel("observed win frequency")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def model_explanations(name, model, X, y):
    """Logistic regression coefficients / tree-model importances."""
    if name == "logreg":
        names = model.named_steps["prep"].get_feature_names_out()
        coef = model.named_steps["clf"].coef_[0]
        df = pd.DataFrame({"feature": names, "coef_per_std": coef})
        return df.reindex(df["coef_per_std"].abs().sort_values(ascending=False).index)
    if name == "rf":
        names = model.named_steps["prep"].get_feature_names_out()
        imp = model.named_steps["clf"].feature_importances_
        return pd.DataFrame({"feature": names, "importance": imp}
                            ).sort_values("importance", ascending=False)
    if name == "hgb":  # no built-in importances -> permutation on a test sample
        from sklearn.inspection import permutation_importance
        idx = np.random.default_rng(0).choice(len(X), min(4000, len(X)), replace=False)
        r = permutation_importance(model, X.iloc[idx], y[idx], n_repeats=3,
                                   scoring="neg_log_loss", random_state=0, n_jobs=1)
        return pd.DataFrame({"feature": X.columns, "importance": r.importances_mean}
                            ).sort_values("importance", ascending=False)
    return None


# --------------------------------------------------------------------------
# Hyperparameter tuning (time-based validation INSIDE the training years)
# --------------------------------------------------------------------------
def tune_model(name: str, train: pd.DataFrame, val_year: int | None = None) -> dict:
    """Fit on years < val_year, score match-level log-loss on val_year only.
    val_year defaults to the last training year (e.g. train<=2017, validate 2018).
    The test set is never touched, and both rows of a match are always in the
    same split because we split by year."""
    if name not in GRIDS:
        return {}
    val_year = int(val_year or train["year"].max())
    fit_df, val_df = train[train["year"] < val_year], train[train["year"] == val_year]
    Xf, yf = get_xy(fit_df)
    Xv, _ = get_xy(val_df)
    keys = list(GRIDS[name])
    best, best_ll = None, np.inf
    for vals in itertools.product(*GRIDS[name].values()):
        params = dict(zip(keys, vals))
        m = build_model(name, Xf, **params).fit(Xf, yf)
        ll = match_metrics(score_matches(m, val_df, Xv)["p_win"])["log_loss"]
        print(f"   tune {name} {params} -> val log-loss {ll:.4f}")
        if ll < best_ll:
            best, best_ll = params, ll
    print(f"   best {name}: {best} (val {val_year} log-loss {best_ll:.4f})")
    return best


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def run(test_years=None, models=MODEL_NAMES, tune=False, out_dir="results",
        fig_dir="reports/figures", save_models=False):
    train, test = get_train_test(*test_years) if test_years else get_train_test()
    tag = "-".join(map(str, test_years)) if test_years else "2019"
    Xtr, ytr = get_xy(train)
    Xte, yte = get_xy(test)
    out_dir, fig_dir = Path(out_dir), Path(fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{tag}] train {len(train) // 2:,} matches | test {len(test) // 2:,} matches")

    table, curves, summary = [], {}, {}
    for name in models:
        print(f"== {name}")
        params = tune_model(name, train) if tune else {}
        model = build_model(name, Xtr, **params).fit(Xtr, ytr)
        prob = name != "rank_baseline"
        res = {}
        for split, df, X in (("train", train, Xtr), ("test", test, Xte)):
            mt = score_matches(model, df, X)
            res[split] = match_metrics(mt["p_win"], prob)
            if split == "test":
                test_matches = mt
        t, tr = res["test"], res["train"]
        table.append({"model": name, "test_year": tag, "n_test_matches": t["n_matches"],
                      "train_accuracy": tr["accuracy"], "test_accuracy": t["accuracy"],
                      "test_log_loss": t["log_loss"], "test_brier": t["brier"],
                      "params": json.dumps(params)})
        print(f"   train acc {tr['accuracy']:.3%} | test acc {t['accuracy']:.3%} | "
              f"log-loss {t['log_loss']:.4f} | brier {t['brier']:.4f}")
        if t["accuracy"] > 0.75:
            print("   !! accuracy > 75%: SUSPECT LEAKAGE - tell Harish before going on")
        if prob:
            curves[name] = test_matches["p_win"].to_numpy()
        summary[name] = {"confusion_matrix": confusion(test_matches["p_win"]).tolist()}
        rank_gap_table(test_matches).to_csv(out_dir / f"rank_gap_{name}_{tag}.csv", index=False)
        worst_misses(test_matches).to_csv(out_dir / f"worst_misses_{name}_{tag}.csv", index=False)
        ex = model_explanations(name, model, Xte, yte)
        if ex is not None:
            ex.to_csv(out_dir / f"explain_{name}_{tag}.csv", index=False)
        if save_models and name != "rank_baseline":
            import joblib
            Path("models").mkdir(exist_ok=True)
            joblib.dump(model, f"models/{name}.joblib")

    res_df = pd.DataFrame(table)
    res_df.to_csv(out_dir / f"metrics_{tag}.csv", index=False)
    (out_dir / f"metrics_{tag}.json").write_text(json.dumps(summary | {"table": table}, indent=2))
    if curves:
        calibration_plot(curves, fig_dir / f"calibration_{tag}.png", f"Calibration, test {tag}")
    print("\n" + res_df.drop(columns="params").to_string(index=False))
    return res_df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    ap.add_argument("--test-years", nargs=2, type=int, default=None, metavar=("START", "END"))
    ap.add_argument("--tune", action="store_true", help="time-based validation tuning")
    ap.add_argument("--save-models", action="store_true", help="write models/*.joblib")
    a = ap.parse_args()
    run(a.test_years, a.models, a.tune, save_models=a.save_models)

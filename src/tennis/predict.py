"""Single-match prediction (owner: Jayant).

    from tennis.predict import predict_match
    r = predict_match("Novak Djokovic", "Carlos Alcaraz", "Grass", "2025-07-01",
                      tournament="Wimbledon")
    r["p_a"], r["p_b"], r["top_features"]

Feature rows come from inference.build_match_rows() and the two mirrored rows
are merged with inference.combine(); no feature logic is duplicated here.

Models are NOT in git (models/*.joblib is ignored), so a fresh clone must train
first (about a minute for logistic regression):

    python -m tennis.predict --train                      # logreg, trained on years <= 2022
    python -m tennis.predict --train --models logreg hgb

Errors from build_match_rows (UnknownPlayerError with .suggestions,
AmbiguousPlayerError with .candidates, missing rank, ...) are passed through
unchanged so the CLI / Streamlit app can show them.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from tennis.dataset import get_train_test, train_medians
from tennis.inference import History, build_match_rows, combine, load_history
from tennis.models import MODEL_NAMES, build_model, get_xy

MODEL_DIR = Path("models")
DEFAULT_MODEL = "logreg"
DEFAULT_TEST_YEARS = (2023, 2024)       # model trains on every year before 2023

_BUNDLES: dict = {}
_HIST: dict = {}


# --------------------------------------------------------------------------
# Training / loading
# --------------------------------------------------------------------------
def _path(name: str) -> Path:
    return MODEL_DIR / f"predict_{name}.joblib"


def train_and_save(names=(DEFAULT_MODEL,), test_years=DEFAULT_TEST_YEARS):
    """Fit models on the training split and save them together with the fill
    medians (inference must fill missing values exactly like training did)."""
    train, _ = get_train_test(*test_years)
    X, y = get_xy(train)
    medians = train_medians(train)
    MODEL_DIR.mkdir(exist_ok=True)
    for n in names:
        model = build_model(n, X).fit(X, y)
        joblib.dump({"name": n, "model": model, "medians": medians,
                     "columns": list(X.columns),
                     "trained_before_year": int(train["year"].max()) + 1}, _path(n))
        print(f"saved {_path(n)}  (trained on years < {int(train['year'].max()) + 1})")


def load_bundle(name: str = DEFAULT_MODEL) -> dict:
    if name not in _BUNDLES:
        p = _path(name)
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Train it first:  python -m tennis.predict --train")
        _BUNDLES[name] = joblib.load(p)
    return _BUNDLES[name]


def _history():
    """Load the match history once (a few seconds), then reuse it."""
    if not _HIST:
        matches = load_history()
        _HIST["matches"], _HIST["full"] = matches, History(matches)
    return _HIST["matches"], _HIST["full"]


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
def _display_name(rows: pd.DataFrame, side: str, fallback: str) -> str:
    for col in (f"player_{side}_name", f"player_{side}"):
        if col in rows.columns:
            return str(rows[col].iloc[0])
    return str(fallback)


def _logreg_contributions(model, X: pd.DataFrame, rows: pd.DataFrame, top_k: int):
    """coefficient x standardized value, in log-odds, from player A's side.

    Row 0 is (A,B) and row 1 is (B,A); a feature that favours B in row 1 favours
    A's opponent, so A's net contribution is 0.5 * (row0 - row1). Columns that
    are identical in both rows (h2h_played, ...) cancel to 0 by construction.
    """
    prep, clf = model.named_steps["prep"], model.named_steps["clf"]
    z = np.asarray(prep.transform(X)) * clf.coef_[0]
    contrib = 0.5 * (z[0] - z[1])
    names = prep.get_feature_names_out()
    out = []
    for i in np.argsort(-np.abs(contrib))[:top_k]:
        raw = rows[names[i]].iloc[0] if names[i] in rows.columns else None
        out.append({"feature": str(names[i]), "contribution": float(contrib[i]),
                    "favours": "A" if contrib[i] > 0 else "B",
                    "value": float(raw) if isinstance(raw, (int, float, np.number)) else None})
    return out


def predict_match(player_a, player_b, surface, date, *, tournament=None,
                  attributes=None, model: str = DEFAULT_MODEL, top_k: int = 8) -> dict:
    """Win probability for each player, plus the top contributing features.

    Returns a dict:
        player_a, player_b : resolved display names
        p_a, p_b           : combined win probabilities (p_a + p_b == 1)
        p_ab, p_ba         : the model's raw outputs for the two mirrored rows
        top_features       : list of {feature, contribution, favours, value}
                             (logistic regression only; empty for other models)
        model, surface, date
    `attributes` overrides rank / points / height / hand / age, e.g.
    {"rank_a": 2, "rank_points_b": 9000}.
    """
    bundle = load_bundle(model)
    date = str(pd.Timestamp(date).date())
    kw = dict(tournament=tournament, attributes=attributes, medians=bundle["medians"])
    matches, full = _history()
    try:                       # fine for dates after the data ends
        rows = build_match_rows(player_a, player_b, surface, date, full, **kw)
    except ValueError:         # date inside the data: use only matches before it
        rows = build_match_rows(player_a, player_b, surface, date, matches, **kw)

    X = rows[bundle["columns"]]
    p = bundle["model"].predict_proba(X)[:, 1]
    p_ab, p_ba = float(p[0]), float(p[1])
    p_a = float(combine(p_ab, p_ba))
    top = (_logreg_contributions(bundle["model"], X, rows, top_k)
           if bundle["name"] == "logreg" else [])
    return {"player_a": _display_name(rows, "a", player_a),
            "player_b": _display_name(rows, "b", player_b),
            "p_a": p_a, "p_b": 1.0 - p_a, "p_ab": p_ab, "p_ba": p_ba,
            "top_features": top, "model": bundle["name"],
            "surface": surface, "date": date}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train/save the demo models.")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--models", nargs="+", default=[DEFAULT_MODEL],
                    choices=[m for m in MODEL_NAMES if m != "rank_baseline"])
    ap.add_argument("--test-years", nargs=2, type=int, default=list(DEFAULT_TEST_YEARS))
    a = ap.parse_args()
    if a.train:
        train_and_save(a.models, tuple(a.test_years))
    else:
        ap.print_help()

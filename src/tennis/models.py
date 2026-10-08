"""Models for match-winner prediction (owner: Jayant).

Every model is an sklearn estimator with predict_proba, trained on the
*pairs* dataset: each match appears twice, as (A, B) with label 1 if A won and
as the mirrored row (B, A). Combining the two rows into one match-level
probability happens in evaluate.py via inference.combine().

ASSUMPTIONS about Harish's code (change the two imports below if the names differ):
  * FEATURE_COLUMNS lives in tennis.features or tennis.dataset
  * the dataframe from get_train_test() has columns match_id, label, rank_a, rank_b
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

SEED = 0
SURFACE_COL = "surface"
MODEL_NAMES = ("rank_baseline", "logreg", "rf", "linear_svc", "hgb")


# --------------------------------------------------------------------------
# Data helpers
# --------------------------------------------------------------------------
def feature_columns() -> list[str]:
    """Use the team's FEATURE_COLUMNS list; never hardcode feature names."""
     for mod in ("tennis.dataset", "tennis.features"):
        try:
            return list(getattr(__import__(mod, fromlist=["FEATURE_COLUMNS"]),
                                "FEATURE_COLUMNS"))
        except (ImportError, AttributeError):
            continue
    raise ImportError("FEATURE_COLUMNS not found in tennis.features or "
                      "tennis.dataset - check the README and fix feature_columns().")


def get_xy(df: pd.DataFrame, features=None, use_surface: bool = True):
    """X (DataFrame indexed by match_id), y (0/1).

    The match_id index is how grouped CV keeps the two mirrored rows of one
    match in the same fold. It is NOT a feature.
    """
    cols = list(features if features is not None else feature_columns())
    if use_surface and SURFACE_COL in df.columns and SURFACE_COL not in cols:
        cols.append(SURFACE_COL)
    X = df[cols].copy()
    X.index = df["match_id"].to_numpy()
    return X, df["label"].to_numpy()


def make_preprocessor(X: pd.DataFrame, scale: bool) -> ColumnTransformer:
    """Scale numeric columns (fit on train only, because it sits in a Pipeline)
    and one-hot encode text columns such as surface."""
    cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num = [c for c in X.columns if c not in cat]
    num_tf = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [("num", num_tf, num),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)],
        verbose_feature_names_out=False)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
class RankBaseline(BaseEstimator, ClassifierMixin):
    """The better-ranked player (smaller rank number) always wins."""

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = np.where(X["rank_a"].to_numpy() < X["rank_b"].to_numpy(), 1.0,
                     np.where(X["rank_a"].to_numpy() > X["rank_b"].to_numpy(), 0.0, 0.5))
        return np.column_stack([1 - p, p])


class GroupCalibratedSVC(BaseEstimator, ClassifierMixin):
    """LinearSVC + sigmoid (Platt) calibration.

    LinearSVC has no predict_proba, so we calibrate it. The calibration folds
    are grouped by match_id (taken from X.index): if the two mirrored rows of a
    match landed in different folds, calibration would be fit on near-copies of
    its own training data.
    """

    def __init__(self, preprocessor=None, C=0.1, n_splits=3):
        self.preprocessor = preprocessor
        self.C = C
        self.n_splits = n_splits

    def fit(self, X, y):
        base = Pipeline([("prep", self.preprocessor),
                         ("clf", LinearSVC(C=self.C, dual=False, max_iter=5000,
                                           random_state=SEED))])
        groups = np.asarray(X.index)
        splits = list(GroupKFold(self.n_splits).split(X, y, groups))
        self.model_ = CalibratedClassifierCV(base, method="sigmoid", cv=splits).fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


DEFAULTS = {
    "logreg": dict(C=1.0),
    "rf": dict(n_estimators=300, min_samples_leaf=10, max_features="sqrt"),
    "linear_svc": dict(C=0.1),
    "hgb": dict(learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
                max_iter=300),
}

# Small grids so tuning finishes in minutes. Tuned on a time-based validation
# split INSIDE the training years (see evaluate.tune_model), never on test.
GRIDS = {
    "logreg": {"C": [0.01, 0.1, 1.0, 10.0]},
    "rf": {"min_samples_leaf": [5, 20, 50], "max_features": ["sqrt", 0.5]},
    "linear_svc": {"C": [0.01, 0.1, 1.0]},
    "hgb": {"learning_rate": [0.03, 0.1], "max_leaf_nodes": [8, 15, 31],
            "l2_regularization": [0.0, 1.0]},
}


def build_model(name: str, X: pd.DataFrame, **params):
    """Unfitted model. X is only used to learn column names/dtypes."""
    p = {**DEFAULTS.get(name, {}), **params}
    if name == "rank_baseline":
        return RankBaseline()
    if name == "logreg":   # the paper's setup: liblinear + L2
        return Pipeline([("prep", make_preprocessor(X, scale=True)),
                         ("clf", LogisticRegression(solver="liblinear", max_iter=1000,
                                                    random_state=SEED, **p))])
    if name == "rf":       # trees do not need scaling
        return Pipeline([("prep", make_preprocessor(X, scale=False)),
                         ("clf", RandomForestClassifier(n_jobs=-1, random_state=SEED, **p))])
    if name == "linear_svc":
        return GroupCalibratedSVC(preprocessor=make_preprocessor(X, scale=True), **p)
    if name == "hgb":      # early_stopping=False: its internal random validation
        return Pipeline([("prep", make_preprocessor(X, scale=False)),   # split would
                         ("clf", HistGradientBoostingClassifier(        # separate mirrored
                             early_stopping=False, random_state=SEED, **p))])  # rows
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")

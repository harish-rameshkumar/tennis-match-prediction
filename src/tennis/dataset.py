"""Build the mirrored (player A, player B) pairs table and split it into train/test.

Each cleaned match becomes two rows: (A=winner, B=loser, label=1) and the swapped
(A=loser, B=winner, label=0). See README "Dataset interface" for the column contract.

    python -m tennis.dataset    # build data/processed/pairs.parquet and print a summary
"""

import numpy as np
import pandas as pd

from tennis import config
from tennis.clean import clean
from tennis.features import DIFF_FEATURES, FLAG_FEATURES, PAIR_FEATURES, compute_features
from tennis.load import load_raw

ID_COLUMNS = ["match_id", "match_date", "year", "tour_level", "surface", "label",
              "player_a_id", "player_b_id"]
META_COLUMNS = ["tourney_name", "round", "player_a_name", "player_b_name"]

# Per-player attributes: winner_/loser_ column in the raw data -> name in the pairs table.
PLAYER_ATTRIBUTES = {
    "id": "player_{}_id",
    "name": "player_{}_name",
    "rank": "rank_{}",
    "rank_points": "rank_points_{}",
    "ht": "height_{}",
    "hand": "hand_{}",
    "age": "age_{}",
}
BASE_FEATURES = ["rank", "rank_points", "height", "hand", "age"]
# rank is never missing after cleaning, so it gets no indicator.
IMPUTED_FEATURES = ["rank_points", "height", "hand", "age"]

# History-based features from tennis.features: diff_<f> = A - B, flags as <f>_a / <f>_b.
HISTORY_COLUMNS = (
    [f"diff_{f}" for f in DIFF_FEATURES]
    + [f"{f}_{s}" for f in FLAG_FEATURES for s in "ab"]
    + PAIR_FEATURES
)

FEATURE_COLUMNS = (
    [f"{f}_{s}" for f in BASE_FEATURES for s in "ab"]
    + [f"{f}_{s}_missing" for f in IMPUTED_FEATURES for s in "ab"]
    + HISTORY_COLUMNS
)

MAIN_TOUR_LEVELS = ["G", "M", "A", "F"]


def _side(matches: pd.DataFrame, prefix: str, side: str) -> pd.DataFrame:
    return pd.DataFrame({
        target.format(side): matches[f"{prefix}_{attr}"]
        for attr, target in PLAYER_ATTRIBUTES.items()
    })


def build_pairs(matches: pd.DataFrame, feats: pd.DataFrame | None = None) -> pd.DataFrame:
    """Two rows per match: winner as A (label 1), then loser as A (label 0).

    Feature columns keep NaN where data is missing; *_missing columns flag those cells.
    Imputation happens in get_train_test so that it only uses training-set statistics.
    hand is encoded 1.0 = left-handed, 0.0 = right-handed.
    feats: output of tennis.features.compute_features for `matches` (computed if omitted).
    """
    common = matches[["match_id", "match_date", "year", "tour_level", "surface",
                      "tourney_name", "round"]].reset_index(drop=True)
    won = pd.concat([common, _side(matches, "winner", "a"), _side(matches, "loser", "b")],
                    axis=1).assign(label=1)
    lost = pd.concat([common, _side(matches, "loser", "a"), _side(matches, "winner", "b")],
                     axis=1).assign(label=0)
    # Interleave so both rows of a match are adjacent and the chronological order is kept.
    won.index = won.index * 2
    lost.index = lost.index * 2 + 1
    pairs = pd.concat([won, lost]).sort_index().reset_index(drop=True)

    for s in "ab":
        pairs[f"hand_{s}"] = pairs[f"hand_{s}"].map({"L": 1.0, "R": 0.0})
        pairs[f"rank_{s}"] = pairs[f"rank_{s}"].astype(int)
    for f in IMPUTED_FEATURES:
        for s in "ab":
            pairs[f"{f}_{s}_missing"] = pairs[f"{f}_{s}"].isna().astype(int)

    if feats is None:
        feats = compute_features(matches)
    pairs = pd.concat([pairs, _history_columns(feats, len(pairs))], axis=1)
    return pairs[ID_COLUMNS + META_COLUMNS + FEATURE_COLUMNS]


def _history_columns(feats: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    """Mirror tennis.features output onto the interleaved (won, lost) rows."""
    out = {}
    for f in DIFF_FEATURES:
        diff = feats[f"winner_{f}"].to_numpy() - feats[f"loser_{f}"].to_numpy()
        col = np.empty(n_rows)
        col[0::2], col[1::2] = diff, -diff
        out[f"diff_{f}"] = col
    for f in FLAG_FEATURES:
        w, l = feats[f"winner_{f}"].to_numpy(int), feats[f"loser_{f}"].to_numpy(int)
        a, b = np.empty(n_rows, int), np.empty(n_rows, int)
        a[0::2], a[1::2] = w, l
        b[0::2], b[1::2] = l, w
        out[f"{f}_a"], out[f"{f}_b"] = a, b
    for f in PAIR_FEATURES:
        out[f] = np.repeat(feats[f].to_numpy(), 2)
    return pd.DataFrame(out)


def build(path=config.PAIRS_PATH) -> pd.DataFrame:
    pairs = build_pairs(clean(load_raw()))
    path.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(path, index=False)
    return pairs


def load_pairs(path=config.PAIRS_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m tennis.dataset` first")
    return pd.read_parquet(path)


def train_medians(train: pd.DataFrame) -> dict[str, float]:
    """Median of each imputed feature, pooled over its _a and _b columns.

    Works on raw or already-filled training rows: filling with the median keeps the median.
    """
    return {f: train[[f"{f}_a", f"{f}_b"]].stack().median() for f in IMPUTED_FEATURES}


def fill_missing(df: pd.DataFrame, medians: dict[str, float]) -> pd.DataFrame:
    """Fill NaNs in the imputed features with `medians` (from train_medians), in place."""
    for f, median in medians.items():
        cols = [f"{f}_a", f"{f}_b"]
        df[cols] = df[cols].fillna(median)
    return df


def get_train_test(test_year: int = config.DEFAULT_TEST_YEAR, test_end_year: int | None = None,
                   pairs: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological train/test split of the pairs table.

    train: every pair (all tour levels, incl. Challenger and qualifying) with year < test_year.
    test:  ATP main-tour pairs (tour_level in MAIN_TOUR_LEVELS) with
           test_year <= year <= test_end_year (defaults to test_year).
    Both rows of a match share match_id and year, so a match is never split. Rows stay
    in chronological order and feature columns contain no NaN.
    """
    if pairs is None:
        pairs = load_pairs()
    test_end_year = test_year if test_end_year is None else test_end_year

    train = pairs[pairs["year"] < test_year].copy()
    test = pairs[pairs["year"].between(test_year, test_end_year)
                 & pairs["tour_level"].isin(MAIN_TOUR_LEVELS)].copy()
    medians = train_medians(train)
    fill_missing(train, medians)
    fill_missing(test, medians)
    return train.reset_index(drop=True), test.reset_index(drop=True)


if __name__ == "__main__":
    pairs = build()
    print(f"wrote {config.PAIRS_PATH}: {pairs.shape[0]} rows x {pairs.shape[1]} columns "
          f"({pairs['match_id'].nunique()} matches)")

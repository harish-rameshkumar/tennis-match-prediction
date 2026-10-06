import numpy as np
import pytest

from tennis import config
from tennis.dataset import FEATURE_COLUMNS, HISTORY_COLUMNS, get_train_test, load_pairs

pytestmark = pytest.mark.skipif(not config.PAIRS_PATH.exists(),
                                reason="run `python -m tennis.dataset` to build pairs.parquet")

DIFF_COLUMNS = [c for c in FEATURE_COLUMNS if c.startswith("diff_")]
A_COLUMNS = [c for c in FEATURE_COLUMNS if c.endswith("_a") or "_a_" in c]
B_COLUMNS = [c for c in FEATURE_COLUMNS if c.endswith("_b") or "_b_" in c]
SYMMETRIC_COLUMNS = [c for c in FEATURE_COLUMNS
                     if c not in DIFF_COLUMNS + A_COLUMNS + B_COLUMNS]


@pytest.fixture(scope="module")
def pairs():
    return load_pairs()


def _won_lost(df):
    won = df[df["label"] == 1].set_index("match_id").sort_index()
    lost = df[df["label"] == 0].set_index("match_id").sort_index()
    assert won.index.equals(lost.index)
    return won, lost


def _check(df):
    won, lost = _won_lost(df)
    for c in DIFF_COLUMNS:
        assert np.array_equal(won[c].to_numpy(), -lost[c].to_numpy(), equal_nan=True), c
    for a in A_COLUMNS:
        b = a[:-2] + "_b" if a.endswith("_a") else a.replace("_a_", "_b_")
        assert np.array_equal(won[a].to_numpy(), lost[b].to_numpy(), equal_nan=True), a
    for c in SYMMETRIC_COLUMNS:
        assert np.array_equal(won[c].to_numpy(), lost[c].to_numpy(), equal_nan=True), c


def test_every_feature_has_a_symmetry_rule():
    assert DIFF_COLUMNS
    assert len(A_COLUMNS) == len(B_COLUMNS)
    assert set(DIFF_COLUMNS + A_COLUMNS + B_COLUMNS + SYMMETRIC_COLUMNS) == set(FEATURE_COLUMNS)


def test_history_columns_present_and_finite(pairs):
    assert set(HISTORY_COLUMNS) <= set(pairs.columns)
    assert np.isfinite(pairs[HISTORY_COLUMNS].to_numpy(float)).all()


def test_mirrored_rows_are_exact_mirrors(pairs):
    _check(pairs)


def test_mirrors_survive_train_test_imputation(pairs):
    train, test = get_train_test(pairs=pairs)
    _check(train)
    _check(test)

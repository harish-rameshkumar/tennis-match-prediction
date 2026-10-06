import pandas as pd
import pytest

from tennis import config
from tennis.dataset import FEATURE_COLUMNS, ID_COLUMNS, MAIN_TOUR_LEVELS, get_train_test, load_pairs

pytestmark = pytest.mark.skipif(not config.PAIRS_PATH.exists(),
                                reason="run `python -m tennis.dataset` to build pairs.parquet")


@pytest.fixture(scope="module")
def pairs():
    return load_pairs()


@pytest.fixture(scope="module", params=[(2019, None), (2023, 2024)], ids=["2019", "2023-24"])
def split(request, pairs):
    test_year, test_end_year = request.param
    return test_year, test_end_year, *get_train_test(test_year, test_end_year, pairs=pairs)


def test_required_columns(pairs):
    assert set(ID_COLUMNS + FEATURE_COLUMNS) <= set(pairs.columns)


def test_every_match_appears_exactly_twice(pairs):
    assert (pairs["match_id"].value_counts() == 2).all()


def test_labels_balanced_one_win_one_loss_per_match(pairs):
    assert set(pairs["label"].unique()) == {0, 1}
    assert pairs["label"].mean() == 0.5
    assert (pairs.groupby("match_id")["label"].agg(sorted).map(tuple) == (0, 1)).all()


def test_mirrored_rows_swap_players(pairs):
    won = pairs[pairs["label"] == 1].set_index("match_id")
    lost = pairs[pairs["label"] == 0].set_index("match_id").loc[won.index]
    for col in ["player_{}_id", "rank_{}", "height_{}", "age_{}"]:
        pd.testing.assert_series_equal(won[col.format("a")], lost[col.format("b")], check_names=False)


def test_no_match_split_across_train_and_test(split):
    _, _, train, test = split
    assert not set(train["match_id"]) & set(test["match_id"])
    for df in (train, test):
        assert (df["match_id"].value_counts() == 2).all()
        assert df["label"].mean() == 0.5


def test_split_years_levels_and_order(split):
    test_year, test_end_year, train, test = split
    assert train["year"].max() < test_year
    assert test["year"].between(test_year, test_end_year or test_year).all()
    assert test["tour_level"].isin(MAIN_TOUR_LEVELS).all()
    assert train["match_date"].is_monotonic_increasing
    assert test["match_date"].is_monotonic_increasing


def test_no_missing_features_after_split(split):
    _, _, train, test = split
    assert not train[FEATURE_COLUMNS].isna().any().any()
    assert not test[FEATURE_COLUMNS].isna().any().any()

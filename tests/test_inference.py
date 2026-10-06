"""The demo path (tennis.inference) must reproduce the training rows in pairs.parquet.

For real past matches, rebuilding the (A,B) and (B,A) rows from the history strictly before
the match date must give exactly the stored feature values. Only matches where neither
player has played an earlier round of the same tournament are used: for those, "history
before the date" and the pipeline's history are the same set of matches.

rank, ranking points, height, hand and age are passed in from the match record, because
they are tournament-date facts rather than history. Their defaults are tested separately.
"""

import numpy as np
import pandas as pd
import pytest

from tennis import config
from tennis.dataset import FEATURE_COLUMNS, get_train_test, load_pairs, train_medians
from tennis.inference import (AmbiguousPlayerError, History, UnknownPlayerError, build_match_row,
                              build_match_rows, combine, load_history)

pytestmark = pytest.mark.skipif(not config.PAIRS_PATH.exists(),
                                reason="run `python -m tennis.dataset` to build pairs.parquet")

N_MATCHES = 20


@pytest.fixture(scope="module")
def matches():
    return load_history()


@pytest.fixture(scope="module")
def pairs():
    return load_pairs()


def _first_match_in_tournament(m: pd.DataFrame) -> pd.Series:
    """True where neither player has an earlier-round match in the same tournament."""
    first = pd.concat([m[["tourney_id", "winner_id", "round_order"]].set_axis(["t", "p", "r"], axis=1),
                       m[["tourney_id", "loser_id", "round_order"]].set_axis(["t", "p", "r"], axis=1)]
                      ).groupby(["t", "p"])["r"].min()
    w = first.loc[list(zip(m["tourney_id"], m["winner_id"]))].to_numpy()
    l = first.loc[list(zip(m["tourney_id"], m["loser_id"]))].to_numpy()
    return pd.Series((m["round_order"].to_numpy() == w) & (m["round_order"].to_numpy() == l), index=m.index)


@pytest.fixture(scope="module")
def sample(matches):
    """20 eligible matches over 5 dates: two Grand Slams, two random dates, and a debut."""
    m = matches
    eligible = m[_first_match_in_tournament(m)]
    seen = set()
    debut = []
    for i, w, l in zip(m.index, m["winner_id"], m["loser_id"]):
        if (w not in seen or l not in seen) and i in eligible.index and m.at[i, "year"] >= 2015:
            debut.append(i)
        seen.update((w, l))
    rng = np.random.default_rng(229)
    debut_date = m.at[debut[len(debut) // 2], "match_date"]
    slam_date = lambda name, year: m.loc[(m["tourney_name"] == name) & (m["year"] == year)
                                         & (m["tour_level"] == "G"), "match_date"].iloc[0]
    dates = [slam_date("Australian Open", 2019), slam_date("Roland Garros", 2023), debut_date]
    other = eligible[~eligible["match_date"].isin(dates)]["match_date"].drop_duplicates()
    dates += list(rng.choice(other.to_numpy(), size=2, replace=False))

    picked = [debut[len(debut) // 2]]
    for d in dates:
        pool = eligible.index[(eligible["match_date"] == d) & ~eligible.index.isin(picked)]
        picked += list(rng.choice(pool, size=min(4 if d != debut_date else 3, len(pool)), replace=False))
    assert len(picked) == N_MATCHES
    return picked


def _record_attributes(row: pd.Series) -> dict:
    out = {}
    for side, prefix in (("a", "winner"), ("b", "loser")):
        for name, raw in (("rank", "rank"), ("rank_points", "rank_points"), ("height", "ht"),
                          ("hand", "hand"), ("age", "age")):
            out[f"{name}_{side}"] = row[f"{prefix}_{raw}"]
    return out


def test_rebuilt_rows_match_stored_rows(matches, pairs, sample):
    stored = pairs.set_index(["match_id", "label"])
    histories = {}
    debut_seen = False
    for i in sample:
        row = matches.loc[i]
        date = row["match_date"]
        if date not in histories:
            histories[date] = History(matches, before=date)
        rebuilt = build_match_rows(int(row["winner_id"]), int(row["loser_id"]), row["surface"], date,
                                   histories[date], tournament=row["tourney_id"],
                                   tour_level=row["tour_level"], attributes=_record_attributes(row))
        for k, label in enumerate((1, 0)):
            expected = stored.loc[(row["match_id"], label), FEATURE_COLUMNS].to_numpy(float)
            got = rebuilt.iloc[k][FEATURE_COLUMNS].to_numpy(float)
            mismatched = [c for c, e, g in zip(FEATURE_COLUMNS, expected, got)
                          if not (e == g or (np.isnan(e) and np.isnan(g)))]
            assert not mismatched, f"{row['match_id']} label {label}: {mismatched}"
        debut_seen |= rebuilt["has_history_a"].eq(0).any()
    assert debut_seen, "sample should include a player with no history"


def test_dataframe_history_equals_prebuilt_history(matches, sample):
    row = matches.loc[sample[0]]
    args = (int(row["winner_id"]), int(row["loser_id"]), row["surface"], row["match_date"])
    attrs = _record_attributes(row)
    a = build_match_rows(*args, matches, attributes=attrs)
    b = build_match_rows(*args, History(matches, before=row["match_date"]), attributes=attrs)
    pd.testing.assert_frame_equal(a, b)


def test_mirrored_rows_and_single_row(matches):
    h = History(matches)
    rows = build_match_rows("Novak Djokovic", "Rafael Nadal", "Clay", h.before, h)
    diffs = [c for c in FEATURE_COLUMNS if c.startswith("diff_")]
    assert np.array_equal(rows.iloc[0][diffs].to_numpy(float), -rows.iloc[1][diffs].to_numpy(float))
    assert rows.iloc[0]["player_a_name"] == rows.iloc[1]["player_b_name"] == "Novak Djokovic"
    single = build_match_row("Novak Djokovic", "Rafael Nadal", "Clay", h.before, h)
    pd.testing.assert_frame_equal(single, rows.iloc[[0]])
    assert combine(0.7, 0.4) == pytest.approx(0.65)


def test_median_fill_matches_get_train_test(matches, pairs):
    train, test = get_train_test(2019, pairs=pairs)
    medians = train_medians(train)
    eligible = matches[_first_match_in_tournament(matches) & (matches["year"] == 2019)
                       & matches["tour_level"].isin(["G", "M", "A", "F"])]
    row = eligible[eligible["winner_ht"].isna() | eligible["loser_ht"].isna()].iloc[0]
    rebuilt = build_match_rows(int(row["winner_id"]), int(row["loser_id"]), row["surface"],
                               row["match_date"], matches, tournament=row["tourney_id"],
                               attributes=_record_attributes(row), medians=medians)
    expected = test[test["match_id"] == row["match_id"]].sort_values("label", ascending=False)
    np.testing.assert_array_equal(rebuilt[FEATURE_COLUMNS].to_numpy(float),
                                  expected[FEATURE_COLUMNS].to_numpy(float))


def test_default_attributes_come_from_latest_earlier_match(matches):
    h = History(matches, before="2019-01-14")
    pid = h.resolve_player("Rafael Nadal")
    past = h.past[(h.past["winner_id"] == pid) | (h.past["loser_id"] == pid)].iloc[-1]
    side = "winner" if past["winner_id"] == pid else "loser"
    attrs = h.attributes(pid, pd.Timestamp("2019-01-14"))
    assert attrs["rank"] == past[f"{side}_rank"]
    assert attrs["age"] > past[f"{side}_age"]


@pytest.mark.parametrize("query, error, expected", [
    ("novak djokovic", None, "Novak Djokovic"),
    ("Federer", None, "Roger Federer"),
    ("Novak Djokovc", UnknownPlayerError, "Novak Djokovic"),
    ("Murray", AmbiguousPlayerError, "Andy Murray"),
    ("Zzyzx Qwerty", UnknownPlayerError, None),
])
def test_player_lookup(matches, query, error, expected):
    h = History(matches, before="2000-01-03")  # lookup covers all players, whatever the cutoff
    if error is None:
        assert h.names[h.resolve_player(query)] == expected
        return
    with pytest.raises(error) as info:
        h.resolve_player(query)
    if error is UnknownPlayerError:
        assert (expected in info.value.suggestions) if expected else not info.value.suggestions
    else:
        assert expected in [name for _, name, _ in info.value.candidates]


def test_rejects_history_that_reaches_the_match_date(matches):
    h = History(matches, before="2020-01-01")
    with pytest.raises(ValueError, match="history contains matches"):
        build_match_rows("Novak Djokovic", "Rafael Nadal", "Hard", "2019-06-01", h)

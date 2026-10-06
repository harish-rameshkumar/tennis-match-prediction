"""Leakage tests for the history-based features in tennis.features.

The history of a match is every match from a tournament with a strictly earlier start date,
plus earlier rounds of the same tournament. Never same-tournament same-or-later rounds,
never other tournaments with the same start date.

The reference implementation below rebuilds every feature from scratch for a sample of
matches by filtering the cleaned data explicitly. It is deliberately slow and shares no
code with tennis.features.
"""

import numpy as np
import pandas as pd
import pytest

from tennis import config
from tennis.clean import clean
from tennis.load import load_raw

pytestmark = pytest.mark.skipif(not (config.RAW_DIR / config.atp_file(2019)).exists(),
                                reason="run `python scripts/download_data.py` first")

WINDOWS = (5, 10, 15, 25)
STATS = ["svpt", "1stIn", "1stWon", "2ndWon", "ace", "df", "SvGms", "bpSaved", "bpFaced"]
ELO_START = 1500.0


@pytest.fixture(scope="module")
def matches():
    return clean(load_raw())


@pytest.fixture(scope="module")
def features():
    return pytest.importorskip("tennis.features")


# ---------------------------------------------------------------- reference implementation

def event_key(tourney_id: str) -> str:
    code = tourney_id.split("-", 1)[1]
    return (code.lstrip("0") or "0") if code.isdigit() else code


def history(m: pd.DataFrame, i) -> pd.DataFrame:
    row = m.loc[i]
    earlier = m["match_date"] < row["match_date"]
    earlier_round = (m["tourney_id"] == row["tourney_id"]) & (m["round_order"] < row["round_order"])
    return m[earlier | earlier_round]


def _rate(won: np.ndarray) -> float:
    return float(won.mean()) if len(won) else 0.0


def _k(n: int) -> float:
    return 250.0 / (n + 5) ** 0.4


def _elo_update(r, n, w, l):
    rw, rl = r.get(w, ELO_START), r.get(l, ELO_START)
    expected = 1.0 / (1.0 + 10 ** ((rl - rw) / 400.0))
    r[w] = rw + _k(n.get(w, 0)) * (1 - expected)
    r[l] = rl - _k(n.get(l, 0)) * (1 - expected)
    n[w] = n.get(w, 0) + 1
    n[l] = n.get(l, 0) + 1


def player_reference(hist: pd.DataFrame, row: pd.Series, p: int, o: int) -> tuple[dict, np.ndarray, np.ndarray]:
    h = hist[(hist["winner_id"] == p) | (hist["loser_id"] == p)]
    won = (h["winner_id"] == p).to_numpy()
    opp = np.where(won, h["loser_id"], h["winner_id"])
    n = len(h)
    f = {"has_history": int(n > 0), "career_wins": won.sum(), "career_losses": n - won.sum(),
         "career_win_rate": _rate(won)}
    for k in WINDOWS:
        f[f"last{k}_played"] = min(k, n)
        f[f"last{k}_win_rate"] = _rate(won[-k:])

    recent = ((h["match_date"] >= row["match_date"] - pd.Timedelta(days=30))
              & (h["match_date"] < row["match_date"])).to_numpy()
    f["recent30_wins"] = won[recent].sum()
    f["recent30_losses"] = (~won[recent]).sum()

    surf = (h["surface"] == row["surface"]).to_numpy()
    f["surface_wins"] = won[surf].sum()
    f["surface_losses"] = (~won[surf]).sum()
    f["surface_win_rate"] = _rate(won[surf])
    for k in WINDOWS:
        f[f"surface_last{k}_played"] = min(k, surf.sum())
        f[f"surface_last{k}_win_rate"] = _rate(won[surf][-k:])

    same_event = (h["tourney_id"].map(event_key) == event_key(row["tourney_id"])).to_numpy()
    f["tourney_wins"] = won[same_event].sum()
    f["tourney_losses"] = (~won[same_event]).sum()

    vs = opp == o
    f["h2h_wins"] = won[vs].sum()
    f["h2h_surface_wins"] = won[vs & surf].sum()

    vals = np.where(won[:, None], h[[f"w_{s}" for s in STATS]].to_numpy(float),
                    h[[f"l_{s}" for s in STATS]].to_numpy(float))
    ok = ~np.isnan(vals).any(axis=1) & (vals[:, 0] > 0) if n else np.zeros(0, bool)
    t = dict(zip(STATS, vals[ok].sum(axis=0))) if ok.any() else dict.fromkeys(STATS, 0.0)
    cnt = ok.sum()
    div = lambda a, b: a / b if b > 0 else 0.0
    f["has_serve_stats"] = int(cnt > 0)
    f["first_in_pct"] = div(t["1stIn"], t["svpt"])
    f["first_won_pct"] = div(t["1stWon"], t["1stIn"])
    f["second_won_pct"] = div(t["2ndWon"], t["svpt"] - t["1stIn"])
    f["aces_per_match"] = div(t["ace"], cnt)
    f["dfs_per_match"] = div(t["df"], cnt)
    f["bp_saved_pct"] = div(t["bpSaved"], t["bpFaced"])
    f["bp_saved_per_match"] = div(t["bpSaved"], cnt)
    f["bp_faced_per_match"] = div(t["bpFaced"], cnt)
    f["sv_games_per_match"] = div(t["SvGms"], cnt)
    return f, won, opp


def reference(m: pd.DataFrame, i) -> dict:
    """All features of match i, as named by tennis.features.compute_features."""
    row = m.loc[i]
    hist = history(m, i)
    w, l = row["winner_id"], row["loser_id"]
    fw, won_w, opp_w = player_reference(hist, row, w, l)
    fl, won_l, opp_l = player_reference(hist, row, l, w)

    common = (set(opp_w) & set(opp_l)) - {w, l}
    fw["common_opp_win_rate"] = _rate(won_w[np.isin(opp_w, list(common))])
    fl["common_opp_win_rate"] = _rate(won_l[np.isin(opp_l, list(common))])

    r, n, rs, ns = {}, {}, {}, {}
    for a, b, s in zip(hist["winner_id"], hist["loser_id"], hist["surface"]):
        _elo_update(r, n, a, b)
        _elo_update(rs, ns, (a, s), (b, s))
    fw["elo"], fl["elo"] = r.get(w, ELO_START), r.get(l, ELO_START)
    fw["surface_elo"] = rs.get((w, row["surface"]), ELO_START)
    fl["surface_elo"] = rs.get((l, row["surface"]), ELO_START)

    out = {f"winner_{k}": v for k, v in fw.items()} | {f"loser_{k}": v for k, v in fl.items()}
    out["h2h_played"] = fw["h2h_wins"] + fl["h2h_wins"]
    out["h2h_surface_played"] = fw["h2h_surface_wins"] + fl["h2h_surface_wins"]
    out["common_opp_count"] = len(common)
    return out


def sample_matches(m: pd.DataFrame) -> list:
    """Random matches plus the cases most likely to leak."""
    rng = np.random.default_rng(229)
    pick = lambda idx, k: list(rng.choice(np.asarray(idx), size=min(k, len(idx)), replace=False))

    players = pd.concat([m[["match_date", "tourney_id", "winner_id"]].rename(columns={"winner_id": "pid"}),
                         m[["match_date", "tourney_id", "loser_id"]].rename(columns={"loser_id": "pid"})])
    multi = players.drop_duplicates().groupby(["match_date", "pid"])["tourney_id"].nunique()
    multi = multi[multi > 1].reset_index()[["match_date", "pid"]]
    same_date_other_event = m.reset_index().merge(multi, left_on=["match_date", "winner_id"],
                                                  right_on=["match_date", "pid"])["index"]
    appearances = pd.DataFrame({"i": np.r_[m.index, m.index], "pid": np.r_[m["winner_id"], m["loser_id"]]})
    first_ever = appearances.groupby("pid")["i"].min().unique()

    idx = (pick(m.index, 30)
           + pick(m.index[m["round"] == "F"], 6)
           + pick(m.index[m["round"] == "RR"], 12)  # only RR can expose same-round leaks
           + pick(same_date_other_event, 10)
           + pick(first_ever, 4))
    return sorted(set(idx))


# ---------------------------------------------------------------- tests

def test_later_season_never_dated_before_earlier_season(matches):
    """So strictly-earlier history can never pull a test-year match into a training match."""
    by_date = matches.groupby("match_date")["year"].agg(["min", "max"]).sort_index()
    assert (by_date["min"] >= by_date["max"].cummax().shift().fillna(0)).all()


def test_pipeline_matches_brute_force(matches, features):
    computed = features.compute_features(matches)
    sample = sample_matches(matches)
    expected = {i: reference(matches, i) for i in sample}

    missing = set(computed.columns) - set(expected[sample[0]])
    assert not missing, f"features without a brute-force reference: {sorted(missing)}"

    for i in sample:
        for col in computed.columns:
            assert np.isclose(computed.at[i, col], expected[i][col], rtol=1e-9, atol=1e-9), (
                f"match {matches.at[i, 'match_id']} ({matches.at[i, 'round']}): "
                f"{col} pipeline={computed.at[i, col]} brute_force={expected[i][col]}")


@pytest.fixture(scope="module")
def recent(matches):
    return matches[matches["year"].between(2017, 2019)].reset_index(drop=True)


def test_changing_a_future_result_changes_no_earlier_features(recent, features):
    base = features.compute_features(recent)

    i = recent.index[(recent["year"] == 2019) & (recent["round"] == "QF")
                     & (recent["tour_level"] == "M")][0]
    flipped = recent.copy()
    swaps = [(c, c.replace("winner_", "loser_", 1)) for c in recent.columns if c.startswith("winner_")]
    swaps += [(c, "l_" + c[2:]) for c in recent.columns if c.startswith("w_")]
    for a, b in swaps:
        flipped.at[i, a], flipped.at[i, b] = recent.at[i, b], recent.at[i, a]
    after = features.compute_features(flipped)

    row = recent.loc[i]
    cannot_see = ((recent["match_date"] < row["match_date"])
                  | ((recent["match_date"] == row["match_date"]) & (recent["tourney_id"] != row["tourney_id"]))
                  | ((recent["tourney_id"] == row["tourney_id"]) & (recent["round_order"] <= row["round_order"])))
    cannot_see.loc[i] = False  # its own winner/loser columns are swapped, compared below
    pd.testing.assert_frame_equal(base[cannot_see], after[cannot_see])

    # Sanity: the flip must actually reach later matches, or this test proves nothing.
    assert not base[~cannot_see].drop(index=i).equals(after[~cannot_see].drop(index=i))


def test_training_features_ignore_test_year(recent, features):
    """Training-year features are identical with and without the test year in the data."""
    full = features.compute_features(recent)
    train_only = recent[recent["year"] < 2019]
    pd.testing.assert_frame_equal(features.compute_features(train_only), full.loc[train_only.index])

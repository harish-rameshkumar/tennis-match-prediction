"""History-based pre-match features, computed in one chronological pass.

The history of a match is every match from a tournament with a strictly earlier start date,
plus earlier rounds of the same tournament. Matches in the same round, later rounds and
other tournaments with the same start date are never used. tests/test_no_leakage.py checks
this against a brute-force reference.

Each player keeps a running state that is updated after every match. On a given start date,
each tournament reads the state as it was at the end of the previous date plus its own
earlier rounds. When a player appears in two tournaments with the same start date (e.g.
Grand Slam qualifying and a Challenger), each tournament works on its own copy of the
state, and the date's matches are then applied to the shared state in row order.
"""

from collections import deque

import numpy as np
import pandas as pd

WINDOWS = (5, 10, 15, 25)
RECENT_DAYS = 30
# Sackmann serve-stat columns (w_<stat> for the winner, l_<stat> for the loser).
SERVE_STATS = ["svpt", "1stIn", "1stWon", "2ndWon", "ace", "df", "SvGms", "bpSaved", "bpFaced"]
SVPT, FIRST_IN, FIRST_WON, SECOND_WON, ACE, DF, SV_GMS, BP_SAVED, BP_FACED = range(len(SERVE_STATS))
ELO_START = 1500.0

# Per-player features. The pairs table holds diff_<name> = A - B for each.
DIFF_FEATURES = (
    ["career_wins", "career_losses", "career_win_rate"]
    + [f"last{k}_{s}" for k in WINDOWS for s in ("played", "win_rate")]
    + ["recent30_wins", "recent30_losses"]
    + ["surface_wins", "surface_losses", "surface_win_rate"]
    + [f"surface_last{k}_{s}" for k in WINDOWS for s in ("played", "win_rate")]
    + ["tourney_wins", "tourney_losses"]
    + ["h2h_wins", "h2h_surface_wins", "common_opp_win_rate"]
    + ["first_in_pct", "first_won_pct", "second_won_pct", "aces_per_match", "dfs_per_match",
       "bp_saved_pct", "bp_saved_per_match", "bp_faced_per_match", "sv_games_per_match"]
    + ["elo", "surface_elo"]
)
# Per-player flags, kept as <name>_a / <name>_b.
FLAG_FEATURES = ["has_history", "has_serve_stats"]
# Symmetric pair features (same value in both mirrored rows).
PAIR_FEATURES = ["h2h_played", "h2h_surface_played", "common_opp_count"]

PLAYER_FEATURES = FLAG_FEATURES + DIFF_FEATURES


def event_key(tourney_id: str) -> str:
    """Identify a tournament across years: '2019-0854' and '2010-854' are the same event."""
    code = tourney_id.split("-", 1)[1]
    return (code.lstrip("0") or "0") if code.isdigit() else code


def _rate(wins, played) -> float:
    return wins / played if played else 0.0


class PlayerState:
    __slots__ = ("wins", "played", "results", "recent", "surface", "surface_results", "event",
                 "opp", "opp_surface", "serve", "serve_matches", "elo", "surface_elo")

    def __init__(self):
        self.wins = 0
        self.played = 0
        self.results = deque(maxlen=max(WINDOWS))   # 1 = win, 0 = loss, oldest first
        self.recent = deque()                       # (day, won) of matches in the last RECENT_DAYS
        self.surface = {}                           # surface -> [wins, played]
        self.surface_results = {}                   # surface -> deque of 1/0
        self.event = {}                             # event key -> [wins, played]
        self.opp = {}                               # opponent id -> [wins, played]
        self.opp_surface = {}                       # (opponent id, surface) -> [wins, played]
        self.serve = [0.0] * len(SERVE_STATS)       # career sums over matches with full stats
        self.serve_matches = 0
        self.elo = [ELO_START, 0]                   # [rating, matches rated]
        self.surface_elo = {}                       # surface -> [rating, matches rated]

    def copy(self) -> "PlayerState":
        c = PlayerState()
        c.wins, c.played = self.wins, self.played
        c.results = self.results.copy()
        c.recent = self.recent.copy()
        c.surface = {k: v.copy() for k, v in self.surface.items()}
        c.surface_results = {k: v.copy() for k, v in self.surface_results.items()}
        c.event = {k: v.copy() for k, v in self.event.items()}
        c.opp = {k: v.copy() for k, v in self.opp.items()}
        c.opp_surface = {k: v.copy() for k, v in self.opp_surface.items()}
        c.serve, c.serve_matches = self.serve.copy(), self.serve_matches
        c.elo = self.elo.copy()
        c.surface_elo = {k: v.copy() for k, v in self.surface_elo.items()}
        return c


def _last(results, k) -> tuple[int, float]:
    tail = list(results)[-k:]
    return len(tail), _rate(sum(tail), len(tail))


def _player_features(p: PlayerState, opp_id: int, common: set, day: int, surface: str,
                     event: str) -> list:
    while p.recent and p.recent[0][0] < day - RECENT_DAYS:
        p.recent.popleft()
    recent = [won for d, won in p.recent if d < day]
    s_wins, s_played = p.surface.get(surface, (0, 0))
    s_results = p.surface_results.get(surface, ())
    e_wins, e_played = p.event.get(event, (0, 0))

    out = [int(p.played > 0), int(p.serve_matches > 0)]  # FLAG_FEATURES first
    out += [p.wins, p.played - p.wins, _rate(p.wins, p.played)]
    for k in WINDOWS:
        out += _last(p.results, k)
    out += [sum(recent), len(recent) - sum(recent)]
    out += [s_wins, s_played - s_wins, _rate(s_wins, s_played)]
    for k in WINDOWS:
        out += _last(s_results, k)
    out += [e_wins, e_played - e_wins]

    c_wins = c_played = 0
    for c in common:
        c_wins += p.opp[c][0]
        c_played += p.opp[c][1]
    out += [p.opp.get(opp_id, (0, 0))[0], p.opp_surface.get((opp_id, surface), (0, 0))[0],
            _rate(c_wins, c_played)]

    t, n = p.serve, p.serve_matches
    out += [_rate(t[FIRST_IN], t[SVPT]), _rate(t[FIRST_WON], t[FIRST_IN]),
            _rate(t[SECOND_WON], t[SVPT] - t[FIRST_IN]), _rate(t[ACE], n), _rate(t[DF], n),
            _rate(t[BP_SAVED], t[BP_FACED]), _rate(t[BP_SAVED], n), _rate(t[BP_FACED], n),
            _rate(t[SV_GMS], n)]
    out += [p.elo[0], p.surface_elo.get(surface, (ELO_START, 0))[0]]
    return out


def _pair_features(w: PlayerState, loser_id: int, common: set, surface: str) -> list:
    return [w.opp.get(loser_id, (0, 0))[1], w.opp_surface.get((loser_id, surface), (0, 0))[1],
            len(common)]


def _common_opponents(w: PlayerState, l: PlayerState, winner_id: int, loser_id: int) -> set:
    return (w.opp.keys() & l.opp.keys()) - {winner_id, loser_id}


def _record(p: PlayerState, opp_id: int, won: int, day: int, surface: str, event: str,
            stats: np.ndarray) -> None:
    p.wins += won
    p.played += 1
    p.results.append(won)
    p.recent.append((day, won))
    s = p.surface.setdefault(surface, [0, 0])
    s[0] += won
    s[1] += 1
    p.surface_results.setdefault(surface, deque(maxlen=max(WINDOWS))).append(won)
    for record in (p.event.setdefault(event, [0, 0]), p.opp.setdefault(opp_id, [0, 0]),
                   p.opp_surface.setdefault((opp_id, surface), [0, 0])):
        record[0] += won
        record[1] += 1
    # Only matches with every stat present (and svpt > 0) count towards serve averages.
    if stats[SVPT] > 0 and not np.isnan(stats).any():
        p.serve = [a + b for a, b in zip(p.serve, stats.tolist())]
        p.serve_matches += 1


def _elo_k(n: int) -> float:
    """FiveThirtyEight-style K-factor: large for newcomers, shrinking with experience."""
    return 250.0 / (n + 5) ** 0.4


def _elo_update(w: list, l: list) -> None:
    """Update [rating, matches] of winner w and loser l in place."""
    expected = 1.0 / (1.0 + 10 ** ((l[0] - w[0]) / 400.0))
    w[0] += _elo_k(w[1]) * (1 - expected)
    l[0] -= _elo_k(l[1]) * (1 - expected)
    w[1] += 1
    l[1] += 1


def _update(w: PlayerState, l: PlayerState, winner_id: int, loser_id: int, day: int,
            surface: str, event: str, w_stats: np.ndarray, l_stats: np.ndarray) -> None:
    _elo_update(w.elo, l.elo)
    _elo_update(w.surface_elo.setdefault(surface, [ELO_START, 0]),
                l.surface_elo.setdefault(surface, [ELO_START, 0]))
    _record(w, loser_id, 1, day, surface, event, w_stats)
    _record(l, winner_id, 0, day, surface, event, l_stats)


def _one_match(w: PlayerState, l: PlayerState, winner_id: int, loser_id: int, day: int,
               surface: str, event) -> tuple[list, list, list]:
    """Pre-match features of one match from the two players' current states."""
    common = _common_opponents(w, l, winner_id, loser_id)
    return (_player_features(w, loser_id, common, day, surface, event),
            _player_features(l, winner_id, common, day, surface, event),
            _pair_features(w, loser_id, common, surface))


def _to_frame(out_w, out_l, out_pair, index) -> pd.DataFrame:
    cols_w = [f"winner_{f}" for f in PLAYER_FEATURES]
    cols_l = [f"loser_{f}" for f in PLAYER_FEATURES]
    frames = [pd.DataFrame(np.asarray(out_w, dtype=float).reshape(len(index), -1), index=index, columns=cols_w),
              pd.DataFrame(np.asarray(out_l, dtype=float).reshape(len(index), -1), index=index, columns=cols_l),
              pd.DataFrame(np.asarray(out_pair, dtype=float).reshape(len(index), -1), index=index,
                           columns=PAIR_FEATURES)]
    return pd.concat(frames, axis=1)


def _day(dates) -> np.ndarray:
    return np.asarray(dates, dtype="datetime64[D]").astype(np.int64)


def _arrays(matches: pd.DataFrame) -> dict:
    a = {
        "winner": matches["winner_id"].to_numpy(),
        "loser": matches["loser_id"].to_numpy(),
        "day": _day(matches["match_date"].to_numpy()),
        "tourney": matches["tourney_id"].to_numpy(),
        "rnd": matches["round_order"].to_numpy(),
        "surface": matches["surface"].to_numpy(),
        "event": matches["tourney_id"].map(event_key).to_numpy(),
        "w_stats": matches[[f"w_{s}" for s in SERVE_STATS]].to_numpy(float),
        "l_stats": matches[[f"l_{s}" for s in SERVE_STATS]].to_numpy(float),
    }
    if (np.diff(a["day"]) < 0).any():
        raise ValueError("matches must be sorted chronologically (use tennis.clean.clean)")
    return a


def replay(matches: pd.DataFrame) -> dict[int, PlayerState]:
    """Player states after every match in `matches`, applied in row order.

    This is the state compute_features holds at the end of the data, so features built
    from it (see match_features) agree with the pipeline for a match on a later date.
    """
    a = _arrays(matches)
    states: dict[int, PlayerState] = {}
    for i in range(len(matches)):
        w = states.get(a["winner"][i]) or states.setdefault(a["winner"][i], PlayerState())
        l = states.get(a["loser"][i]) or states.setdefault(a["loser"][i], PlayerState())
        _update(w, l, a["winner"][i], a["loser"][i], a["day"][i], a["surface"][i], a["event"][i],
                a["w_stats"][i], a["l_stats"][i])
    return states


def match_features(states: dict[int, PlayerState], player_a: int, player_b: int, date,
                   surface: str, tourney_id: str | None = None) -> pd.DataFrame:
    """Features of one hypothetical match, A in the winner_ columns and B in loser_.

    states comes from replay() over every match before `date`. Players missing from
    states have no history (zeros, has_history = 0). Without tourney_id the
    tournament-record features are 0. states is not modified.
    """
    a = states[player_a].copy() if player_a in states else PlayerState()
    b = states[player_b].copy() if player_b in states else PlayerState()
    event = event_key(tourney_id) if tourney_id else None
    fa, fb, pair = _one_match(a, b, player_a, player_b, int(_day([date])[0]), surface, event)
    return _to_frame([fa], [fb], [pair], pd.RangeIndex(1))


def compute_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Pre-match features for every match, aligned with matches.index.

    matches must be in the order produced by tennis.clean.clean (start date, tournament,
    round, match number). Columns: winner_<f> and loser_<f> for f in PLAYER_FEATURES,
    plus PAIR_FEATURES.
    """
    n = len(matches)
    arrays = _arrays(matches)
    winner, loser, day, tourney, rnd = (arrays[k] for k in ("winner", "loser", "day", "tourney", "rnd"))
    surface, event, w_stats, l_stats = (arrays[k] for k in ("surface", "event", "w_stats", "l_stats"))

    out_w, out_l, out_pair = [None] * n, [None] * n, [None] * n
    base: dict[int, PlayerState] = {}

    def run_tournament(positions, state):
        """Features for each round from the state before it, then apply that round."""
        start = 0
        while start < len(positions):
            end = start
            while end < len(positions) and rnd[positions[end]] == rnd[positions[start]]:
                end += 1
            block = positions[start:end]
            for i in block:
                out_w[i], out_l[i], out_pair[i] = _one_match(
                    state(winner[i]), state(loser[i]), winner[i], loser[i], day[i], surface[i], event[i])
            for i in block:
                _update(state(winner[i]), state(loser[i]), winner[i], loser[i], day[i],
                        surface[i], event[i], w_stats[i], l_stats[i])
            start = end

    def base_state(pid):
        s = base.get(pid)
        if s is None:
            s = base[pid] = PlayerState()
        return s

    boundaries = np.flatnonzero(np.diff(day)) + 1
    for lo, hi in zip(np.r_[0, boundaries], np.r_[boundaries, n]):
        tournaments: dict[str, list[int]] = {}
        for i in range(lo, hi):
            tournaments.setdefault(tourney[i], []).append(i)
        for positions in tournaments.values():
            if positions[-1] - positions[0] + 1 != len(positions) or (np.diff(rnd[positions]) < 0).any():
                raise ValueError("matches must be sorted by tournament and round within a date")

        seen, shared = set(), False
        for positions in tournaments.values():
            players = set(winner[positions]) | set(loser[positions])
            shared |= not seen.isdisjoint(players)
            seen |= players

        if not shared:
            for positions in tournaments.values():
                run_tournament(positions, base_state)
            continue

        for positions in tournaments.values():
            local: dict[int, PlayerState] = {}

            def local_state(pid):
                s = local.get(pid)
                if s is None:
                    s = local[pid] = base[pid].copy() if pid in base else PlayerState()
                return s

            run_tournament(positions, local_state)
        for i in range(lo, hi):
            _update(base_state(winner[i]), base_state(loser[i]), winner[i], loser[i], day[i],
                    surface[i], event[i], w_stats[i], l_stats[i])

    return _to_frame(out_w, out_l, out_pair, matches.index)

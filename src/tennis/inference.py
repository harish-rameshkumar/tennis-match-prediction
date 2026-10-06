"""Feature rows for a hypothetical match, built with the training pipeline's own code.

    from tennis.inference import History, build_match_rows, combine
    history = History(load_history())                      # all matches in the data
    rows = build_match_rows("Djokovic", "Alcaraz", "Grass", "2025-07-01", history)
    p_ab, p_ba = model.predict_proba(rows[FEATURE_COLUMNS])[:, 1]
    p_a_wins = combine(p_ab, p_ba)

History features come from tennis.features.replay over every match strictly before `date`,
and the rows are assembled by tennis.dataset.build_pairs, so they match the training rows
column for column. See README "Inference interface" for the full contract.
"""

import difflib
import re
import unicodedata

import numpy as np
import pandas as pd

from tennis.clean import clean
from tennis.dataset import FEATURE_COLUMNS, ID_COLUMNS, META_COLUMNS, build_pairs, fill_missing
from tennis.features import match_features, replay
from tennis.load import load_raw

SURFACES = ("Hard", "Clay", "Grass", "Carpet")
# attribute name in the pairs table -> suffix of the raw winner_/loser_ column
ATTRIBUTES = {"rank": "rank", "rank_points": "rank_points", "height": "ht", "hand": "hand", "age": "age"}
OUTPUT_COLUMNS = [c for c in ID_COLUMNS if c != "label"] + META_COLUMNS + FEATURE_COLUMNS


class UnknownPlayerError(ValueError):
    """No player matches the name; .suggestions holds the closest known names."""

    def __init__(self, query, suggestions: list[str]):
        self.query, self.suggestions = query, suggestions
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else " No similar names found."
        super().__init__(f"Unknown player {query!r}.{hint}")


class AmbiguousPlayerError(ValueError):
    """Several players match the name; .candidates lists them as (player_id, name, last match date)."""

    def __init__(self, query, candidates: list[tuple[int, str, str]]):
        self.query, self.candidates = query, candidates
        listed = "; ".join(f"{name} (id {pid}, last match {last})" for pid, name, last in candidates)
        super().__init__(f"{query!r} matches several players: {listed}. Use a fuller name or the player id.")


def load_history() -> pd.DataFrame:
    """All cleaned matches, in pipeline order."""
    return clean(load_raw())


def _normalise(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^a-z ]", " ", name.lower()).split())


class History:
    """Player states after every match strictly before `before`.

    before defaults to the day after the last match, i.e. all data is history. The
    player and tournament directories cover all of `matches` (names are not leakage).
    """

    def __init__(self, matches: pd.DataFrame, before=None):
        self.before = (pd.Timestamp(before) if before is not None
                       else matches["match_date"].max() + pd.Timedelta(days=1))
        self.matches = matches
        self.past = matches[matches["match_date"] < self.before]
        self.states = replay(self.past)

        appearances = pd.concat([
            matches[["winner_id", "winner_name", "match_date"]].set_axis(["pid", "name", "date"], axis=1),
            matches[["loser_id", "loser_name", "match_date"]].set_axis(["pid", "name", "date"], axis=1),
        ]).sort_values("date", kind="stable")
        latest = appearances.drop_duplicates("pid", keep="last")
        self.names = dict(zip(latest["pid"], latest["name"]))
        self.last_match = dict(zip(latest["pid"], latest["date"].dt.date.astype(str)))
        self.ids_by_name: dict[str, set[int]] = {}
        for pid, name in appearances[["pid", "name"]].drop_duplicates().itertuples(index=False):
            self.ids_by_name.setdefault(_normalise(name), set()).add(int(pid))

    def resolve_player(self, query) -> int:
        """Player id for an id or a (possibly partial or misspelled) name."""
        if isinstance(query, (int, np.integer)):
            if int(query) not in self.names:
                raise UnknownPlayerError(query, [])
            return int(query)

        norm = _normalise(str(query))
        ids = self.ids_by_name.get(norm)
        if ids is None:
            tokens = set(norm.split())
            partial = [n for n in self.ids_by_name if tokens and tokens <= set(n.split())]
            if len(partial) == 1:
                ids = self.ids_by_name[partial[0]]
            elif partial:
                ids = set().union(*(self.ids_by_name[n] for n in partial))
            else:
                raise UnknownPlayerError(query, self._suggest(norm))
        if len(ids) > 1:
            candidates = sorted(((pid, self.names[pid], self.last_match[pid]) for pid in ids),
                                key=lambda c: c[2], reverse=True)
            raise AmbiguousPlayerError(query, candidates[:10])
        return next(iter(ids))

    def _suggest(self, norm: str, n: int = 5) -> list[str]:
        names = list(self.ids_by_name)
        close = difflib.get_close_matches(norm, names, n=n, cutoff=0.6)
        # Also compare against surnames, so "djokovc" finds "novak djokovic".
        by_last = {}
        for name in names:
            by_last.setdefault(name.split()[-1], []).append(name)
        for last in difflib.get_close_matches(norm.split()[-1] if norm else "", list(by_last), n=n, cutoff=0.75):
            close += by_last[last]
        display = {_normalise(v): v for v in self.names.values()}
        return list(dict.fromkeys(display[c] for c in close if c in display))[:n]

    def resolve_tournament(self, query: str) -> tuple[str, str]:
        """(tourney_id, tourney_name) for a tourney_id or a tournament name (latest edition)."""
        m = self.matches
        if (m["tourney_id"] == query).any():
            return query, m.loc[m["tourney_id"] == query, "tourney_name"].iloc[0]
        hits = m[m["tourney_name"].str.lower() == str(query).lower()]
        if hits.empty:
            known = m["tourney_name"].drop_duplicates().tolist()
            close = difflib.get_close_matches(str(query), known, n=5, cutoff=0.6)
            hint = f" Did you mean: {', '.join(close)}?" if close else ""
            raise ValueError(f"Unknown tournament {query!r}.{hint}")
        last = hits.iloc[-1]
        return last["tourney_id"], last["tourney_name"]

    def attributes(self, pid: int, date: pd.Timestamp) -> dict:
        """rank, rank_points, height, hand, age as of the player's latest match before `before`.

        Missing values are NaN (no earlier match, or never recorded); age is advanced to `date`.
        """
        m = self.past
        rows = pd.concat([
            m.loc[m["winner_id"] == pid, ["match_date"] + [f"winner_{c}" for c in ATTRIBUTES.values()]]
             .set_axis(["match_date", *ATTRIBUTES], axis=1),
            m.loc[m["loser_id"] == pid, ["match_date"] + [f"loser_{c}" for c in ATTRIBUTES.values()]]
             .set_axis(["match_date", *ATTRIBUTES], axis=1),
        ]).sort_index()
        if rows.empty:
            return dict.fromkeys(ATTRIBUTES, np.nan)
        last = rows.iloc[-1]
        out = {"rank": last["rank"], "rank_points": last["rank_points"]}
        for col in ("height", "hand"):
            known = rows[col].dropna()
            out[col] = known.iloc[-1] if len(known) else np.nan
        aged = rows.dropna(subset=["age"])
        out["age"] = (aged["age"].iloc[-1] + (date - aged["match_date"].iloc[-1]).days / 365.25
                      if len(aged) else np.nan)
        return out


def combine(p_ab: float, p_ba: float) -> float:
    """P(A beats B) from the model's probabilities for the (A,B) and (B,A) rows."""
    return (p_ab + 1 - p_ba) / 2


def build_match_rows(player_a, player_b, surface: str, date, history, *, tournament: str | None = None,
                     tour_level: str | None = None, attributes: dict | None = None,
                     medians: dict | None = None) -> pd.DataFrame:
    """The (A,B) and (B,A) rows for a match between player_a and player_b on `date`.

    player_a / player_b: player id, or a name (case/accent-insensitive; a unique partial
        name such as "Djokovic" works). Unknown or ambiguous names raise
        UnknownPlayerError / AmbiguousPlayerError with suggestions.
    history: a History, or a DataFrame of cleaned matches (then History(history, before=date)
        is built, ~1-2 s). A History must not contain matches on or after `date`.
    tournament: tourney_id or tournament name; enables the tourney_* features (else 0).
    attributes: overrides such as {"rank_a": 3, "rank_points_b": 4100, "hand_a": "L"};
        keys are <rank|rank_points|height|hand|age>_<a|b>. Defaults come from each player's
        latest match before `date`. rank is required for both players.
    medians: from tennis.dataset.train_medians(train); fills NaNs exactly like get_train_test.
        Without it, rows keep NaN plus *_missing flags, like pairs.parquet.

    Returns 2 rows with OUTPUT_COLUMNS (pairs.parquet columns without label): row 0 is (A,B),
    row 1 is (B,A). Every diff_* in row 1 is the exact negative of row 0.
    """
    date = pd.Timestamp(date)
    if surface not in SURFACES:
        raise ValueError(f"surface must be one of {SURFACES}, got {surface!r}")
    if not isinstance(history, History):
        history = History(history, before=date)
    elif date < history.before:
        raise ValueError(f"history contains matches up to {history.before.date()} (exclusive); "
                         f"build History(matches, before={date.date()}) to predict a match on {date.date()}")

    a, b = history.resolve_player(player_a), history.resolve_player(player_b)
    if a == b:
        raise ValueError("player_a and player_b are the same player")
    tourney_id, tourney_name = history.resolve_tournament(tournament) if tournament else (None, None)

    attrs = {"a": history.attributes(a, date), "b": history.attributes(b, date)}
    for key, value in (attributes or {}).items():
        name, _, side = key.rpartition("_")
        if name not in ATTRIBUTES or side not in attrs:
            raise ValueError(f"unknown attribute {key!r}; use <{'|'.join(ATTRIBUTES)}>_<a|b>")
        attrs[side][name] = value
    for side, pid in (("a", a), ("b", b)):
        if pd.isna(attrs[side]["rank"]):
            raise ValueError(f"no ranking known for {history.names[pid]}; pass attributes={{'rank_{side}': ...}}")

    match = {"match_id": None, "match_date": date, "year": date.year, "tour_level": tour_level,
             "surface": surface, "tourney_name": tourney_name, "round": None}
    for prefix, side, pid in (("winner", "a", a), ("loser", "b", b)):
        match[f"{prefix}_id"] = pid
        match[f"{prefix}_name"] = history.names[pid]
        for name, raw in ATTRIBUTES.items():
            match[f"{prefix}_{raw}"] = attrs[side][name]

    feats = match_features(history.states, a, b, date, surface, tourney_id)
    rows = build_pairs(pd.DataFrame([match]), feats)[OUTPUT_COLUMNS]
    return fill_missing(rows, medians) if medians else rows


def build_match_row(player_a, player_b, surface: str, date, history, **kwargs) -> pd.DataFrame:
    """Just the (A,B) row of build_match_rows, as a 1-row DataFrame."""
    return build_match_rows(player_a, player_b, surface, date, history, **kwargs).iloc[[0]]

"""Clean raw matches into one row per completed match, in chronological order.

Cleaning rules (also listed in the README):

1. Drop rows missing winner_id, loser_id, tourney_date, winner_rank or loser_rank,
   and rows where winner_id == loser_id.
2. Exclude walkovers: score contains "W/O" or "Walkover". Rows with no score at all are
   also dropped, since they have no match statistics and can't be told apart from walkovers.
3. Keep retirements ("RET"), defaults ("DEF") and abandoned matches: the recorded winner
   stands, as in the official results.
4. tour_level is Sackmann's tourney_level (G, M, A, F, D, O, C), except qualifying rounds
   (Q1-Q4), which become "Q" because their tourney_level is that of the parent event.
5. Missing surface becomes "Unknown". Hand codes other than R/L ("U" unknown, "A") become NaN.
6. match_date is the tournament start date (the data has no per-match date); within a
   tournament, matches are ordered by round and then match_num.
"""

import pandas as pd

ROUND_ORDER = {
    "Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3, "ER": 4,
    "R128": 5, "R64": 6, "R32": 7, "R16": 8, "RR": 9,
    "QF": 10, "SF": 11, "BR": 12, "F": 13,
}

WALKOVER_PATTERN = r"W/O|Walkover"


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.dropna(subset=["winner_id", "loser_id", "tourney_date", "winner_rank", "loser_rank"])
    df = df[df["winner_id"] != df["loser_id"]]
    df = df[df["score"].notna() & ~df["score"].str.contains(WALKOVER_PATTERN, case=False)]

    is_qualifying = df["round"].str.fullmatch(r"Q\d")
    unknown_round = ~df["round"].isin(ROUND_ORDER.keys())
    if unknown_round.any():
        raise ValueError(f"unknown round codes: {sorted(df.loc[unknown_round, 'round'].unique())}")

    df = df.assign(
        match_id=df["tourney_id"] + "-" + df["match_num"].astype(str),
        match_date=pd.to_datetime(df["tourney_date"].astype(int).astype(str), format="%Y%m%d"),
        year=df["season"],
        tour_level=df["tourney_level"].where(~is_qualifying, "Q"),
        surface=df["surface"].fillna("Unknown"),
        round_order=df["round"].map(ROUND_ORDER),
        winner_hand=df["winner_hand"].where(df["winner_hand"].isin(["R", "L"])),
        loser_hand=df["loser_hand"].where(df["loser_hand"].isin(["R", "L"])),
        winner_id=df["winner_id"].astype(int),
        loser_id=df["loser_id"].astype(int),
    )
    if df["match_id"].duplicated().any():
        raise ValueError("match_id is not unique after cleaning")

    return (df.sort_values(["match_date", "tourney_id", "round_order", "match_num"], kind="stable")
              .reset_index(drop=True))

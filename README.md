# Tennis Match Prediction

Pre-match winner prediction for ATP singles matches using machine learning, reproducing
*Tennis Match Prediction using Machine Learning* (A. K. Amudan, Stanford CS229).

**Course:** UE24CS352A – Machine Learning, Mini-Project

## Setup and run

### Requirements

- **Python 3.13** (developed and tested on 3.13.0). Use a plain `venv`, not a conda base environment.
- **macOS or Linux** (tested on macOS 27, Apple Silicon).
- **Disk:** about 90 MB of data (40 MB raw CSVs plus 48 MB `pairs.parquet`) and about
  700 MB for the virtual environment.
- **Time** (measured on a fresh clone, MacBook Air, Apple Silicon): `pip install` 25 s with a
  warm pip cache (longer the first time, when every package has to download), data download
  30 s, building `pairs.parquet` 23 s, tests 65 s, EDA notebook export 7 s.

### Setup

```bash
git clone https://github.com/harish-rameshkumar/tennis-match-prediction.git
cd tennis-match-prediction
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins every package version and also installs this project (`-e .`), so
`import tennis` works from scripts, tests and notebooks. Check with `python --version`
that the venv uses Python 3.13.

### Run steps

Run these from the repository root with the venv activated, in this order.

**1. Download and verify the data**

```bash
python scripts/download_data.py
```

Downloads 43 CSVs (about 40 MB) into `data/raw/` and checks each one against
`data/SHA256SUMS`. It prints one `downloaded …` line per file, then
`all 43 files present and verified`. Running it again skips files that are already there.

**2. Build the pairs table**

```bash
python -m tennis.dataset
```

Cleans the matches and computes every feature. Expect
`wrote …/data/processed/pairs.parquet: 393560 rows x 77 columns (196780 matches)`.

**3. Run the tests**

```bash
pytest
```

Expect `29 passed`. The tests need steps 1 and 2 first; without the data they are reported as
skipped, not passed.

**4. Open the EDA notebook**

```bash
jupyter nbconvert --to html --execute notebooks/01_eda.ipynb
open notebooks/01_eda.html          # Linux: xdg-open notebooks/01_eda.html
```

Re-runs the notebook from the raw data and writes `notebooks/01_eda.html` (gitignored),
then opens it in the browser. The committed notebook already has its figures, so you can
also just open `notebooks/01_eda.ipynb` in VS Code or on GitHub. In VS Code, pick the
`.venv` kernel.

**Shortcuts:** the [Makefile](Makefile) wraps the same commands. Run it with the venv activated.

| Command | Same as |
| --- | --- |
| `make data` | `python scripts/download_data.py` |
| `make features` | `make data`, then `python -m tennis.dataset` |
| `make test` | `pytest` |
| `make clean` | deletes `data/processed/`, caches and the exported notebook HTML (keeps `data/raw/`) |

### Models and demo

> **TODO (Jayant):** fill in the real commands when the code lands. Nothing below exists yet.

| Stage | Command | What it does / expected output |
| --- | --- | --- |
| Train and evaluate | _TODO_ | _TODO_ |
| Streamlit app | _TODO_ | _TODO_ |
| CLI demo | _TODO_ | _TODO_ |
| Results notebook (`notebooks/02_results.ipynb`) | _TODO_ | _TODO_ |

The models use `get_train_test` ([Dataset interface](#dataset-interface)), and the demo uses
`build_match_rows` ([Inference interface](#inference-interface-srctennisinferencepy)).

### Repository layout

```
tennis-match-prediction/
├── README.md
├── Makefile                 # shortcuts: data, features, test, clean
├── requirements.txt         # pinned dependencies + this package (-e .)
├── pyproject.toml           # package metadata for src/tennis and pytest settings
├── data/
│   ├── SHA256SUMS           # checksums of the 43 raw CSVs (committed)
│   ├── raw/                 # downloaded CSVs (gitignored)
│   └── processed/           # pairs.parquet (gitignored)
├── scripts/
│   └── download_data.py     # pinned, checksum-verified download
├── src/tennis/
│   ├── config.py            # paths, year ranges, mirror commit
│   ├── load.py              # read the raw CSVs
│   ├── clean.py             # cleaning rules
│   ├── features.py          # history-based features (form, head-to-head, serve, Elo)
│   ├── dataset.py           # pairs.parquet, FEATURE_COLUMNS, get_train_test
│   └── inference.py         # feature rows for a new match (used by the demo)
├── tests/                   # dataset, leakage, symmetry and inference tests
└── notebooks/
    └── 01_eda.ipynb         # exploratory data analysis
```

### Team

| Name | SRN | Owns |
| --- | --- | --- |
| Harish Ramesh Kumar | PES1UG24AM111 | data pipeline, features, inference, EDA |
| Jayant Mahishi | PES1UG24AM122 | models, evaluation, `predict.py`, Streamlit app, results notebook |

### Data source and licence

The data is Jeff Sackmann's [`tennis_atp`](https://github.com/JeffSackmann/tennis_atp) ATP match
data, released under **CC BY-NC-SA 4.0** (attribution, non-commercial, share-alike). The original
repository is no longer available on GitHub, so we download it from the archival mirror
[`Aneeshers/tennis-sackmann-archive`](https://github.com/Aneeshers/tennis-sackmann-archive),
pinned to commit `83733587353df8a41f2fd4f516147d5aa83f5a8d`. Every file we use matches the
mirror's own record of that commit (git blob hashes), and the download is checked against
[data/SHA256SUMS](data/SHA256SUMS).

| Files | Years | Used for |
| --- | --- | --- |
| `atp_matches_YYYY.csv` | 1997–2024 | ATP main tour (incl. Davis Cup) |
| `atp_matches_qual_chall_YYYY.csv` | 2010–2024 | Challengers and ATP qualifying |

## Data

### Cleaning rules ([src/tennis/clean.py](src/tennis/clean.py))

1. Drop rows missing `winner_id`, `loser_id`, `tourney_date`, `winner_rank` or `loser_rank`,
   and rows where winner and loser are the same player.
2. Exclude walkovers (`W/O`, `Walkover` in the score). Rows with no score are also dropped:
   they have no match statistics and can't be told apart from walkovers.
3. Keep retirements (`RET`), defaults (`DEF`) and abandoned matches. The recorded winner stands.
4. Qualifying rounds (`Q1`–`Q4`) get `tour_level = "Q"`. Otherwise `tour_level` is Sackmann's
   `tourney_level`.
5. Missing surface becomes `"Unknown"`. Hand codes other than `R`/`L` are treated as missing.

This keeps 196,780 of 205,168 matches. 7,453 rows are dropped for a missing ID, date or rank,
3 for self-matches and 932 as walkovers or score-less rows.

## Dataset interface

`data/processed/pairs.parquet` has **one row per (player A, player B, match)**. Every match
appears twice: once with A = winner (`label = 1`) and once with the players swapped
(`label = 0`). The two rows are adjacent, and the table is in chronological order.

### Columns

| Column | Type | Meaning |
| --- | --- | --- |
| `match_id` | str | `"{tourney_id}-{match_num}"`, unique per match, shared by its two rows |
| `match_date` | datetime | Tournament **start** date (the data has no per-match date) |
| `year` | int | Season (the year in the source file name). Early-January events can have a December `match_date`. |
| `tour_level` | str | `G` Grand Slam, `M` Masters, `A` ATP 250/500, `F` Tour Finals, `D` Davis Cup, `O` Olympics, `C` Challenger, `Q` qualifying |
| `surface` | str | `Hard`, `Clay`, `Grass`, `Carpet`, `Unknown` |
| `label` | int | 1 if player A won, 0 if player A lost |
| `player_a_id`, `player_b_id` | int | Sackmann player IDs |
| `tourney_name`, `round`, `player_a_name`, `player_b_name` | str | Metadata for display and error analysis. **Not features.** |

**Features** (`tennis.dataset.FEATURE_COLUMNS`) are raw values, not differences, as in the paper:

| Column | Meaning | Missing in raw data |
| --- | --- | --- |
| `rank_a`, `rank_b` | ATP rank at the tournament date | never (rows without rank are dropped) |
| `rank_points_a`, `rank_points_b` | ATP ranking points | median-filled |
| `height_a`, `height_b` | height in cm | median-filled |
| `hand_a`, `hand_b` | 1 = left-handed, 0 = right-handed | median-filled (→ 0) |
| `age_a`, `age_b` | age in years at the tournament date | median-filled |
| `<feature>_missing` | 1 if that value was missing and filled in, else 0 (for rank_points, height, hand, age) | — |

#### History-based features ([src/tennis/features.py](src/tennis/features.py))

These are computed in one chronological pass over **all** levels, including Challenger,
qualifying and Davis Cup. The **history** of a match is every match from a tournament with
a strictly earlier start date, plus earlier rounds of the same tournament. Same-round
matches, later rounds and other tournaments that start on the same date are never used.
[tests/test_no_leakage.py](tests/test_no_leakage.py) checks every feature against a
brute-force rebuild from filtered data.

Naming conventions:
- `diff_<f>` = value for A − value for B. The mirrored row holds exactly the negative.
- `<f>_a` / `<f>_b` are per-player flags. They swap in the mirrored row.
- A player with no history gets 0 for every value, and `has_history_<side>` = 0.

| Column | Per-player value behind the difference / flag |
| --- | --- |
| `has_history_a`, `has_history_b` | 1 if the player has at least one match in their history |
| `diff_career_wins`, `diff_career_losses`, `diff_career_win_rate` | career record |
| `diff_last{5,10,15,25}_played`, `diff_last{5,10,15,25}_win_rate` | matches counted (≤ window) and win rate over the last N matches |
| `diff_recent30_wins`, `diff_recent30_losses` | record in tournaments starting in the 30 days before this tournament's start |
| `diff_surface_wins`, `diff_surface_losses`, `diff_surface_win_rate` | career record on this match's surface |
| `diff_surface_last{5,10,15,25}_played`, `diff_surface_last{5,10,15,25}_win_rate` | the same last-N form, on this surface only |
| `diff_tourney_wins`, `diff_tourney_losses` | record at this event in any year, including earlier rounds this year |
| `diff_h2h_wins`, `diff_h2h_surface_wins` | wins against this opponent, overall and on this surface |
| `diff_common_opp_win_rate` | win rate against opponents both players have faced (0 if there are none) |
| `h2h_played`, `h2h_surface_played` | *symmetric:* previous meetings of the two players, overall and on this surface |
| `common_opp_count` | *symmetric:* number of opponents both players have faced |
| `has_serve_stats_a`, `has_serve_stats_b` | 1 if the player has at least one previous match with full serve stats |
| `diff_first_in_pct`, `diff_first_won_pct`, `diff_second_won_pct` | career % of first serves in, and of first- and second-serve points won |
| `diff_aces_per_match`, `diff_dfs_per_match` | career aces and double faults per match |
| `diff_bp_saved_pct`, `diff_bp_saved_per_match`, `diff_bp_faced_per_match` | career break points saved %, and saved and faced per match |
| `diff_sv_games_per_match` | career service games per match |
| `diff_elo`, `diff_surface_elo` | Elo rating before the match, overall and on this surface |

Serve averages use previous matches only, and only those with all nine serve stats present
and `svpt > 0` (about 96% of matches). The current match's stats are added only after its
features have been computed.

Elo ratings start at 1500 and use the FiveThirtyEight-style K-factor K = 250 / (n + 5)^0.4,
where n is the number of matches already rated. Surface Elo keeps a separate rating and
match count for each surface. Every level counts the same: there is no Grand Slam multiplier
and no decay for inactivity.

In the parquet file the basic feature columns (rank points, height, hand, age) still contain NaN; history-based columns never do. `get_train_test` fills them with
the **training-set median**, pooled over the `_a` and `_b` columns so the two rows of a match stay mirror images.
This way no statistics from the test years leak into training.

### `get_train_test(test_year=2019, test_end_year=None, pairs=None) -> (train, test)`

```python
from tennis.dataset import get_train_test, FEATURE_COLUMNS

train, test = get_train_test()                 # paper setup: test on 2019
train, test = get_train_test(2023, 2024)       # robustness check: train <= 2022, test 2023-24
X_train, y_train = train[FEATURE_COLUMNS], train["label"]
```

Contract:
- **train** contains every pair with `year < test_year`, at every tour level, including
  Challenger, qualifying and Davis Cup.
- **test** contains only ATP main-tour pairs (`tour_level` in `G, M, A, F`) with
  `test_year <= year <= test_end_year`. `test_end_year` defaults to `test_year`.
- Both rows of a match are always in the same split, because they share `match_id` and `year`.
- Both DataFrames are sorted chronologically, have a fresh `RangeIndex`, and their
  `FEATURE_COLUMNS` contain no NaN.
- `pairs` is optional. Pass an already-loaded table to avoid re-reading the parquet file.

To evaluate one prediction per match, combine the two rows: P(A wins) =
½·(p(A,B) + 1 − p(B,A)).

| Split | Train | Test |
| --- | --- | --- |
| `get_train_test()` | 263,124 rows (131,562 matches, 1997–2018) | 5,194 rows (2,597 matches) |
| `get_train_test(2023, 2024)` | 339,630 rows (169,815 matches, 1997–2022) | 10,928 rows (5,464 matches) |

### Inference interface ([src/tennis/inference.py](src/tennis/inference.py))

`build_match_rows(...)` returns the feature rows for a match that isn't in the data, built with
the **same code** as the training rows: `replay` and `match_features` from `features.py`, and
`build_pairs` and `fill_missing` from `dataset.py`.

```python
from tennis.dataset import FEATURE_COLUMNS, get_train_test, train_medians
from tennis.inference import History, build_match_rows, combine, load_history

train, _ = get_train_test()                      # whatever the model was trained on
medians = train_medians(train)                   # same fill values as get_train_test
history = History(load_history())                # all data as history (~4 s); build once and reuse

rows = build_match_rows("Novak Djokovic", "Carlos Alcaraz", "Grass", "2025-07-01", history,
                        tournament="Wimbledon", medians=medians)
p_ab, p_ba = model.predict_proba(rows[FEATURE_COLUMNS])[:, 1]
p_djokovic = combine(p_ab, p_ba)                 # ½·(p(A,B) + 1 − p(B,A))
```

`build_match_rows(player_a, player_b, surface, date, history, *, tournament=None, tour_level=None, attributes=None, medians=None)`

**Returns** a DataFrame with 2 rows: row 0 is (A, B) and row 1 is (B, A).
- Its columns are those of `pairs.parquet` without `label` (`tennis.inference.OUTPUT_COLUMNS`), so
  `rows[FEATURE_COLUMNS]` feeds straight into a model trained on `get_train_test` output.
- Every `diff_*` value in row 1 is the exact negative of row 0.
- `build_match_row(...)` takes the same arguments and returns only row 0.

**Arguments:**

| Argument | Meaning |
| --- | --- |
| `player_a`, `player_b` | Player id, or a name. Names are case- and accent-insensitive, and a unique partial name works (`"carlos alcaraz"`, `"Federer"`). An unknown name raises `UnknownPlayerError`, whose `.suggestions` holds close matches (`"Novak Djokovc"` → Novak Djokovic). A name that fits several players raises `AmbiguousPlayerError` with `.candidates` (`"Murray"` → Andy, Jamie, Jack). |
| `surface` | `"Hard"`, `"Clay"`, `"Grass"` or `"Carpet"` |
| `date` | Match date. History is every match **strictly before** this date. |
| `history` | A `History`, or a DataFrame of cleaned matches (`load_history()`), in which case `History(matches, before=date)` is built (~1–2 s). `History(matches)` with no cutoff uses all data and suits predictions after the data ends (2024-12-18). A `History` whose cutoff is later than `date` raises `ValueError`, because it would include future matches. |
| `tournament` | `tourney_id` or tournament name (latest edition). It drives the `tourney_*` features, which are 0 without it. |
| `attributes` | Overrides such as `{"rank_a": 2, "rank_points_b": 9000, "hand_a": "L", "height_b": 183, "age_a": 37.1}` |
| `medians` | From `train_medians(train)`. NaNs are then filled exactly as in `get_train_test`. Without it, rows keep NaN plus the `*_missing` flags, like `pairs.parquet`. |

**Defaults for rank, points, height, hand and age** come from each player's latest match
before `date`, with age advanced to `date`. A rank is required for both players: if a player
has no earlier match, pass `attributes={"rank_a": ...}`. A player with no history gets
`has_history = 0` and zeros, exactly as in the pipeline.

**Verified:** [tests/test_inference.py](tests/test_inference.py) rebuilds 20 real matches from the
history before their date and gets exactly the stored `pairs.parquet` rows. The test uses
matches where neither player had played an earlier round of the same tournament: for later
rounds, the pipeline also counts the earlier rounds of that tournament, which a "before this
date" query can't see.

## Results

| Test set | Higher-ranked player wins |
| --- | --- |
| 2019 | 61.3% |
| 2023–24 | 63.5% |

## Data notes

- **Rank baseline:** on our 2019 test set, the higher-ranked player wins **61.3%** of matches.
  The paper reports 65.5%. Earlier years agree with the paper (2015–2018: 67.9%, 67.2%,
  65.0%, 63.6%), so 2019 is a low year rather than a data problem. The paper probably used
  only part of the 2019 season. Compare models against 61.3%, not 65.5%.
- **ATP main tour:** Grand Slams (`G`), Masters (`M`), ATP 250/500 (`A`) and the Tour
  Finals (`F`). Davis Cup, Olympics, Challenger and qualifying matches are used for training
  only, never as test matches.
- **`match_date`** is the tournament **start** date, because the data has no per-match date.
  Matches in the same tournament share it and are ordered by round. Features computed from
  history must only use earlier tournaments or earlier rounds.
- **Missing values:** rank points, height, hand and age are filled with the training-set median,
  pooled over players A and B, and flagged in a `*_missing` column. Rows with no rank are dropped.
- **Walkovers and score-less rows are excluded** (932 rows): no tennis was played, or we can't
  tell whether it was. Retirements are kept.
- **Spot check:** the 2019 Grand Slam finals in the cleaned data match the real results
  (Djokovic d. Nadal, Nadal d. Thiem, Djokovic d. Federer 13-12(3) in the fifth set, Nadal d. Medvedev).

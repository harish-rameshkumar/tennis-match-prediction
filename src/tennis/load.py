"""Read the raw Sackmann match CSVs into one DataFrame."""

import pandas as pd

from tennis import config


def _read(name: str, source: str, season: int) -> pd.DataFrame:
    path = config.RAW_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python scripts/download_data.py` first")
    df = pd.read_csv(path, low_memory=False)
    return df.assign(source=source, season=season)


def load_raw() -> pd.DataFrame:
    """All ATP main-tour and Challenger/qualifying matches, one row per match.

    Adds two columns to Sackmann's schema:
      source -- "atp" (atp_matches_YYYY.csv) or "qual_chall" (atp_matches_qual_chall_YYYY.csv)
      season -- the YYYY in the file name. A season's first events can start in late
                December of the previous calendar year (e.g. 2019 Brisbane on 2018-12-31).
    """
    frames = [_read(config.atp_file(y), "atp", y) for y in config.ATP_YEARS]
    frames += [_read(config.qual_chall_file(y), "qual_chall", y) for y in config.QUAL_CHALL_YEARS]
    return pd.concat(frames, ignore_index=True)

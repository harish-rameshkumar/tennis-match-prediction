"""Paths, year ranges and data-source settings shared across the project."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PAIRS_PATH = PROCESSED_DIR / "pairs.parquet"
CHECKSUMS_PATH = ROOT / "data" / "SHA256SUMS"

# Jeff Sackmann's tennis_atp repo is no longer on GitHub; we use an archival mirror,
# pinned to one commit so every download is identical (verified against CHECKSUMS_PATH).
MIRROR_REPO = "Aneeshers/tennis-sackmann-archive"
MIRROR_COMMIT = "83733587353df8a41f2fd4f516147d5aa83f5a8d"
MIRROR_SUBDIR = "atp"

# Paper: ATP main tour from 1997 (match stats available), Challengers/qualifying from 2010.
# We extend both to 2024 for the 2023-24 robustness check.
ATP_YEARS = range(1997, 2025)
QUAL_CHALL_YEARS = range(2010, 2025)

DEFAULT_TEST_YEAR = 2019


def atp_file(year: int) -> str:
    return f"atp_matches_{year}.csv"


def qual_chall_file(year: int) -> str:
    return f"atp_matches_qual_chall_{year}.csv"


def raw_files() -> list[str]:
    return [atp_file(y) for y in ATP_YEARS] + [qual_chall_file(y) for y in QUAL_CHALL_YEARS]

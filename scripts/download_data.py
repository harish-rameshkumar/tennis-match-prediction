"""Download the raw ATP match CSVs into data/raw/.

Source: Jeff Sackmann's tennis_atp data (CC BY-NC-SA 4.0), fetched from an archival
mirror pinned to a single commit (see tennis.config). The original GitHub repo is gone.

    python scripts/download_data.py                # download + verify against data/SHA256SUMS
    python scripts/download_data.py --write-sums   # (re)generate data/SHA256SUMS
"""

import argparse
import hashlib
import shutil
import ssl
import sys
import urllib.request

import certifi

from tennis import config

# python.org builds on macOS ship without root certificates; use certifi's bundle.
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(name: str) -> None:
    dest = config.RAW_DIR / name
    if dest.exists():
        return
    url = (f"https://raw.githubusercontent.com/{config.MIRROR_REPO}/"
           f"{config.MIRROR_COMMIT}/{config.MIRROR_SUBDIR}/{name}")
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url, context=SSL_CONTEXT) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out)
    tmp.rename(dest)
    print(f"downloaded {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write-sums", action="store_true",
                        help="write data/SHA256SUMS instead of verifying against it")
    args = parser.parse_args()

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    names = config.raw_files()
    for name in names:
        download(name)

    sums = {name: sha256(config.RAW_DIR / name) for name in names}
    if args.write_sums:
        config.CHECKSUMS_PATH.write_text("".join(f"{h}  {n}\n" for n, h in sums.items()))
        print(f"wrote {config.CHECKSUMS_PATH}")
        return 0

    expected = dict(line.split()[::-1] for line in config.CHECKSUMS_PATH.read_text().splitlines())
    bad = [n for n in names if expected.get(n) != sums[n]]
    if bad:
        print(f"checksum mismatch for {len(bad)} file(s): {', '.join(bad)}", file=sys.stderr)
        return 1
    print(f"all {len(names)} files present and verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())

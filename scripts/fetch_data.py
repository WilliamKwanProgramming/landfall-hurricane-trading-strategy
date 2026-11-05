#!/usr/bin/env python3
"""Download the public hurricane track data into the git-ignored data/raw/ directory.

Default (`--source mirrors`): two GitHub mirrors of NOAA NHC HURDAT2, pinned to a commit and
checked against SHA-256 checksums. These are the files used for every result in this repository.

- timtrice/HURDAT `data/AL.rda` (R data file, HURDAT2 1851-2019, MIT licence).
- tidyverse/dplyr `data-raw/storms.csv` (made from the NHC file hurdat2-1851-2024-040425.txt,
  1975-2024, with the processing described in landfall.data.hurdat2.load_dplyr_mirror).

`--source nhc` downloads the official HURDAT2 text file from www.nhc.noaa.gov instead. That route
was NOT exercised in the sandbox where this repository was built (the NHC site was not reachable
from it), so the file has no pinned checksum; the script prints the SHA-256 it computed so that it
can be recorded. Read the file with landfall.data.hurdat2.read_hurdat2.

Usage: python scripts/fetch_data.py [--source mirrors|nhc] [--nhc-url URL] [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys
import urllib.request

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from landfall.data import paths  # noqa: E402

MIRRORS = [
    (paths.RDA_MIRROR,
     "https://raw.githubusercontent.com/timtrice/HURDAT/"
     "ecc8a6eaaf4c91b6943e8baf3597206fe591c76d/data/AL.rda",
     "08457d2ab2e936c276e11a87916c0b9c1b043cb01fe6cef735ca3a1c1ea62b75"),
    (paths.DPLYR_MIRROR,
     "https://raw.githubusercontent.com/tidyverse/dplyr/"
     "819c99f7284b3a413b391c60bea49ae7bfb8bafc/data-raw/storms.csv",
     "ee46bd48b49ebd586da86192678de412fb84dfc6534d250863b8fc8f4ff16ebd"),
]
NHC_URL = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2024-040425.txt"


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(url: str, dest: pathlib.Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as fh:
        fh.write(r.read())
    tmp.replace(dest)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", choices=["mirrors", "nhc"], default="mirrors")
    ap.add_argument("--nhc-url", default=NHC_URL,
                    help="official HURDAT2 file (check https://www.nhc.noaa.gov/data/#hurdat)")
    ap.add_argument("--force", action="store_true", help="download even if a valid file exists")
    args = ap.parse_args()
    out = paths.raw_dir()
    out.mkdir(parents=True, exist_ok=True)

    if args.source == "nhc":
        dest = paths.raw_file(paths.NHC_FILE)
        download(args.nhc_url, dest)
        print(f"wrote {dest.relative_to(REPO) if dest.is_relative_to(REPO) else dest}")
        print(f"sha256 {sha256(dest)} (not pinned: record it)")
        return 0

    status = 0
    for name, url, digest in MIRRORS:
        dest = paths.raw_file(name)
        if dest.exists() and not args.force and sha256(dest) == digest:
            print(f"ok      {name} (already present, checksum verified)")
            continue
        download(url, dest)
        got = sha256(dest)
        if got != digest:
            print(f"FAILED  {name}: sha256 {got}, expected {digest}", file=sys.stderr)
            dest.unlink()
            status = 1
        else:
            print(f"fetched {name} (checksum verified)")
    return status


if __name__ == "__main__":
    sys.exit(main())

"""Location of downloaded data (git-ignored `data/` at the repository root by default)."""

from __future__ import annotations

import os
import pathlib

RDA_MIRROR = "hurdat2_al_1851_2019_timtrice.rda"
DPLYR_MIRROR = "hurdat2_al_1975_2024_dplyr_storms.csv"
NHC_FILE = "hurdat2_al_nhc.txt"


def data_dir() -> pathlib.Path:
    """`$LANDFALL_DATA` if set, otherwise `<repository>/data`."""
    env = os.environ.get("LANDFALL_DATA")
    if env:
        return pathlib.Path(env)
    return pathlib.Path(__file__).resolve().parents[3] / "data"


def raw_dir() -> pathlib.Path:
    return data_dir() / "raw"


def raw_file(name: str) -> pathlib.Path:
    return raw_dir() / name

import os
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
FIXTURES = HERE / "fixtures"

# The C++ extension is built by CMake into build/cpp (see cpp/CMakeLists.txt); a directory in
# LANDFALL_CPP_DIR takes precedence (inserted last, so it ends up first on sys.path).
for p in (str(REPO / "build" / "cpp"), os.environ.get("LANDFALL_CPP_DIR", "")):
    if p and p not in sys.path and pathlib.Path(p).is_dir():
        sys.path.insert(0, p)

from landfall.data import paths  # noqa: E402


def need_raw(name: str) -> pathlib.Path:
    """Path of a downloaded data file; skip the test with a clear message if it is absent."""
    f = paths.raw_file(name)
    if not f.exists():
        pytest.skip(f"{f} not found: run `python scripts/fetch_data.py` first")
    return f


@pytest.fixture(scope="session")
def rda_tracks():
    from landfall.data import hurdat2

    df, _ = hurdat2.load_rda_mirror(need_raw(paths.RDA_MIRROR))
    return df


@pytest.fixture(scope="session")
def dplyr_tracks():
    from landfall.data import hurdat2

    df, _ = hurdat2.load_dplyr_mirror(need_raw(paths.DPLYR_MIRROR))
    return df


@pytest.fixture(scope="session")
def land_grid():
    from landfall.geo import atlantic_land_grid

    return atlantic_land_grid()


def cpp_module():
    """Import landfall_cpp; skip if it is not built, unless LANDFALL_REQUIRE_CPP=1 (CI)."""
    try:
        import landfall_cpp
    except ImportError:
        if os.environ.get("LANDFALL_REQUIRE_CPP") == "1":
            raise
        pytest.skip("landfall_cpp not built: cmake -S . -B build && cmake --build build")
    return landfall_cpp

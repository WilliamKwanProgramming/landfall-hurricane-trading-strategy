"""HURDAT2 parser on a fixture written from the NHC format document, and the two mirrors."""

import numpy as np
import pandas as pd
import pytest
from conftest import FIXTURES

from landfall.data import hurdat2


@pytest.fixture(scope="module")
def sample():
    return hurdat2.parse_hurdat2((FIXTURES / "hurdat2_sample.txt").read_text())


def test_counts_and_schema(sample):
    assert list(sample.columns) == hurdat2.TRACK_COLUMNS
    assert sample.groupby("storm_id").size().to_dict() == {
        "AL992011": 6, "AL991967": 3, "AL982022": 2}
    assert set(sample["season"]) == {1967, 2011, 2022}


def test_hemispheres_and_signs(sample):
    s = sample[sample.storm_id == "AL991967"].reset_index(drop=True)
    assert s.loc[0, "lon"] == -35.0 and s.loc[1, "lon"] == -0.5
    assert s.loc[2, "lat"] == -1.0 and s.loc[2, "lon"] == 2.1


def test_missing_values(sample):
    s = sample[sample.storm_id == "AL991967"].reset_index(drop=True)
    assert np.isnan(s.loc[0, "vmax_kt"]) and np.isnan(s.loc[0, "mslp_mb"])
    assert s.loc[2, "vmax_kt"] == 25 and s.loc[2, "mslp_mb"] == 1010
    t = sample[sample.storm_id == "AL992011"].reset_index(drop=True)
    assert t.loc[3, ["r34_ne", "r64_nw"]].isna().all()  # -999: not analysed
    assert t.loc[0, "r34_sw"] == 0  # 0: no 34-kt wind in that quadrant, not missing


def test_record_identifiers_and_time(sample):
    t = sample[sample.storm_id == "AL992011"].reset_index(drop=True)
    assert t["record"].tolist() == ["", "", "I", "L", "", ""]
    assert t.loc[3, "time"] == pd.Timestamp("2011-08-24 15:30")
    assert t["status"].tolist() == ["TS", "HU", "HU", "HU", "HU", "EX"]


def test_radii_quadrant_order(sample):
    t = sample[sample.storm_id == "AL992011"].iloc[2]
    assert [t[c] for c in hurdat2.RADII] == [150, 120, 60, 90, 60, 50, 30, 40, 30, 25, 20, 25]


def test_twenty_one_field_format(sample):
    t = sample[sample.storm_id == "AL982022"].reset_index(drop=True)
    assert t.loc[0, "rmw_nm"] == 15 and np.isnan(t.loc[1, "rmw_nm"])
    assert np.isnan(sample[sample.storm_id == "AL992011"]["rmw_nm"]).all()


def test_malformed_files_rejected():
    text = (FIXTURES / "hurdat2_sample.txt").read_text()
    with pytest.raises(ValueError, match="announces"):
        hurdat2.parse_hurdat2(text.replace("TESTONE,      6,", "TESTONE,      7,"))
    with pytest.raises(ValueError, match="bad longitude"):
        hurdat2.parse_hurdat2(text.replace("76.8W", "76.8X"))
    with pytest.raises(ValueError, match="unknown status"):
        hurdat2.parse_hurdat2(text.replace(" TS, 25.4N", " XX, 25.4N"))


def test_synoptic_filter(sample):
    s = hurdat2.synoptic(sample)
    assert (s["time"].dt.hour % 6 == 0).all() and (s["time"].dt.minute == 0).all()
    assert len(s) == len(sample) - 2  # the 1530 and 1905 entries


# ---- mirrors (skip without data) ------------------------------------------------------------

def test_rda_mirror_shape_and_repair(rda_tracks):
    from landfall.data import paths

    _, notes = hurdat2.load_rda_mirror(paths.raw_file(paths.RDA_MIRROR))
    assert not any("dropped" in n for n in notes)
    assert len(rda_tracks) == 51840
    assert rda_tracks["storm_id"].notna().all() and rda_tracks["lon"].notna().all()

    def lon_at(sid, t):
        return rda_tracks[(rda_tracks.storm_id == sid)
                          & (rda_tracks.time == pd.Timestamp(t))]["lon"].item()

    assert lon_at("AL041980", "1980-08-10 18:00") == -99.0  # Allen; dplyr mirror: -99
    assert lon_at("AL051986", "1986-08-26 12:00") == 0.0  # Charley; dplyr mirror: -0
    first = rda_tracks[rda_tracks.storm_id == "AL011851"]
    assert len(first) == 14 and first["time"].min() == pd.Timestamp("1851-06-25")
    assert (rda_tracks["record"] == "L").sum() == 1016
    assert rda_tracks["lon"].between(-180, 180).all()


def test_dplyr_mirror_shape(dplyr_tracks):
    assert len(dplyr_tracks) == 20778
    assert dplyr_tracks["season"].min() == 1975 and dplyr_tracks["season"].max() == 2024
    assert dplyr_tracks["record"].isna().all()  # the mirror has no record identifiers
    zeta = dplyr_tracks[dplyr_tracks.storm_id == "ZETA_2005"]
    assert zeta["time"].max().year == 2006  # re-joined across the year boundary


def test_mirrors_agree_where_both_have_data(rda_tracks, dplyr_tracks):
    r = hurdat2.compare_sources(rda_tracks, dplyr_tracks)
    # Every storm of the dplyr mirror is in the rda mirror; storms missing from dplyr are
    # explained by its filters (all pressures missing, or never a TD/TS/HU).
    assert r["storms_only_dplyr"] == []
    assert len(r["storms_only_rda"]) == (r["storms_only_rda_all_pressure_missing"]
                                         + r["storms_only_rda_no_td_ts_hu"])
    assert r["fixes_matched"] == r["fixes_dplyr"]
    # Positions agree except for a handful of later best-track revisions (reported, not hidden).
    assert r["diff_lat"] + r["diff_lon"] <= 0.001 * r["fixes_matched"]
    assert r["diff_vmax_kt"] <= 0.005 * r["fixes_matched"]

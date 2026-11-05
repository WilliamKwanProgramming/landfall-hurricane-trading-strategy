"""The best-track table used by the experiments: rda mirror up to 2019, dplyr mirror after.

The rda mirror (HURDAT2 1851-2019) keeps record identifiers, minutes and wind radii, so it is
used for every season it covers; seasons 2020-2024 come from the dplyr mirror (made from the
NHC file released in April 2025), which lacks them (see landfall.data.hurdat2).
"""

from __future__ import annotations

import pandas as pd

from landfall.data import hurdat2, paths

SPLIT_SEASON = 2020


def load_combined_tracks() -> tuple[pd.DataFrame, list[str]]:
    rda, n1 = hurdat2.load_rda_mirror(paths.raw_file(paths.RDA_MIRROR))
    dpl, n2 = hurdat2.load_dplyr_mirror(paths.raw_file(paths.DPLYR_MIRROR))
    df = pd.concat([rda[rda.season < SPLIT_SEASON], dpl[dpl.season >= SPLIT_SEASON]],
                   ignore_index=True)
    df["record"] = df["record"].astype("string")
    notes = [f"rda: {n}" for n in n1] + [f"dplyr: {n}" for n in n2]
    return df.sort_values(["season", "storm_id", "time"], kind="stable").reset_index(
        drop=True), notes

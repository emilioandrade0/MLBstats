"""Park × offense interaction features.

Error analysis #4 showed hitter-friendly venues (Coors, Camden, Nationals,
Great American) account for many blown picks. The model treats `park_runs_factor`
and `off_xwoba_l30` as independent features, but in reality they INTERACT:
- A hot offense at Coors is different from the same offense at Citi Field
- A weak offense at a pitcher's park is doubly limited
- An ERA-1.50 pitcher at Coors is not the same as the same pitcher at Petco

LightGBM does see interactions in tree splits, but explicit interaction
features can help when training data per (park, offense profile) cell is sparse.

Features (per game, per side, then diffed):
  park_off_xwoba       = park_runs_factor * off_xwoba_l30
  park_off_barrel      = park_runs_factor * off_barrel_rate_l30
  park_def_xwoba       = park_runs_factor * def_xwoba_l30
  park_starter_xwoba   = park_runs_factor * starter_xwoba_l15  (starter shielded by park?)
  park_run_diff        = park_runs_factor * run_diff_l10

These get added to train.parquet during build.py. Computed in-line because
they're just multiplications of existing columns.

This module just documents the intent — the work happens in features/build.py.
"""

INTERACTION_PAIRS = [
    ("park_runs_factor", "off_xwoba_l30",        "park_off_xwoba"),
    ("park_runs_factor", "off_barrel_rate_l30",  "park_off_barrel"),
    ("park_runs_factor", "def_xwoba_l30",        "park_def_xwoba"),
    ("park_runs_factor", "starter_xwoba_l15",    "park_starter_xwoba"),
    ("park_runs_factor", "run_diff_l10",         "park_run_diff"),
]

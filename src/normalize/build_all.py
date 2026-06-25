"""Run every normalizer in the right order.

Order matters: games.parquet must exist before xref.parquet.

Usage:
    python -m src.normalize.build_all
    python -m src.normalize.build_all games team_box pitches    # subset
"""
from __future__ import annotations

import sys

from . import games, team_box, player_box, plays, pitches, odds, win_probability, xref

STEPS = {
    "games": games.build,
    "team_box": team_box.build,
    "player_box": player_box.build,
    "plays": plays.build,
    "pitches": pitches.build,
    "odds": odds.build,
    "win_probability": win_probability.build,
    "xref": xref.build,  # depends on games
}


def main() -> None:
    selected = sys.argv[1:] or list(STEPS)
    for name in selected:
        if name not in STEPS:
            print(f"unknown step: {name}; valid: {list(STEPS)}")
            sys.exit(1)
        print(f"\n=== {name} ===")
        STEPS[name]()


if __name__ == "__main__":
    main()

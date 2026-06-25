"""Filesystem layout for the raw ingest.

data/
  raw/
    scoreboard/YYYY/YYYYMMDD.json
    summary/YYYY/{event_id}.json
    odds/YYYY/{event_id}.json
    probabilities/YYYY/{event_id}.json
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson

ROOT = Path(__file__).resolve().parents[1] / "data" / "raw"


def _path(kind: str, year: int, name: str) -> Path:
    p = ROOT / kind / str(year) / f"{name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save(kind: str, year: int, name: str, payload: dict[str, Any]) -> Path:
    p = _path(kind, year, name)
    p.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    return p


def exists(kind: str, year: int, name: str) -> bool:
    return _path(kind, year, name).exists()


def load(kind: str, year: int, name: str) -> dict[str, Any]:
    return orjson.loads(_path(kind, year, name).read_bytes())

"""Path resolver adaptado para vivir dentro del repo unificado STRIKECAST.

Los parquets NFL se leen/escriben en ``data/processed/nfl/`` en la raíz del
repo, no dentro de la app original ``StrikeCast NFL/data/``. Así el CI puede
retrainar sin depender de esa carpeta (que sigue siendo local de 650 MB).
"""
from __future__ import annotations

import os
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / ".git").exists() or (parent / "requirements.txt").exists():
            return parent
    return start


ROOT = Path(os.environ.get("STRIKECAST_ROOT") or _find_repo_root(Path(__file__).resolve()))
DATA = ROOT / "data"
RAW = DATA / "raw" / "nfl"
PROCESSED = DATA / "processed" / "nfl"
MODELS = DATA / "models" / "nfl"
LOGS = ROOT / "logs"

for _p in (RAW, PROCESSED, MODELS, LOGS):
    _p.mkdir(parents=True, exist_ok=True)

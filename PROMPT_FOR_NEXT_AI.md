# Prompt para seguir mejorando STRIKECAST con otra IA

Copia todo lo que está debajo del `---` como el primer mensaje.

---

Estoy trabajando en STRIKECAST, un modelo LightGBM que predice quién gana partidos MLB (win/lose binary). El modelo entrena con walk-forward mensual y expone picks vía FastAPI + frontend HTML.

## Estado actual (2026-07-10)

- **Global acc**: 62.85% (n=8,832)
- **2026 acc backtest**: 59.49% (n=1,412)
- **2026 acc REAL con series double-down** (runtime): ~60.32%
- **vs Mercado 2026**: +3.02pp
- **Meta**: subir 2026 acc (más importante que global — el modelo pega 66-67% en 2023-2024 pero drifta en 2025-2026 porque el mercado se puso al día)

## Disciplina estricta (obligatoria)

1. **Prioridad**: acc > AUC > log_loss.
2. **Solo implementar reglas que:** (a) suben acc 2026 en backtest y (b) no dañan 2023-2024 o 2025.
3. **Todo override nuevo va con gate `season >= 2025`** (o `>= 2026` si el patrón es regime-specific).
4. **Holdout obligatorio**: si un signal tiene signos opuestos entre 2023-2024 y 2025-2026, gate a solo el régimen que funciona.
5. **N mínimo**: al menos 3-5 juegos en 2026 por combo específico, N grande para reglas globales.
6. **Documentar en `docs/rules_backlog.md`**: cada regla activa, retirada o descartada con evidencia.
7. **NO agregar features al modelo** (probado en el pasado, no supera holdout). Solo overrides a `p_home` en runtime.
8. **NO archivos temporales en raíz**: usa scratchpad para scripts de análisis.

## Arquitectura de overrides en `src/serve/api.py`

Todos los sesgos se aplican a `p_home` ANTES del threshold. El flujo:

```python
bias = h2h_bias + pitcher_bias + streak_biases + momentum_biases
p_adj = clip(p_home + bias, 0.001, 0.999)
raw_pick = p_adj >= threshold(season)
final_pick = flip(raw_pick) if flip_reason else raw_pick  # TRAP_* y PICK_DAY_FADE flippean
```

Overrides activos ahora mismo:
- `_winner_threshold` season-aware (2026+ simétrico 0.500)
- `TRAP_PICK_TEAMS = {LAA}`, `TRAP_FADE_TEAMS = {MIL}`, `DAY_TRAP_FADE = {(WSH, 5)}`
- `NIGHT_TRAP_FADE = {DET}` (gate 2026+)
- `PICK_DAY_FADE = {(BOS,6), (HOU,0), (CLE,0), (BAL,1), (BAL,5), (CLE,6)}` (gate 2026+)
- `H2H_PICK_BIAS` (8 pares team vs team)
- `PITCHER_PICK_BIAS` (10 pitchers específicos, gate 2025+)
- `HOME_COLD_STREAK_BIAS +0.05` (local con 3+ derrotas, gate 2026+)
- `AWAY_HOT_STREAK_BIAS +0.07` (visita con 3+ victorias, gate 2026+)
- `HOME_BLOWOUT_MOMENTUM_BIAS +0.07` (local viene de +8 carreras W, gate 2025+)
- `AWAY_COLD_STREAK_BIAS +0.05` (visita con 5+ derrotas, gate 2026+)
- Series double-down (runtime, gate 2026+)
- `_TIER_BANDS` recalibrado a 2025-2026

## Ideas ya PROBADAS Y DESCARTADAS (NO re-explorar)

- Rest-day mismatch (92% same-rest, buckets insuficientes)
- 6 protocolos pattern search (8,280 combos, todos fallan holdout)
- Quintile scans (~200 features, fallan k-fold)
- Team×day sweeps globales (~420 combos, fallan)
- Bullpen fatigue como override (mercado la precia)
- Umpire signals (features del modelo ya lo capturan)
- Model vs Market divergence (holdout falla, signo flippea)
- Series game number (signal existe pero no flippeable — acc >50%)
- Ballpark drift (residuo en coin flip tras DET fade)
- First-time starter this season (no traduce a bias que suba acc)
- Consecutive road games (signal inconsistente cross-year)
- Home hot streak fade (gap no revert en 2026)
- Starter L5 vs L15 residual (no supera holdout)

## Ideas frescas SIN probar (menciónalas antes de saltar a otras)

- Bullpen quality differential extremo (una crushea, otra pena)
- Weather extremes 2026-specific (>100F, high wind con team split)
- Standings position × month (playoff race vs tank)
- Umpire × market band 2026 (interacción específica no probada)
- Manager change effects (buscar equipos que cambiaron)
- Post-IL pitcher return debut
- Team-vs-team recent series carryover
- Model probability × home_team calibration (per team recalibration)
- Doubleheader game 2 (N ~100-150)
- Interleague AL vs NL residual post-DH universal

## Formato de trabajo esperado

1. **Antes de implementar cualquier regla**: hacer análisis + backtest y mostrar tabla de resultados por año (2023, 2024, 2025, 2026).
2. **Preguntar al usuario cuál probar** si hay varios candidatos.
3. **Implementar solo si sube 2026 sin dañar 23-25**. Si no cumple, descartar y pasar a otra.
4. **Actualizar `docs/rules_backlog.md`** después de cada win/descarte.
5. **Backtest final** después de cada cambio con el pipeline completo.
6. **Los `_check.py`, `_out.txt` etc en raíz son temporales** — al terminar cada búsqueda puedes sobreescribir.

## Archivos clave

- `src/serve/api.py` — pipeline de picks + todos los overrides + `_flip_reason` + `_pick_home_from_phome`
- `src/serve/static/strike.html` — badges UI (agregar nuevos siguiendo el patrón)
- `data/models/lgb_cls.pkl` — modelo LightGBM SeedEnsemble (5 seeds), NO retrain aquí
- `data/processed/train.parquet` — data completa con features
- `docs/rules_backlog.md` — registro de todas las reglas probadas

## Herramientas que necesitarás

```python
# Cargar modelo y datos
import pandas as pd, numpy as np, pickle, sys
sys.path.insert(0, '.')
from src.serve.api import (
    _winner_threshold, _flip_reason, _h2h_bias, _pitcher_bias,
    _home_cold_streak_bias, _away_hot_streak_bias,
    _home_blowout_momentum_bias, _away_cold_streak_bias,
)

with open('data/models/lgb_cls.pkl','rb') as f:
    b = pickle.load(f)
model = b['model']

tr = pd.read_parquet('data/processed/train.parquet')
tr = tr[tr['home_score'].notna() & tr['away_score'].notna()].copy()
tr['game_date'] = pd.to_datetime(tr['game_date'])
tr['home_won'] = (tr['home_score'] > tr['away_score']).astype(int)
tr['season'] = tr['game_date'].dt.year
tr['p_home'] = model.predict_proba(tr[b['feature_names']])[:,1]

gm = pd.read_parquet('data/processed/games.parquet')[['game_pk','day_night']]
tr = tr.merge(gm, on='game_pk', how='left')
tr['day_night'] = tr['day_night'].str.lower()
```

Cuando el user pida seguir buscando señales para subir 2026 acc, primero pregunta cuál ángulo probar de la lista "Ideas frescas SIN probar". Corre backtest riguroso con las tablas por año antes de implementar. Sigue la disciplina al pie de la letra.

Empieza confirmando que entiendes la disciplina y preguntándole al user qué ángulo quiere probar primero.

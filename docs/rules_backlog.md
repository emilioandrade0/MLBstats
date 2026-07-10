# Reglas H2H · Estado y Backlog

Registro de todas las reglas H2H probadas — activas, retiradas por inertes,
y retiradas por hacer daño global. Revisar tras cada retrain mensual.

**Última actualización**: 2026-07-09 (barrido exhaustivo de 112 pairs)

---

## Reglas ACTIVAS (8) en `src/serve/api.py::H2H_PICK_BIAS`

Backtest 2026-07-09 sobre 8,819 juegos:
- **Global**: +0.22pp acc vs baseline (61.15% → 61.37%)
- **2026**: +0.64pp acc vs baseline (56.04% → 56.68%)
- **En los 193 pairs cubiertos**: +9.85pp acc (57.0% → 66.8%)
- **En los pairs de 2026 (n=28)**: **+32.14pp acc (46.4% → 78.6%)**

| Par | Sesgo | Δ acc en el par | Flippeos |
|---|---|---|---|
| BAL vs BOS | -0.09 | +16.7pp | 18/24 |
| DET vs CLE | -0.09 | +16.0pp | 16/25 |
| DET vs KC  | +0.10 | +13.0pp | 3/23 |
| WSH vs MIA | -0.09 | +13.0pp | 9/23 |
| PIT vs CIN | +0.10 | +7.4pp  | 10/27 |
| KC vs MIN  | +0.10 | +4.5pp  | 15/22 |
| SEA vs TEX | +0.10 | +4.2pp  | 1/24  |
| PHI vs NYM | +0.08 | +4.0pp  | 1/25  |

---

## Reglas RETIRADAS por INERTES (0 flippeos)

El modelo actual ya predice el lado correcto sin ayuda del sesgo. Guardar para
reactivar si el modelo drifta en próximos retrains.

| Par | Sesgo sugerido | Motivo | Cuándo reactivar |
|---|---|---|---|
| LAD vs COL | +0.12 | Modelo picka LAD con confianza alta (~66% p_home) | Si p_home LAD baja de 0.55 en Dodger Stadium |
| CIN vs MIL | -0.12 | Redundante con TRAP_FADE_TEAMS={MIL} | Si MIL sale de TRAP_FADE_TEAMS |
| AZ vs COL  | +0.10 | Modelo picka AZ con confianza alta | Si p_home AZ baja de 0.60 en Chase Field |

---

## Reglas RETIRADAS por HACER DAÑO GLOBAL

Flippean picks pero en la mayoría de casos flippean AL LADO EQUIVOCADO —
suben en la dirección predicha pero bajan acc real. Alta chance de overfit.
**No reactivar** salvo evidencia contundente en nuevo barrido.

| Par | Sesgo probado | Δ en el par | Motivo del retiro |
|---|---|---|---|
| TEX vs HOU | -0.10 | **-10.7pp** | 9 flippeos pero HOU visita no gana tanto |
| SEA vs HOU | +0.10 | **-8.7pp** | 10 flippeos malos |
| NYY vs BAL | +0.08 | -4.4pp | 2026 gap era artefacto de N=4 |
| AZ vs SF   | +0.09 | -4.0pp | 3 flippeos malos |

---

## Reglas RETIRADAS por NEUTRALES (flippean pero no netean)

Cambian picks pero el net es cero — algunos suben, otros bajan, se cancelan.
Frontera. Vigilar próximo backtest.

| Par | Sesgo probado | Flippeos | Motivo |
|---|---|---|---|
| HOU vs SEA | -0.10 | 4 | Δ 0.00pp neto |
| BOS vs NYY | +0.08 | 10 | Δ 0.00pp neto (mitad y mitad) |
| TB vs TOR  | +0.10 | 2 | Δ 0.00pp neto |

---

## Backlog frío — pares que sobrevivieron holdout pero NO fueron probados

Detectados en el barrido exhaustivo de 112 pares (2026-07-09). Sobreviven
concordancia holdout pero no llegaron al top 15 por N chico o gap moderado.
Candidatos si algún día ampliamos cobertura.

| Par | Gap 23-25 | Gap 2026 | N 2026 |
|---|---|---|---|
| ATL vs MIA (local) | +16.3pp | +4.0pp | 3 |
| SD vs COL (local) | +0.3pp | +37.1pp | 4 |
| MIL vs STL (local) | +7.3pp | +41.6pp | 3 |
| MIA vs PHI (local) | -6.3pp | -19.5pp | 4 |
| BAL vs TB (local) | +3.1pp | +49.7pp | 3 |
| STL vs CIN (local) | +3.3pp | +53.8pp | 3 |
| DET vs CWS (local) | +2.3pp | +43.8pp | 3 |
| ATL vs WSH (local) | -5.4pp | -26.2pp | 3 |
| SF vs SD (local) | -7.9pp | -11.4pp | 3 |
| SEA vs LAA (local) | +6.9pp | +37.7pp | 3 |
| LAA vs SEA (local) | +7.1pp | +24.2pp | 3 |
| BOS vs BAL (local) | -4.4pp | -14.9pp | 3 |

Criterio para promocionar del backlog frío a activo:
- Gap 2026 con misma dirección que gap histórico
- N 2026 >= 4
- Backtest muestra Δ acc positivo en el par sin dañar global

---

## Cómo revisar mensualmente

Después del retrain (día 1 de cada mes), ejecutar el backtest triple:

```python
# analysis/h2h_backtest.py — plantilla
import pandas as pd, pickle, numpy as np, sys
sys.path.insert(0, '.')
from src.serve.api import _winner_threshold, _flip_reason, H2H_PICK_BIAS

with open('data/models/lgb_cls.pkl','rb') as f:
    b = pickle.load(f)
tr = pd.read_parquet('data/processed/train.parquet')
tr = tr[tr['home_score'].notna() & tr['away_score'].notna()].copy()
tr['home_won'] = (tr['home_score'] > tr['away_score']).astype(int)
tr['season'] = pd.to_datetime(tr['game_date']).dt.year
tr['p_home'] = b['model'].predict_proba(tr[b['feature_names']])[:,1]

# Correr backtest con H2H_PICK_BIAS actual vs H2H_OFF (dict vacío)
# Reportar acc global, 2026, y por-par flippeos + Δ
# Cualquier par con Δ 2026 < 0 o gap_2026 < 8pp → retirar al backlog
```

Ver la versión completa que usamos hoy en `_check.py` (temporal, no comitear).

---

## Pitcher-specific pick biases (2026-07-09)

**Estado**: 10 pitchers activos en `PITCHER_PICK_BIAS` de `src/serve/api.py`,
gate 2025+.

**Backtest 2026-07-09** (n=8,819):
- Global: +0.06pp acc
- 2026: +0.07pp acc, +1.27pp en los 79 juegos afectados (73.4% → 74.7%)
- 2023-2024: cero cambio (gate funciona)

### Pitchers activos

| Pitcher | Side | Bias | N total | N 2026 | Gap 26 |
|---|---|---|---|---|---|
| Bryan Woo | H | +0.109 | 39 | 8 | +19.5pp |
| Chris Sale | H | +0.099 | 40 | 8 | +16.1pp |
| Zack Wheeler | H | +0.089 | 52 | 6 | +40.5pp |
| Jacob deGrom | H | +0.087 | 29 | 8 | +19.7pp |
| Yoshinobu Yamamoto | A | -0.085 | 37 | 7 | +15.4pp |
| Aaron Civale | H | -0.084 | 44 | 7 | -19.7pp |
| Cristopher Sánchez | H | +0.068 | 58 | 11 | +22.1pp |
| Jack Kochanowicz | A | +0.067 | 27 | 8 | -16.4pp |
| Landen Roupp | H | -0.063 | 20 | 8 | -15.8pp |
| Kyle Freeland | A | +0.054 | 49 | 9 | -13.9pp |

**Nota**: para side='A', el bias en el dict es la delta directa a p_home
(NO el gap del reporte). Positivo = pitcher malo (home gana más). Negativo =
pitcher elite (home gana menos).

**Cuándo retirar**: si un pitcher es traded, se lesiona, o su gap_2026 baja
de |6pp| durante 20+ apariciones. Revalidar mensual con retrain.

---

## Tier bands (recalibrados 2026-07-09)

Los tier bands viejos (LOCK 71.5% / FUERTE 68% / MODERADO 60% / PAREJO 56.6%)
venían de walk-forward histórico 5,763 games. En 2025-2026 esos hit rates NO
son reales — el modelo drifta y los tiers mentían.

**Bands nuevos** calibrados sobre n=3,876 (2025-2026):

| Tier | Umbral \|p-0.5\| | Hit rate real | Coverage |
|---|---|---|---|
| LOCK | ≥ 0.145 | **72.4%** | 4.4% |
| FUERTE | ≥ 0.115 | **66.1%** | 12.8% |
| MODERADO | ≥ 0.045 | **58.0%** | 57.4% |
| PAREJO | ELSE | **52.6%** | 42.6% |

**Recalibrar cada 2 meses** o tras cambio de modelo. Ejecutar el script de
cutoff-search y actualizar `_TIER_BANDS` en api.py.

---

## Búsquedas probadas y DESCARTADAS (para no repetir)

Registro de análisis que resultaron NEGATIVOS. No re-explorar salvo cambio
material en features del modelo.

### Model vs Market divergence (2026-07-10)

**Idea probada**: cuando modelo y mercado disienten fuerte (>=X pp),
seguir al mercado en 2026.

**Resultado**: descartado. Holdout falla catastróficamente.
- 2023-2025: seguir mercado en disagreement >=0.10 → **-33.5pp acc**
- 2026: seguir mercado en disagreement >=0.10 → **+10pp acc** (n=60)
- El signo se invierte completamente entre régimenes. El +10pp 2026 podría
  ser puro noise en muestra chica.

**Insight guardado**: el modelo perdió su edge en disagreement. Antes ganaba
80-20 vs mercado; ahora es coin flip. Confirma que el mercado se puso al día.

### Series game number (2026-07-10)

**Idea probada**: reglas basadas en posición dentro de la serie (juego 1 vs
2 vs 3).

**Resultado**: descartado. Signal existe (2026 game 1 acc 60.1%, game 2
54.4%) pero no hay flip rule que ayude.
- Ningún bucket tiene acc <50%, así que flippear siempre baja acc.
- Solo servería para downgrade cosmético de tier (game 2 → parejo), pero
  eso no mejora acc.

### Ballpark drift (2026-07-10)

**Idea probada**: fade picks en venues donde el modelo colapsó (Citi Field,
Nationals Park, Globe Life, etc).

**Resultado**: descartado. Tras aplicar DET fade night (que ya limpiaba
varios juegos problemáticos), los venues con acc_26 <=0.50 son:
- Citi Field: 50.00% (n=46) → flip da 50%
- Camden Yards: 50.00% (n=50) → flip da 50%
- Ambos en coin flip exacto → ganancia cero.

### Rest-day mismatch (2026-07-09)

**Idea probada**: buscar señal cuando un equipo llega descansado y el otro
juega back-to-back. Diferencia de `prev_game_date` como override.

**Resultado**: negativo.
- 92% de juegos son "same rest" (rest_days=1 para ambos) → bucket dominante.
- Buckets con mismatch de 1 día (283 y 383 juegos): gap 0.0-0.2pp.
- Buckets con mismatch de 2+ días: N=5 y N=22, insuficiente.
- 2026 mostró gap +13.1pp en bucket "home +1d rest" (n=69) pero 23-25
  tira -2.9pp en el mismo bucket → **discord, small-sample noise**.

**Por qué no funciona**: el modelo ya integra rest vía `bullpen_load`, team
form rolling, y elo. Los días de descanso son una feature muy indirecta —
el efecto real está en la calidad del bullpen disponible, que sí se captura.

**Cuándo revisar**: solo si añadimos `series_position` o features de fatiga
más granulares.

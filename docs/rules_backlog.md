# Reglas H2H · Estado y Backlog

Registro de todas las reglas H2H probadas — activas, retiradas por inertes,
y retiradas por hacer daño global. Revisar tras cada retrain mensual.

**Última actualización**: 2026-07-10 (validación Codex + retiro de 2 inertes)

---

## Validación independiente de las 6 reglas de Codex (2026-07-10)

Codex agregó 6 reglas nuevas al api.py con claim de +2.6pp acc 2026 (62.54%).
Validación independiente con backtest completo desde el modelo base:

| Regla Codex | Triggers 2026 | Flippeos | acc PRE | acc CON | Δ | Verdicto |
|---|---:|---:|---:|---:|---:|---|
| **stl_calibration** | 11 | 7 | 18.2% | 63.6% | **+45.5pp** | ✅ Retenida |
| **interleague_nl** | 116 | 31 | 57.8% | 70.7% | **+12.9pp** | ✅ Retenida |
| **travel_resilience** | 60 | 14 | 60.0% | 70.0% | **+10.0pp** | ✅ Retenida |
| **umpire_market** | 86 | 10 | 61.6% | 70.9% | **+9.3pp** | ✅ Retenida |
| ~~weather_extreme~~ | 50 | 6 | 62.0% | 62.0% | **+0.0pp** aislado | ❌ **Retirada 2026-07-12** (audit) |
| circadian_extreme | 36 | 6 | 63.9% | 63.9% | **+0.0pp** aislado | ✅ Retenida (interacción) |

**Impacto real medido**: PRE Codex 59.92% → CON Codex 62.11% en 2026 (+2.20pp)
vs claim de Codex 62.54% (diferencia 0.43pp, probablemente por caches de
streaks entre corridas).

**Correción sobre "inertes"**: probamos retirar weather_extreme y circadian
por su 0pp aislado, pero el pipeline completo bajó -0.10pp global y -0.28pp
en 2025 -0.07pp en 2026. Aunque en aislamiento cancelan flippeos, en el
pipeline completo alteran el flow downstream de otras rules y contribuyen
marginalmente positivo. **Se mantienen**. Lección: no basarse solo en tests
aislados — verificar siempre en el pipeline completo.

**Update 2026-07-12 (audit override)**: creado `analysis/override_audit.py`
que revisa el net win-rate de cada override en ventanas 30d/60d/2026 completo.
Resultados:
- 🟢 h2h, blowout_momentum, travel_resil, home_band_calib, interleague: gana consistente
- 🟡 home_cold_streak, away_hot_streak: neutrales recientes, historial positivo
- 🔴 **weather_extreme: RETIRADA** — 30d 1/3 (-1), 60d 1/4 (-2), 2026 1/4 (-2)
  Pierde en las 3 ventanas. Impacto de retiro: -0.05pp 2026 (interacción pipeline)
  pero elimina rule negativa. Circadian mantenida por muestra chica (2/2 30d).

**Protocolo de audit mensual**: correr `python analysis/override_audit.py`
tras cada retrain. Si un override tiene net negativo en 30d Y 60d, retirar.

---

## Resiliencia visitante ante viaje extremo (ACTIVA 2026-07-10)

**Regla**: si `travel_dist_miles_a - travel_dist_miles_h >= 1200`, aplicar
`-0.07` a `p_home`. Gate `season >= 2025`.

La hipótesis inicial era fatiga visitante, pero el residual observado tuvo el
signo contrario: el modelo penaliza demasiado al visitante que viaja mucho más
que el local. Backtest exacto: `analysis/travel_fatigue_override_search.py`.

| Periodo | n | Baseline | Con regla | Delta |
|---|---:|---:|---:|---:|
| Global | 8,832 | 63.43% | 63.55% | +0.12pp |
| 2023 | 2,471 | 67.26% | 67.26% | +0.00pp |
| 2024 | 2,472 | 66.38% | 66.38% | +0.00pp |
| 2025 | 2,477 | 57.45% | 57.61% | +0.16pp |
| 2026 | 1,412 | 62.04% | 62.54% | +0.50pp |

Cambió 13 picks en 2026 y fue positivo en ambas mitades (+0.42pp y +0.57pp).

---

## Desplazamiento circadiano extremo en juego diurno (ACTIVA 2026-07-10)

**Regla**: si `circ_x_day_home >= 2` y `0.40 <= p_home < 0.50`, aplicar
`+0.03` a `p_home`. Esto representa un visitante viajando al menos dos zonas
al este para un juego diurno. Gate `season >= 2025`.

Backtest exacto: `analysis/circadian_override_search.py`.

| Periodo | n | Baseline | Con regla | Delta |
|---|---:|---:|---:|---:|
| Global | 8,832 | 63.32% | 63.43% | +0.11pp |
| 2023 | 2,471 | 67.26% | 67.26% | +0.00pp |
| 2024 | 2,472 | 66.38% | 66.38% | +0.00pp |
| 2025 | 2,477 | 57.17% | 57.45% | +0.28pp |
| 2026 | 1,412 | 61.83% | 62.04% | +0.21pp |

Cambió 7 picks en 2026 y fue positivo en ambas mitades (+0.14pp y +0.28pp).

---

## Umpire accuracy x market away favorite (ACTIVA 2026-07-10)

**Regla**: cuando `market_p_home < 0.45` y
`ump_acc_above_x >= 1.0360444022` (cuartil 75 aprendido en 2023-2025), aplicar
`-0.05` a `p_home`. Gate `season >= 2025`.

Backtest exacto: `analysis/umpire_market_override_search.py`.

| Periodo | n | Baseline | Con regla | Delta |
|---|---:|---:|---:|---:|
| Global | 8,832 | 63.17% | 63.32% | +0.15pp |
| 2023 | 2,471 | 67.26% | 67.26% | +0.00pp |
| 2024 | 2,472 | 66.38% | 66.38% | +0.00pp |
| 2025 | 2,477 | 57.00% | 57.17% | +0.16pp |
| 2026 | 1,412 | 61.19% | 61.83% | +0.64pp |

Cambió 11 picks en 2026, con aporte positivo en ambas mitades (+0.28pp y
+0.99pp). El threshold del umpire se fijó usando solo temporadas previas.

---

## Señales pendientes por falta de etiqueta (2026-07-10)

- **Cambio de manager**: no existe historial de manager por juego ni fechas de
  contratación/cese en los parquets actuales. No se aproximó por equipo/fecha
  para evitar etiquetado manual retrospectivo.
- **Regreso post-IL de pitcher**: no existe historial de transacciones IL. Una
  ausencia entre aperturas no distingue lesión, descanso, minors u opener.

No se implementó ninguna regla para estos dos ángulos. Reabrir solo después de
incorporar fuentes históricas con fechas efectivas conocidas antes del juego.

---

## Interleague AL vs NL en banda 45%-50% (ACTIVA 2026-07-10)

**Regla**: en juegos interleague con `0.45 <= p_home < 0.50`, aplicar un
sesgo de `0.05` hacia el equipo de Liga Nacional. Gate `season >= 2025`.

Backtest: `analysis/interleague_override_search.py`, usando el classifier
oficial, todos los overrides activos, thresholds recalculados tras cada bias y
series double-down.

| Periodo | n | Baseline | Con regla | Delta |
|---|---:|---:|---:|---:|
| Global | 8,832 | 63.03% | 63.17% | +0.14pp |
| 2023 | 2,471 | 67.26% | 67.26% | +0.00pp |
| 2024 | 2,472 | 66.38% | 66.38% | +0.00pp |
| 2025 | 2,477 | 56.88% | 57.00% | +0.12pp |
| 2026 | 1,412 | 60.55% | 61.19% | +0.64pp |

En 2026 cambió 29 picks con ganancia en ambas mitades (+0.99pp y +0.28pp).
Mantener mientras no dañe 2025 y ambas mitades de 2026 conserven signo no
negativo.

---

## Barrido secuencial de cuatro señales (2026-07-10)

Backtest: `analysis/four_signal_override_search.py`. Cada familia se midió
contra el baseline acumulado de las reglas aceptadas anteriormente. Además del
veto anual, una regla debía cambiar al menos 6 picks de 2026 y no ser negativa
en ninguna mitad cronológica de 2026.

| Orden | Familia | Decisión | Regla retenida | Delta 2025 | Delta 2026 |
|---:|---|---|---|---:|---:|
| 1 | Clima extremo | ACTIVA | temperatura >=90F, `-0.03 p_home`, gate 2026+ | +0.00pp | +0.14pp |
| 2 | Doubleheader juego 2 | DESCARTADA | ninguna; cobertura y flippeos insuficientes | +0.00pp | +0.00pp |
| 3 | Standings x mes | DESCARTADA | los candidatos que subían 2026 dañaban 2025 | +0.00pp | +0.00pp |
| 4 | p_home x local | ACTIVA | STL local con `0.40 <= p_home < 0.45`, `+0.07`, gate 2025+ | +0.04pp | +0.21pp |

### Resultado acumulado

| Periodo | n | Baseline inicial | Final | Delta |
|---|---:|---:|---:|---:|
| Global | 8,832 | 62.98% | 63.04% | +0.07pp |
| 2023 | 2,471 | 67.26% | 67.26% | +0.00pp |
| 2024 | 2,472 | 66.38% | 66.38% | +0.00pp |
| 2025 | 2,477 | 56.84% | 56.88% | +0.04pp |
| 2026 | 1,412 | 60.27% | 60.62% | +0.35pp |

La regla de calor fue neutral en la primera mitad y positiva en la segunda
(+0.28pp). La calibración STL fue positiva en ambas mitades (+0.28pp y
+0.14pp). Revalidar ambas mensualmente y retirarlas ante signo negativo
en cualquier mitad o daño al año previo.

---

## Bullpen quality differential extremo (RETIRADA 2026-07-10)

**Estado**: retirada del runtime.

**Regla**: usar `features_bullpen_quality.parquet::bullpen_fip_recent`.
Si `home_fip_recent - away_fip_recent <= -1.8415`, aplicar `+0.04` a `p_home`.
Si `home_fip_recent - away_fip_recent >= +1.8415`, aplicar `-0.04` a `p_home`.
El sesgo se aplica antes del threshold, junto con H2H/pitcher/streak/momentum.

**Backtest 2026-07-10** (`analysis/bullpen_quality_override_search.py`, baseline de producción reconstruido con gates principales + series double-down vectorizado):

| Periodo | n | Baseline acc | Con regla | Δ acc |
|---|---:|---:|---:|---:|
| Global | 8,832 | 60.38% | 60.64% | +0.26pp |
| 2023 | 2,471 | 62.89% | 62.89% | +0.00pp |
| 2024 | 2,472 | 62.74% | 62.74% | +0.00pp |
| 2025 | 2,477 | 57.29% | 57.90% | +0.61pp |
| 2026 | 1,412 | 57.29% | 57.86% | +0.57pp |

**Motivo del retiro**: el resultado positivo anterior usó por error la
probabilidad mezclada de display. Al repetir sobre `lgb_cls.pkl::predict_proba`,
la ruta oficial de winner accuracy, 2026 cayó de 60.27% a 59.70% (-0.57pp).
No reactivar sin un nuevo holdout sobre el baseline oficial.

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

### Pitcher rest days (4 vs 5+) (2026-07-10)

**Idea**: Pitchers con extra rest tienen ventaja no capturada por modelo.

**Resultado**: descartado. Buckets home_rest = 7, away_rest = 7, y rest_diff
extremos muestran discord cross-year (signos flippean 23-24 vs 25-26). El
modelo ya integra `starter_days_rest` como feature.

### Slate slot (game #1 vs late) (2026-07-10)

**Idea**: Late-slot games menos precias por mercado.

**Resultado**: descartado. Slot 6+ tiene gap +1.7pp concordante pero
cualquier bias empeora acc (todos deltas negativos, mejor -0.354pp 2026).
Modelo ya integra vía features.

### Team-vs-team prev series carryover (2026-07-10)

**Idea**: Si home barrió/fue barrido en serie previa contra este rival,
hay carryover.

**Resultado**: descartado. Signos flippean cross-year (0/3 barrido: +5/+5/-1/-7,
3/3 dominó: +0.6/+0.1/-2/+10). Backtest best +0.071pp 2026 dentro del ruido.

### Doubleheader game 2 (2026-07-10)

**Idea**: Bullpen fatigue + rotación en game 2 de DH.

**Resultado**: descartado. Solo n=3 juegos DH game 2 en 2026 — muestra
totalmente insuficiente para decidir.

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

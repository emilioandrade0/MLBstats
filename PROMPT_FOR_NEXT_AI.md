# Prompt para seguir mejorando STRIKECAST con otra IA

Copia todo lo que está debajo del `---` como el primer mensaje.

---

Estoy trabajando en STRIKECAST, un modelo LightGBM que predice quién gana partidos MLB (win/lose binary). El modelo entrena con walk-forward mensual y expone picks vía FastAPI + frontend HTML.

## Estado actual (2026-07-17)

- **Fuente autoritativa**: `data/processed/walkforward_preds.parquet`, regenerado
  por folds mensuales desde abril de 2025 hasta el 12 de julio de 2026.
- **Global 2025-2026 walk-forward tras retiro pitagorico**: 59.23% (2,237/3,777)
- **2025**: 58.70% (1,414/2,409)
- **2026**: 60.16% (823/1,368)
- **2026 H1**: 60.96% (417/684)
- **2026 H2**: 59.36% (406/684)
- **Meta**: subir 2026 y predicciones futuras sin fuga temporal.
- El `65.79%` reportado antes era una reconstruccion del runtime usando el modelo
  guardado sobre `train.parquet`; replica decisiones, pero no es walk-forward y
  no debe usarse para aprobar señales.
- **Fuente vigente de reglas**: `ReglasAMonitorear.md` y
  `docs/rules_backlog.md`. Las cifras antiguas cercanas a 60% correspondían a
  snapshots o rutas parciales y no deben mezclarse con el pipeline completo.

## Disciplina estricta (obligatoria)

1. **Prioridad**: acc > AUC > log_loss.
2. **Solo implementar reglas que:** (a) suben acc walk-forward 2026, (b) no
   dañan 2025 y (c) no dejan negativa ninguna mitad de 2026.
3. **Todo override nuevo va con gate `season >= 2025`** (o `>= 2026` si el patrón es regime-specific).
4. **Holdout obligatorio**: aprender en 2025 y vetar con 2026; para reglas de
   regimen 2026, desarrollar en H1 y validar en H2 sin usar resultados de la
   fecha actual.
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
- `TRAP_PICK_TEAMS = {LAA}`; `TRAP_FADE_TEAMS = {MIL}` solo aplica si la diferencia modelo-mercado para MIL es `<=0.07`; `DAY_TRAP_FADE = {(WSH, 5)}`
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

## Ideas del snapshot inicial ya resueltas

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

Bullpen quality, weather, standings, umpire-market, team series, home-team
calibration, doubleheader e interleague ya fueron analizados después de crear
este prompt; consultar `docs/rules_backlog.md` antes de repetirlos. Manager
change y post-IL siguen bloqueados por falta de historial fiable. El usuario
pidió no continuar con nuevas reglas basadas en identidad de equipo.

La búsqueda más reciente fue incertidumbre/cobertura pregame. Quedó en
`WATCHLIST` un corte 2026 con cobertura reciente mínima del lineup `<=2`, pero
no se implementó por tener solo 6 flips. Script:
`analysis/information_uncertainty_override_search.py`.

Después se descartó el running game/control defensivo tras 2,371 reglas. La
forma ajustada por Elo L5 dejó una `WATCHLIST` de 17 flips y `+3` netos en
2026, pero no se activó por fragilidad al tamaño del bias. Consultar las
secciones del 2026-07-13 en `docs/rules_backlog.md` antes de repetirlas.

La búsqueda de carreras ajustadas por calidad Elo dejó otra `WATCHLIST` 2026:
residuo ofensivo L20 extremo, 13 flips y `+5` netos, pero H2 solo confirmó `+1`.
No se implementó. Script: `analysis/opponent_adjusted_run_margin_search.py`.

La profundidad/eficiencia del abridor también fue probada con 2,727 reglas. El
mejor corte 2026 dio `+5` en 15 flips, pero solo tuvo dos flips en H1 y perdió
fuertemente en 2025. Fue descartado y no debe repetirse sin datos nuevos.

El ángulo externo de microestructura de mercado también quedó auditado. El
script `analysis/external_market_anomaly_audit.py` reconstruye snapshots de
DraftKings, FanDuel, BetMGM y Pinnacle, de-vigea moneylines y prueba consenso,
sharp-soft gap, dispersión, movimientos y libros atípicos con corte temporal.
Solo hay 135 juegos terminados y 91 con múltiples snapshots; ninguna regla
pasó desarrollo y holdout. No repetir el barrido hasta acumular al menos
300-500 juegos comparables o incorporar una fuente fiable de tickets y dinero.
No existe evidencia para etiquetar juegos como manipulados ni se implementó
ningún cambio al runtime.

Importante para `VALOR SHARP`: es un badge de comparación de precios y no entra
en `p_home` ni en `pick_abbrev`. El 2026-07-20 se corrigió un bug en
`src/normalize/pinnacle_value.py` y su espejo de backend: el agrupamiento por
equipos mezclaba cuotas de juegos consecutivos de una serie. Ahora se agrupa
por `event_id + away + home`; la regresión está en
`tests/test_pinnacle_value.py`. No usar ese badge como señal de ganador sin una
validación OOS independiente.

Tambien se descarto usar la prediccion de un juego futuro cercano para flipear
el actual, o el actual para flipear el siguiente encuentro de la misma serie.
Sobre 3,880 pares separados por uno o dos dias, el primer sentido perdio
`-115/-35` netos en 2025/2026 y el inverso `-37/-75`. Ningun gate paso ambas
temporadas y las dos mitades de 2026. No repetir sin una fuente nueva de
snapshots diarios pregame. Script:
`analysis/consecutive_future_prediction_signal_search.py`.
En vivo, tampoco comparar el pick maduro de hoy con un forecast futuro que aun
no tenga mercado y ambos pitchers probables; esas diferencias son preliminares,
no senales para flipear.

Tambien se descarto flipear a un favorito solo porque perdio el dia anterior
contra el mismo rival. El favorito repetido aun gano `56.88%/53.50%` en
2025/2026 y el flip contra el pick final perdio `-73/-60` aciertos. El bolsillo
de mercado volteado y casi parejo dio `+4/+1`, pero fallo H1/H2 de 2026 y tuvo
solo cinco casos en 2026. Script:
`analysis/repeat_favorite_after_loss_search.py`.

La busqueda de trayectoria de momios descarto cruces de favorito, steam,
movimientos 1-5pp, hold y debilitamiento/fortalecimiento como overrides. Quedo
solo un monitor no activo: favorito pregame sin vig `>=66%` cuando el pipeline
elige underdog. Dio `+10/36` en 2025 y `+3/7` en 2026, positivo en ambas
mitades, pero H2 2026 tiene un solo caso. Umbral congelado; no aplicar hasta
acumular muestra prospectiva. Script:
`analysis/odds_trajectory_pattern_search.py`.

Tambien se descarto usar la suma diaria de momios de los equipos ganadores como
regimen para el dia siguiente. El promedio ganador del mismo dia correlaciona
con menor accuracy (`-0.421/-0.452`), pero es retrospectivo. El lag T-1 cambio
de signo entre 2025 y 2026 (`-0.088/+0.090`) y ninguna regla por suma, promedio,
exceso, T-3 o shock paso validacion. Script:
`analysis/daily_winner_odds_regime_search.py`.

La variante intradia tambien quedo descartada. Se usaron solo juegos ya
terminados antes del primer lanzamiento de cada target tardio; 14 reglas
pasaron H1/H2 de 2025 y ninguna sobrevivio H1/H2 de 2026. La mejor cambio de
`+13` a `-3`. No implementar recarga de picks basada en resultados tempranos.
Script: `analysis/intraday_completed_odds_regime_search.py`.

**Protocolo vigente desde 2026-07-21**: por peticion del usuario, toda nueva
busqueda de patrones debe usar exclusivamente datos de 2026; no incluir 2025
ni temporadas anteriores en thresholds, seleccion o decision. Validar dentro
de 2026 con bloques cronologicos. La primera busqueda bajo este protocolo probo
450 reglas: 2 pasaron marzo-abril, 0 mayo/junio y 0 julio. Veinte reglas eran
positivas en ambas mitades de julio, pero todas negativas pre-julio. Script:
`analysis/odds_patterns_2026_only_search.py`.

**Actualizacion posterior 2026-07-21**: el usuario autorizo busquedas en 2025
y 2026, siempre como experimentos completamente separados. No mezclar datos,
umbrales, seleccion ni reglas entre temporadas. Una regla de 2025 solo puede
describir/aplicarse a 2025; una regla para predicciones futuras de 2026 debe
superar bloques cronologicos independientes dentro de 2026. Las reglas que
fallan algun holdout se descartan incluso si el neto global del ano es positivo.

Tambien se probaron ambas inversiones alrededor de julio: lado opuesto
pre-julio y accion en julio; o accion pre-julio, fallo en H1 e inversion en H2.
Solo 2/450 fueron reversals en el primer sentido y ambas fallaron julio H1; en
el segundo hubo 0 reglas elegibles. No implementar switch temporal. Script:
`analysis/regime_switch_2026_only_search.py`.

El diagnostico de 539 errores de 2026 encontro que 294 (`54.5%`) involucraron
que el equipo elegido anotara dos carreras o menos. Entre picks con abridor top
20% por xwOBA, 55 de 100 derrotas fueron por ofensiva apagada y solo 18 por
colapso del abridor. Se probaron 50 fades de equipo caliente, pitcher fuerte,
carga, muestra, barrel y bullpen: 2 pasaron marzo-abril, 0 mayo/junio y 0 julio.
No flipear automaticamente estos perfiles. Script:
`analysis/error_attribution_2026_only.py`.

El detalle postgame atribuyo los 294 apagones a dominio del abridor rival
(134), trafico varado (79), ponches (47), falta de trafico (30) y otros (4).
Luego se probaron 171 reglas pregame de alineacion vs mano, forma, starter rival
y bullpen rival; cero pasaron siquiera marzo-abril. No repetir estas
interacciones con los mismos datos. Script:
`analysis/offense_shutdown_matchup_2026_search.py`.

Tambien se rehizo el matchup por repertorio sin depender de lanzamientos del
juego objetivo. `analysis/pitch_arsenal_matchup_2026_search.py` usa solo fechas
anteriores de 2026, mezcla de los cinco inicios previos y xwOBA/whiff de la
alineacion por familia de pitcheo. Cobertura fuerte: 1,043 juegos. De 50 reglas,
3 pasaron marzo-abril y 0 mayo/junio. La mejor fue `+3/-4/-1/+3/+1` por bloques
y solo suma `+2` global por el rebote reciente de julio. No implementar ni
reactivar el arsenal antiguo.

También se agotó el historial multicasa estático con
`analysis/historical_multibook_market_search.py`: desarrollo 2023 con 2,469
juegos, validación 2024 con 383 y holdout Odds API 2026 con 117. El pipeline
superó al consenso en los tres cortes. Las únicas tres reglas que sobrevivieron
2023-2024 perdieron `-10` netos en 16 flips durante 2026. No repetir consenso,
dispersión, rango, IQR, votación de libros ni divergencia modelo-mercado con
estos mismos datos; hace falta movimiento temporal nuevo o splits de
tickets/dinero.

La búsqueda H2H completa posterior sí produjo una regla activa. Consultar
`analysis/h2h_all_matchups_search.py` y la sección del 2026-07-13 en
`docs/rules_backlog.md`. Regla: mismo home-away con al menos tres antecedentes,
win rate local Beta(2.5,2.5)-suavizado `>=0.70`, `0.50 <= p_home < 0.60`, bias
`-0.07` al local. Excluye los ocho pares de `H2H_PICK_BIAS` y resultados de la
misma fecha. Impacto exacto: global `+15` netos, 2026 `+3` (`+0.21pp`), positivo
en las cuatro temporadas. Campo API: `h2h_venue_regression_bias`.

## Formato de trabajo esperado

1. **Antes de implementar cualquier regla**: hacer analisis + backtest sobre
   `walkforward_preds.parquet`, separando 2025 y 2026 desde el inicio y
   mostrando bloques cronologicos internos de cada temporada.
2. **Preguntar al usuario cuál probar** si hay varios candidatos.
3. **Implementar solo si sube el ano objetivo y confirma fuera de discovery en
   bloques posteriores del mismo ano**. Nunca transferir una regla anual al
   otro ano. Si no cumple, descartar y pasar a otra.
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
# Cargar probabilidades realmente fuera de muestra y features pregame.
import pandas as pd

wf = pd.read_parquet('data/processed/walkforward_preds.parquet')
train = pd.read_parquet('data/processed/train.parquet')
df = wf.merge(train, on='game_pk', how='left', suffixes=('', '_train'))
df['game_date'] = pd.to_datetime(df['game_date'])

# Nunca reemplazar wf.p_home con model.predict_proba(train): eso es in-sample.
```

Cuando el user pida seguir buscando señales para subir accuracy, corre primero
la baseline walk-forward de los anos autorizados y conserva bloques
cronologicos posteriores de cada uno como veto. Sigue la disciplina al pie de
la letra.

## Ultima ablacion del runtime real - 2026-07-21

`analysis/source_runtime_season_ablation_2526.py` es la referencia para
quitar componentes ya activos. A diferencia de reconstrucciones viejas, usa
solo los biases realmente activos en `src/serve/api.py`; `umpire_market` y
`burn_resilience` estan retirados y no deben volver a entrar a un baseline.

La unica excepcion aplicada fue `interleague` en 2025: desactivarla cambio 58
picks y gano `+10` netos, con `+6` en H1 y `+4` en H2. `_interleague_nl_bias`
devuelve cero solo en 2025, tanto en `src/serve/api.py` como en
`backend/src/serve/api.py`. Esto no modifica 2026 ni anos futuros.
El rerun post-cambio marca `59.40%` en 2025 y conserva `59.69%` en 2026;
el CSV registra que restaurar la version anterior pierde `-10` en 58 flips.

No retirar `h2h_venue_regression` en 2026 pese a `+4`: solo cambio seis picks.
No implementar combinaciones de ablacion sin un holdout independiente. Antes
de cualquier nueva retirada, exigir volumen suficiente y validacion
cronologica dentro del mismo ano.

## Escalas de overrides 2026 - descartadas - 2026-07-21

`analysis/source_runtime_strength_2026_search.py` ya probo escalas
`0/25/50/75/125/150%` de cada componente activo con cuatro bloques
cronologicos. La escala se eligio solo con B1 y B2-B4 fueron vetos. Cero
candidatas cumplieron el minimo de cinco flips y `+2` en discovery; la mejor
aparente (`situational_streaks x1.25`) fue `+2/-2/0/-2`. No repetir este
barrido de escalas individuales sin datos futuros nuevos.

## Coherencia del stack 2026 - descartada - 2026-07-21

`analysis/source_runtime_stack_coherence_2026_search.py` probo ocho acciones
sobre senales que se contradicen, se cancelan o se refuerzan. Retirar el stack
mezclado fue `+3/-2/-2/+2` por B1-B4; las escalas reforzadas solo movieron tres
picks. No repetir conflicto/refuerzo de los mismos overrides sin datos futuros.

## Ejecucion defensiva L20 2026 - descartada - 2026-07-21

`analysis/source_runtime_defense_2026_search.py` evaluo errores por 100
oportunidades, carreras no limpias, costo y recuperacion L20 de forma
leakage-safe. Las 32 variantes predefinidas no lograron cinco flips y `+3` en
B1; la de mayor cobertura perdio `-6` en 40 cambios. No reabrir esta familia
sin datos de campo materialmente nuevos.

## Umpire por banda de mercado 2026 - descartada - 2026-07-21

`analysis/source_runtime_umpire_market_2026_search.py` ya probo
`ump_acc_above_x` alto/bajo x cuatro bandas de mercado x follow/fade de
`0.03/0.05`, con B1 para seleccion y B2-B4 como veto. Cero reglas alcanzaron
cinco flips y `+3` en discovery. Mantener retirado `umpire_market` y no repetir
este barrido sin datos futuros nuevos.

## Auditoria de badges 2026 - 2026-07-24

`analysis/badge_audit_2026.py` uso picks walk-forward finales hasta
2026-07-19. LOCK es `61.5%` (64/104), FUERTE `74.1%` (43/58; B2-B4 28/37),
MODERADO `58.1%` y ELITE NOCHE `69.4%` (34/49). LOCK+FUERTE actual es 66.0%.
Los tiers y Elite son informativos; no cambian ganador. FUERTE es candidato a
filtro de precision menor cobertura, no mejora global.

No retocar runtime por badges: quitar Noche da -1 global pero +4 B4; Viaje es
0 y quitar LOB pierde -4. Mantener LOB, monitorear Noche. Cualquier filtro
FUERTE nuevo debe declararse selector de cobertura reducida, no accuracy de
todos los juegos.

## Calibracion de badges aplicada - 2026-07-24

Las APIs `src` y `backend/src` ya devuelven tiers sincronizados de la auditoria
2026-WF: LOCK 61.5%, FUERTE 74.1%, MODERADO 58.1%, PAREJO 59.2%. La interfaz
reemplazo el filtro LOCK+FUERTE por **Seleccion precisa**, que muestra solo
FUERTE (43/58; B2-B4 28/37). Es menor cobertura y no cambia picks ni accuracy
global. ELITE NOCHE local se corrigio a 69% (34/49).

## Inventario de badges universales 2026 - 2026-07-24

Nuevo artefacto: `analysis/event_badge_inventory_2026.py` y
`data/processed/event_badge_inventory_2026.csv`. Reconstruye el pick final
actual y evalua solo 2026-WF hasta 2026-07-19; B1 descubre, B2-B4 vetan.

Unico candidato que pasa: `Modelo respalda 60%`, `129/190 = 67.89%`, lifts
por bloque `+3.30/+6.18/+16.13/+9.35pp`. NO esta desplegado: incluye los 162
LOCK+FUERTE ya mostrados y solo agrega 28 MODERADO. Un badge extra duplicaria
informacion sin mejorar un selector ni cambiar picks.

No desplegar consenso de mercado (60.43%), triple consenso (61.04%), doble
respaldo 60% (67.44% pero B1 -2.08pp), favorito de mercado 60% (64.57% pero
B1 -1.96pp) ni contra mercado (57.46%, inestable). No hacer cambios runtime
de esta familia sin evidencia futura nueva.

## HR en derrotas como senal de resiliencia - DESCARTADA 2026-07-24

Datos disponibles: `team_box.parquet` trae `bat_homeRuns`; `plays.parquet`
tiene `event_type=home_run` y `rbi`, asi que se pueden medir carreras exactas
impulsadas por HR, no solo el conteo. `analysis/hr_loss_resilience_2026_search.py`
construye features estrictamente lagged L10/L20 de HR/RBI por derrota y por
derrota cerrada. Solo prueba flips cerca de p_home 45-55%, con B1 discovery y
B2-B4 veto.

Las 16 candidatas fallaron B1 y no hay regla: HR-RBI por derrota L10 fue -13
en 103 flips; la mejor global, HR-RBI por derrota cerrada L10, fue +1 en 57
pero -1 en B1. No crear feature, badge ni bias: HR en derrota no separa
resiliencia ofensiva de pitcheo/bullpen deficiente. Resultado:
`data/processed/hr_loss_resilience_2026_search.csv`.

## Poder Statcast / HR - DESCARTADAS 2026-07-24

Dos extensiones ya probadas; no reabrir sin datos futuros nuevos.
`analysis/barrel_conversion_resilience_2026_search.py` mide barriles sin HR
por contacto L10/L20: la unica seleccion B1 fue L20/gap 1.5pp, `+4` en ocho
flips, pero B2 `-14` y ano `-10` en 36. Resultado:
`data/processed/barrel_conversion_resilience_2026_search.csv`.

`analysis/power_vs_starter_hr_2026_search.py` cruza barriles de la ofensiva
con HR permitidos por PA de abridor probable rival (L5/L10). Sus ocho
candidatas no pasaron el minimo B1; la mejor empato en 16 flips. No agregar
feature, bias ni badge. Resultado:
`data/processed/power_vs_starter_hr_2026_search.csv`.

## HR y barriles para Totales / F5 - SIN DESPLIEGUE 2026-07-24

`analysis/hr_power_totals_f5_search.py` hace backtest OOS mensual 2025-01 a
2026-07 sin tocar ganador: baseline vs barriles L10/L20, barriles sin HR,
HR/PA del abridor probable y presion de matchup. Evalua MAE de total/F5 y O/U
de total con edge >=0.5.

F5: `2.5402 -> 2.5308` global, pero 2026 solo `2.5007 -> 2.4958`, B4 pierde
0.0107 y bootstrap 2026 `-0.0074..+0.0171`; no es estable. Total completo
empeora en 2026 `3.3954 -> 3.4034`, y O/U baja `57.61% -> 56.46%`.
NO regenerar `f5_reg`, NO cambiar `pred_total`, no mostrar mercado. Resultado:
`data/processed/hr_power_totals_f5_preds.parquet`.

## Inversion de senales HR - DESCARTADA 2026-07-24

La hipotesis inversa no paso. `analysis/hr_signal_inverse_2026_search.py`
invirtio las 28 variantes HR de ganador (menos HR/RBI en derrotas, menos
barriles sin HR o menor presion HR frente al abridor). Todas fallaron B1;
la mejor por volumen empato en 34 flips y perdio B3/B4. No hacer flips
inversos de ganador. Artefacto:
`data/processed/hr_signal_inverse_2026_search.csv`.

El caso literal de que un pico de HR ayer reduzca el siguiente total/F5 se
probo en `analysis/hr_next_game_regression_totals_f5_2026.py`: condiciones
2+ HR de un equipo, 3+/4+ combinados o HR de ambos, y descuentos de
total `0.25/0.50/0.75` o F5 `0.15/0.30/0.45`. Las pequenas mejoras F5 no
pasaron B1; total tampoco fue estable. No cambiar ganador, `pred_total`,
`pred_f5_total`, badges ni UI. Artefacto:
`data/processed/hr_next_game_regression_totals_f5_2026.csv`.

## Equipo por posicion de serie - DESCARTADO 2026-07-24

`analysis/team_series_position_residual_2025_2026.py` probo entrenar cada
equipo por G1/G2/G3/G4+ usando solo resultados de fechas anteriores: residuo
historico `resultado - probabilidad OOS`, con shrinkage 6/12/24, minimo 5/10,
historial de temporada o de temporadas previas y escala 0.15/0.30/0.45.

2025: ninguna variante alcanzo B1 y las mejores terminaron `-33..-45`.
2026: la seleccion B1 fue `+4`, pero B2/B3 fueron `-3/-2`; termino `0` en
50 flips. No agregar feature, bias, flip, badge ni ajuste por juego de serie.
Artefacto: `data/processed/team_series_position_residual_2025_2026.csv`.

## Mismo dia calendario 2025 -> 2026 - DESCARTADO 2026-07-25

`analysis/yoy_calendar_result_signal_2026.py` probo el patron visual de que
el resultado de un equipo en el mismo mes-dia de 2025 predice 2026 aunque
cambie localia. Evalua repetir/invertir global y por rol previo, con gates
all/40-60/45-55 y B1 discovery, B2-B4 veto. En 2,464 equipo-fechas, los
cuatro grupos quedaron `48.6%..51.3%`; las 12 variantes perdieron desde B1
(mejor `-4` B1, `-23` global). No agregar regla, flip, feature o badge.
Artefacto: `data/processed/yoy_calendar_result_signal_2026.csv`.

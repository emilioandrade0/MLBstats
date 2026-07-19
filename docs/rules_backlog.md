# Reglas H2H · Estado y Backlog

Registro de todas las reglas H2H probadas — activas, retiradas por inertes,
y retiradas por hacer daño global. Revisar tras cada retrain mensual.

**Última actualización**: 2026-07-18 (familia contexto de serie/calendario)

---

## Juego 1 tras ser barrido en la serie anterior (DESCARTADO 2026-07-18)

Se probó si el equipo que viene de perder TODOS los juegos de su serie
anterior (barrida, serie >= 2) rebota o sigue hundido en el primer juego de
la serie siguiente. Evaluado en `walkforward_preds.parquet`; barridas
reconstruidas por equipo desde `games.parquet` (solo juegos previos).

- HOME post-barrida (n=91 en 2025, 59 en 2026): flip de signo total.
  2025 hwr 38.5% (gap `-11.7pp` vs modelo) → 2026 hwr 62.7% (gap `+11.7pp`).
  El fade al local barrido daba `+7` en 2025 pero `-5` en 2026 — caso de
  libro de regla que se ve perfecta en desarrollo y muere en holdout.
- AWAY post-barrida (n=109/45): gaps `-1.4pp` → `+4.2pp`, biases `+3` en
  2025 pero negativos en 2026.

Ninguna variante pasa 2025 + 2026 + mitades. No se tocó el runtime.

## Primer juego de road trip tras homestand >= 6 (DESCARTADO 2026-07-18)

Se probó si el visitante que inicia gira tras una estadía larga en casa
(>= 6 juegos, n=321 en 2025, 182 en 2026) rinde distinto a lo priceado.
Gaps modelo-realidad: `+2.25pp` (2025) → `-1.81pp` (2026), signo inestable.
El sweep de biases fue el peor de la familia: todas las variantes negativas
en ambos años (boost home `-11/-10`, fade home `-13/-8`); la única neutra
dio exactamente 0 flips netos en 2026. El modelo ya captura el ángulo vía
travel_resilience y features de viaje. No se tocó el runtime.

Con esto la familia "contexto de serie/calendario" queda agotada: juego 3
con 2-0, post-barrida y road trip largo, todos descartados. El único
mecanismo de serie activo sigue siendo el series double-down trap.
Script: scratchpad `swept_and_roadtrip.py`.

---

## Juego 3 de serie con un equipo arriba 2-0 (DESCARTADO 2026-07-18)

Se probó si en el tercer juego de una serie, cuando el local o el visitante
ya ganó los primeros dos (2-0), existe miscalibración explotable (barrida vs
evitar-barrida). Evaluado sobre `walkforward_preds.parquet` (1,195 juegos 3,
356 con HOME 2-0, 271 con AWAY 2-0, 568 control 1-1). Series reconstruidas
desde `games.parquet` usando solo resultados previos al juego actual.

Hallazgos:
- El equipo 2-0 gana el juego 3 a su tasa base — sin momentum ni relajación.
  Gaps modelo-realidad: HOME 2-0 `-0.96pp` (2025) y `+0.02pp` (2026);
  AWAY 2-0 `-1.40pp` (2025) y `+2.30pp` (2026). Signos inestables.
- Pick accuracy en el bucket se invierte entre años (HOME 2-0: 59.9% en 2025
  pero 50.4% en 2026; AWAY 2-0: 50.6% en 2025 pero 63.4% en 2026) — ruido.
- Sweep de 8 biases (fade/boost, ±0.05/±0.07, bandas): ninguno positivo en
  2025 Y 2026 Y ambas mitades. El mejor (fade home fuerte con HOME 2-0) dio
  `+4` en 2025 pero `0` en 2026 con H1 `-5`.

La situación 2-0 ya está priceada por modelo y mercado. El ángulo de series
que sí funciona sigue siendo el series double-down trap (activo). Script:
scratchpad `series_2_0_game3.py` (lógica reproducible: series por pareja
home/away con gap <=2 días, wins acumulados con exclusión del juego actual).

---

## Continuidad del lineup confirmado (DESCARTADO 2026-07-12)

Se construyeron señales pregame de continuidad usando el lineup titular del
juego actual contra el juego anterior: bateadores titulares que regresan,
miembros del top 4 que regresan y cantidad de cambios. Se probaron diferenciales
home-away, bandas de probabilidad, persistencia/regresión y biases 0.02-0.07.

Ninguna variante mejoró 2025 y ambas mitades de 2026 con al menos seis flips.
Los mejores resultados cambiaron solo 1-3 picks y fueron neutrales o negativos
en H1. No se agregó override. Script:
`analysis/lineup_continuity_override_search.py`.

---

## Fatiga de relevistas de alto leverage (ACTIVA 2026-07-12)

**Regla general, sin equipos**: antes de cada juego se rankean los relevistas de
cada club por sus 30 días previos (`4 * saves + apariciones`) y se toman los dos
principales. Si la diferencia de pitches lanzados por esos brazos el día anterior
es al menos 27 y `0.45 <= p_home < 0.55`, aplicar bias `0.03` contra el bullpen
más cargado. Gate 2025+.

El ranking y workload se calculan cronológicamente antes de cada juego y se
publican en `features_high_leverage.parquet`; no usan apariciones actuales.

| Periodo | Accuracy antes | Accuracy después | Delta |
|---|---:|---:|---:|
| 2025 | 59.46% | 59.54% | **+0.08pp** |
| 2026 | 65.24% | 65.45% | **+0.21pp** |
| 2026 H1 | 65.70% | 65.70% | **+0.00pp** |
| 2026 H2 | 65.26% | 65.68% | **+0.42pp** |

Cambió 7 picks de 2026. Los umbrales 24-32 y bias 0.03 conservaron signo
positivo. La API expone `highlev_fatigue_bias`; en el smoke window el modelo
actual no produjo un flip, por lo que el badge correctamente permaneció apagado.
Backtest: `analysis/high_leverage_availability_search.py`.

---

## Regresión de carga anterior del abridor (ACTIVA 2026-07-12)

**Regla general, sin equipos**: calcular la diferencia away-home de pitches
lanzados por los abridores probables en su aparición inmediatamente anterior.
Si `abs(diff) >= 34` y `0.40 <= p_home < 0.50`, aplicar un bias de `0.04` contra
la ventaja aparente del lado que lanzó menos: `-sign(diff) * 0.04`. Gate 2025+.

La feature se deriva de `player_box.parquet` con `shift(1)` por pitcher y se
publica en `features_starter_workload.parquet`; no usa el juego actual.

| Periodo | Accuracy antes | Accuracy después | Delta |
|---|---:|---:|---:|
| 2025 | 59.18% | 59.46% | **+0.28pp** |
| 2026 | 65.10% | 65.24% | **+0.14pp** |
| 2026 H1 | 65.42% | 65.70% | **+0.28pp** |
| 2026 H2 | 65.26% | 65.26% | **+0.00pp** |

Cambió 14 picks de 2026 con corte 34/bias 0.04. Los umbrales 32-40 conservaron
signo global positivo; se mantuvo el percentil 85 aprendido en 2025. La API
expone `starter_workload_bias`, Docker incluye el parquet y el serving histórico
confirmó activación. Backtest: `analysis/starter_workload_override_search.py`.

---

## Microclima: viento, techo y temperatura (DESCARTADO 2026-07-12)

Se probaron señales generales de `features_weather.parquet`: viento hacia los
jardines 5/8/10/12 mph, velocidad total 8/12/15/18 mph, estadio indoor y bandas
de temperatura 35-55/55-70/70-85/85-100 F. Se cruzaron con bandas `p_home` y
biases 0.02/0.03/0.05/0.07 hacia ambos lados.

El calor 85-100 F mejoró 2026 H2 hacia el local, pero dañó 2025 entre
`-0.48pp` y `-0.93pp` y fue negativo en H1. Viento >=18 mph tuvo signo positivo
pero solo 1-3 flips 2026. Ninguna variante cumplió estabilidad y muestra mínima.
No se agregó override. Script: `analysis/microclimate_override_search.py`.

---

## Regímenes móviles de liga 7/14 días (DESCARTADO 2026-07-12)

Se construyeron señales leakage-safe usando exclusivamente jornadas terminadas:
tasa local, residuo home real vs `p_home`, accuracy reciente del pipeline y
accuracy de favoritos de mercado, con ventanas 7/14 días y `shift(1)`.
Se probaron colas 20/35/65/80%, bandas de probabilidad y biases 0.02/0.03/0.05.

Los mejores fades locales mejoraron la segunda mitad de 2026 entre
`+0.42pp` y `+0.55pp`, pero dañaron 2025 (`-0.08pp` a `-0.40pp`) y fueron
neutrales o negativos en 2026 H1. Es un régimen reciente no estable, por lo que
no se agregó override. Script: `analysis/league_regime_override_search.py`.

---

## Consenso, conflicto y reescalado de señales (DESCARTADO 2026-07-12)

Se evaluaron interacciones generales entre los biases activos: refuerzo o
amortiguación cuando dos/tres señales concordaban, resolución de conflictos y
mayoría de signos. Los ajustes probaron 0.01-0.05 sobre el pipeline completo.
Ninguna variante mejoró 2025 y ambas mitades de 2026.

También se repitió la ablación/escalado individual incluyendo comeback,
mediana ofensiva y burn. Viaje a 1.25x produjo solo tres flips 2026 y no cumplió
el mínimo; 1.5x dañó la primera mitad. Ningún otro scale 0x/0.5x/0.75x/1.25x/
1.5x fue válido. Se mantienen todas las intensidades actuales. Script:
`analysis/signal_consensus_override_search.py`.

---

## Reliability móvil del modelo por participante (DESCARTADO 2026-07-12)

Se probó una señal general basada en `team_model_acc_l30`, sin identidades de
equipo: diferencia home-away de accuracy móvil del modelo y reliability del
lado elegido, exigiendo al menos 15 observaciones previas por participante.
Se evaluaron bandas de probabilidad, persistencia/regresión y biases 0.02-0.07.

Una variante parecía sumar `+0.07pp` en 2026 con diferencia cercana a 0.20,
pero el resultado dependía de representar el corte como
`0.19999999999999996`. Al usar 0.20 exacto y barrer 0.15-0.25, dañó 2025 o la
primera mitad de 2026. Se considera un artefacto de frontera y fue descartada.
No se agregó ningún override. Script:
`analysis/model_reliability_override_search.py`.

---

## Resiliencia ante burn de bullpen (ACTIVA 2026-07-12)

**Regla general, sin equipos**: `burn_score` resume uso del bullpen, innings,
apariciones, cerrador, extra innings y contexto del juego anterior; todas sus
entradas usan `shift(1)`. Si la diferencia absoluta home-away es al menos 14 y
`0.50 <= p_home < 0.60`, aplicar un bias de `0.06` hacia el lado con mayor burn.
Gate `season >= 2025`.

El signo es contracorriente: no interpreta desgaste como fortaleza, sino que
corrige una penalización excesiva ya incorporada por modelo/mercado. Es análogo
a la resiliencia visitante detectada con viaje extremo.

| Periodo | Accuracy antes | Accuracy después | Delta |
|---|---:|---:|---:|
| 2025 | 59.14% | 59.18% | **+0.04pp** |
| 2026 | 64.75% | 65.10% | **+0.35pp** |
| 2026 H1 | 65.14% | 65.42% | **+0.28pp** |
| 2026 H2 | 64.84% | 65.26% | **+0.42pp** |

Cambió 15 picks de 2026 con el corte 14/bias 0.06. La señal mantuvo signo
positivo entre umbrales 12-18. La API carga `features_burn.parquet`, expone
`burn_resilience_bias` y mostró seis activaciones reales entre el 25 de marzo y
el 10 de abril de 2026. Backtest: `analysis/burn_context_override_search.py`.

---

## Arsenal, tendencias de abridor y situacionales (DESCARTADO 2026-07-12)

Se evaluaron señales generales de matchup sobre el pipeline acumulado: xwOBA de
lineup contra el arsenal real del abridor, xwOBA/K%/velocidad L5 del starter,
frecuencia de blowups y desastres L10. Se probaron extremos 75/85%, bandas
`p_home` 0.40-0.50, 0.45-0.55 y 0.50-0.60, persistencia/regresión y biases
0.03/0.05/0.07.

Ninguna variante mejoró 2025, 2026 y ambas mitades de 2026. Las tendencias de
xwOBA del starter fueron positivas en 2026 pero negativas en 2025; lineup contra
arsenal fue positivo en 2025 y neutral o negativo en 2026. No se agregó ningún
override. Script: `analysis/matchup_quality_override_search.py`.

Las señales situacionales restantes tampoco se promovieron: `mirror_pair` tiene
solo 53 activaciones y `upset_fav60_ctx` 127 en 16,399 filas, y ambas ya entran
al modelo walk-forward. La cobertura es insuficiente para un override adicional.

---

## Persistencia de mediana ofensiva L20 (ACTIVA 2026-07-12)

**Regla general, sin equipos**: calcular la diferencia home-away de
`runs_median_l20`, la mediana de carreras anotadas en los últimos 20 juegos. Si
`abs(diff) >= 2.0` y `0.50 <= p_home < 0.60`, aplicar un bias de `0.04` hacia
el lado con mayor mediana: `sign(diff) * 0.04`. Gate `season >= 2025`.

La señal fue evaluada de forma incremental después de la regresión de déficit
de remontada y del resto del pipeline activo.

| Periodo | Accuracy antes | Accuracy después | Delta |
|---|---:|---:|---:|
| 2025 | 58.98% | 59.14% | **+0.16pp** |
| 2026 | 64.40% | 64.75% | **+0.35pp** |
| 2026 H1 | 65.00% | 65.14% | **+0.14pp** |
| 2026 H2 | 64.29% | 64.84% | **+0.55pp** |

Cambió 7 picks de 2026 con bias `0.04`. La zona `1.75-2.0` fue positiva; un
umbral amplio de `1.5` carreras dañó 2026 y fue descartado. La API carga
`features_game_flow.parquet`, expone `runs_median_bias` y mostró 11 activaciones
reales entre el 25 de marzo y el 30 de abril de 2026. Backtest:
`analysis/process_quality_override_search.py`.

---

## Regresión de déficit promedio de remontada (ACTIVA 2026-07-12)

**Regla general, sin equipos**: calcular la diferencia home-away de
`cb_avg_def_l30`, el déficit promedio de carreras en los juegos que cada lado
logró remontar durante sus 30 juegos previos. Si `abs(diff) >= 0.30` y
`0.50 <= p_home < 0.60`, aplicar un bias de `0.05` contra el extremo:
`-sign(diff) * 0.05`. Gate `season >= 2025`.

Interpretación: un déficit de remontada reciente muy alto no se trata como
fortaleza persistente; se regresa hacia la media dentro de la banda donde el
modelo favorece moderadamente al local.

| Periodo | Accuracy antes | Accuracy después | Delta |
|---|---:|---:|---:|
| 2025 | 58.90% | 58.98% | **+0.08pp** |
| 2026 | 64.19% | 64.40% | **+0.21pp** |
| 2026 H1 | 64.86% | 65.00% | **+0.14pp** |
| 2026 H2 | 64.01% | 64.29% | **+0.28pp** |

Cambió 13 picks de 2026 en el backtest incremental. La zona fue robusta entre
umbrales `0.30-0.333` y biases `0.03-0.05`; cortes más amplios dañaron 2025 y
se descartaron. Script: `analysis/process_quality_override_search.py`.

La API carga `features_comeback.parquet`, expone `comeback_deficit_bias` y fue
verificada con 22 activaciones reales entre el 25 de marzo y el 30 de abril de
2026. No requiere identidad de equipo ni reentrenamiento.

---

## Señales estructurales por perfil de jornada (DESCARTADO 2026-07-12)

Se buscaron señales generales sin identidad de equipo sobre el baseline activo
con PHI y WSH. El barrido combinó perfiles pregame de la jornada con colas de
mercado, movimiento de línea, dispersión entre books, lineup, bullpen,
abridores, descanso, umpire, clima, Elo, forma ofensiva y suerte pitagórica.

Se evaluaron 19,254 combinaciones elegibles. Los cortes se aprendieron en 2025,
se congelaron para 2026 y debían conservar signo en un cuantil adyacente, además
de ser no negativos en ambas mitades de 2026. **Ninguna combinación cumplió**.
No se agregó ningún override. Script:
`analysis/structural_slate_signal_search.py`.

También se probaron 727 cambios condicionales de fuente entre el pipeline de
producción, el blend al 50% y el pick del mercado. Ninguno mejoró 2025, 2026 y
ambas mitades de 2026 simultáneamente. Script:
`analysis/structural_source_switch_search.py`.

Conclusión: las jornadas malas no muestran una corrección general estable por
colas estructurales ni por sustitución de fuente. Las reglas activas permanecen
sin cambios.

---

## WSH resiliente contra pick fuerte (ACTIVA 2026-07-12)

**Regla**: si el pipeline final elige al rival de WSH con confianza mayor a
`8pp` y al menos 25% de la jornada se juega de día, invertir el pick final hacia
WSH. Gate `season >= 2025`.

La señal se encontró después de incorporar PHI visitante al baseline, por lo
que su mejora es incremental. Los cortes conservaron signo positivo al variar
la confianza entre 7-9pp y la proporción diurna entre 20-40%.

| Periodo | Total juegos | Picks afectados | Accuracy antes | Accuracy después | Delta |
|---|---:|---:|---:|---:|---:|
| 2025 | 2,477 | 30 | 58.82% | 58.90% | **+0.08pp** |
| 2026 | 1,441 | 13 | 63.84% | 64.19% | **+0.35pp** |
| 2026 H1 | 313 | 6 | 64.22% | 64.86% | **+0.64pp** |
| 2026 H2 | 1,128 | 7 | 63.74% | 64.01% | **+0.27pp** |

Neto del bucket: `+2` juegos en 2025 y `+5` en 2026. La distribución mensual
es variable, por lo que debe auditarse mensualmente. Se ejecuta después de PHI
visitante y del resto del pipeline. Backtest:
`analysis/bad_day_pick_type_search.py`.

El fade de NYM local en jornadas con poco desacuerdo de mercado fue descartado:
su neto 2025 se volvió negativo al ampliar el umbral, señal de sobreajuste.

---

## PHI visitante en jornada incierta (ACTIVA 2026-07-12)

**Regla**: si PHI juega como visitante, el pipeline final eligió al local y la
media de `abs(p_home - 0.50)` de toda la jornada es `<= 0.055`, invertir el pick
final hacia PHI. Gate `season >= 2025`.

El corte se derivó de perfiles de jornada de 2025 y se validó con el pipeline
completo en 2026. El rango `0.0525-0.0593` mantuvo signo positivo en ambos años;
se eligió `0.055` por ser un corte redondo y conservador.

| Periodo | Total juegos | Picks afectados | Accuracy antes | Accuracy después | Delta |
|---|---:|---:|---:|---:|---:|
| 2025 | 2,477 | 20 | 58.66% | 58.82% | **+0.16pp** |
| 2026 | 1,441 | 10 | 63.43% | 63.84% | **+0.42pp** |
| 2026 H1 | 313 | 2 | 63.58% | 64.22% | **+0.64pp** |
| 2026 H2 | 1,128 | 8 | 63.39% | 63.74% | **+0.35pp** |

Neto dentro del bucket: `+4` juegos en 2025 y `+6` en 2026. Se ejecuta al
final del pipeline para respetar primero biases, traps y series double-down.
La API conserva la cartelera completa durante el cálculo, por lo que consultar
con `team=PHI` produce el mismo pick. Backtest:
`analysis/bad_day_pick_type_search.py`.

---

## Perfiles de jornadas con baja accuracy (DESCARTADO 2026-07-12)

Se reconstruyó el pipeline activo y se agruparon los resultados por fecha para
buscar condiciones detectables antes del primer lanzamiento. El barrido incluyó
tamaño de la jornada, confianza media, porcentaje de juegos cerrados,
desacuerdo modelo-mercado, proporción de picks locales, mezcla día/noche y día
de la semana. También se probaron regímenes temporales basados exclusivamente
en los resultados ya conocidos de los 3, 7 y 14 días anteriores.

Los cortes se aprendieron con 2025 y se congelaron para validar en 2026. Se
exigieron al menos 20 juegos por temporada, neto positivo en 2025 y 2026, y
resultado no negativo en ambas mitades de 2026. Ninguna combinación cumplió.
No se agregó ningún override. Script reproducible:
`analysis/low_accuracy_day_signal_search.py`.

Accuracy diaria del pipeline actual: media 58.69% en 2025 (210 jornadas) y
63.37% en 2026 (110 jornadas). Los días malos son reales, pero no forman un
perfil pregame estable que justifique invertir picks.

## Rachas de jornadas malas y fatiga agregada (DESCARTADO 2026-07-13)

Se amplió el análisis con el pipeline acumulado para estudiar si las jornadas
por debajo de 50% comparten un régimen temporal detectable. En 2026 hubo 15 de
110 jornadas malas. Frente al resto, tuvieron más equipos con descanso corto
(`30.6%` vs `23.3%`), más juegos diurnos (`44.9%` vs `36.5%`) y menor accuracy
el día anterior (`59.3%` vs `66.2%`). No mostraron más viaje: el promedio de
viaje visitante fue menor (`226` vs `323` millas). La asimetría de bullpen
también fue menor, mientras la carga desigual de relevistas de alto leverage
solo subió ligeramente (`13.9` vs `12.8` lanzamientos).

La relación descriptiva más fuerte fue la continuidad: 4 de las 15 jornadas
malas llegaron después de otra mala. En términos condicionales, una jornada
fue mala en `28.6%` de los casos posteriores a un día malo, contra `11.5%`
cuando el día anterior no fue malo. Esta señal usa `shift(1)` y no tiene fuga,
pero la muestra es pequeña.

Se probaron 580 reglas combinando régimen previo, descanso corto, viaje,
burn, alto leverage y lineup con tres bandas de `p_home`. Ninguna mejoró 2025,
2026 y ambas mitades de 2026 con al menos cinco flips. El mejor candidato de
2026 (`+0.28pp`) dañó 2025 y la primera mitad de 2026. No se agregó ningún
override. Script: `analysis/bad_day_regime_cluster_search.py`.

## Jornadas malas y victorias de underdogs 2023-2026 (AUDITORIA 2026-07-13)

Por petición exploratoria se compararon jornadas de al menos ocho juegos con
accuracy menor a 50% contra el resto. En la reconstrucción retrospectiva hubo
74 jornadas malas entre 612 elegibles. Los underdogs ganaron `52.5%` de 935
juegos en jornadas malas y `42.0%` de 6,497 juegos en el resto (odds ratio
`1.53`; Fisher `p=1.46e-9`). La diferencia fue positiva en cada temporada:
`+18.1pp` en 2023, `+17.7pp` en 2024, `+7.1pp` en 2025 y `+13.1pp` en 2026.

La fuerza implícita promedio del favorito fue prácticamente idéntica (`7.39pp`
vs `7.42pp`). El movimiento hacia favoritos no fue consistente por temporada:
solo resaltó en 2026; 2023 no tiene historial de movimiento. Además, el dataset
solo conserva un proveedor y dispersión entre books igual a cero, por lo que
no permite inferir cuotas coordinadas ni manipulación.

La verificación walk-forward disponible desde mayo de 2024 hasta junio de 2026
confirmó el aumento de underdogs: `+19.1pp` en 2024, `+12.6pp` en 2025 y
`+12.3pp` en 2026. El resultado describe un régimen de upsets, pero no es una
señal pregame por sí mismo y no se agregó ningún override. Script:
`analysis/market_upset_regime_audit.py`.

## Platoon completo contra bullpen disponible (DESCARTADO 2026-07-13)

Se construyeron splits móviles leakage-safe del lineup confirmado contra ambas
manos usando las 30 apariciones previas. Para cada rival se identificaron sus
tres relevistas principales por saves y apariciones de los 30 días anteriores,
se descontó disponibilidad según lanzamientos del día previo y se calculó la
mezcla esperada de pitchers zurdos y derechos. El resultado se combinó con el
matchup ya conocido contra la mano del abridor.

Se evaluaron 324 reglas sobre xwOBA, K%, BB%, barrel rate, matchup combinado y
cambio esperado entre abridor y bullpen, con cortes aprendidos en 2025. Ninguna
mejoró 2025, 2026 y ambas mitades de 2026 con al menos seis flips. El mejor
resultado 2026 fue `+0.14pp`, pero perdió `-0.20pp` en 2025 y quedó neutral en
la segunda mitad de 2026. No se agregó ningún override ni carga al runtime.
Script: `analysis/full_platoon_override_search.py`. Artefacto de análisis:
`data/processed/features_full_platoon.parquet` (21,541 filas).

## Regresión catcher-batería K/BB (ACTIVA EXPERIMENTAL 2026-07-13)

Se identificó el catcher titular desde la alineación y se construyeron métricas
lagged de sus 20 juegos previos: wild pitches, robos permitidos, caught stealing,
pickoffs, carreras permitidas y K/BB. También se midieron las ocho aperturas
previas de cada pareja catcher-abridor, continuidad del receptor, descanso y el
matchup contra la agresividad en bases del rival.

Se evaluaron 756 reglas generales. El mejor bolsillo basado en K/BB de la
batería agregaba `+0.42pp` en 2026 y era positivo en ambas mitades, aunque
reducía 2025 en `-0.36pp`. Por ello se adoptó únicamente como régimen 2026.

**Regla activa**: si `season == 2026`, `0.50 <= p_home < 0.60` y la diferencia
home-away absoluta de `battery_kbb_l8` es al menos `2.60`, aplicar bias de
regresión `-sign(diff) * 0.05`. Si falta catcher, abridor o historial, no aplica.

El umbral `2.60` es el percentil 75 (`2.616`) de la primera mitad de 2026,
redondeado. Se congeló antes de validar desde el 19 de mayo: en esa segunda
mitad produjo 7 flips y neto `+5` (`+0.69pp`). En todo 2026 produjo 9 flips y
neto `+7`, equivalente a `+0.49pp`. Los cortes 2.0-3.71 conservaron signo
positivo. Debe retirarse al finalizar 2026 o si dos auditorías consecutivas de
30/60 días son negativas.

La API expone `catcher_battery_bias` y la UI muestra `Batería 2026`. Script:
`analysis/catcher_control_override_search.py`. Artefacto runtime:
`data/processed/features_catcher_control.parquet` (23,606 filas).

---

## Ablación del pipeline y escalado de biases (2026-07-12)

Enfoque: retirar cada override activo, retirar pares y escalar individualmente
cada familia a 0.5x, 0.75x, 1.25x y 1.5x. Script:
`analysis/pipeline_ablation_search.py`.

**Ablación**: ninguna eliminación individual ni por pares mejoró 2026 sin
dañar 2025. Todas las reglas activas se mantienen.

**Escalado ganador**: `HOME_BAND_CALIBRATION` a `1.50x`.

| Periodo | Baseline reconstruido | Delta |
|---|---:|---:|
| 2023 | sin cambio | +0.00pp |
| 2024 | sin cambio | +0.00pp |
| 2025 | 58.62% | +0.04pp |
| 2026 | 63.01% | +0.42pp |

Cambió 12 picks en 2026. Aportó +0.83pp en la primera mitad y quedó neutral
en la segunda. Se conserva el mismo signo y las mismas 15 bandas; únicamente
los biases pasan de +/-0.07 a +/-0.105 mediante un multiplicador central.

---

## PHI en juegos nocturnos (DESCARTADA 2026-07-12)

**Regla**: cuando participa PHI y `day_night == night`, aplicar `0.07` hacia
PHI. Gate `season >= 2025`.

Se revalidó después de activar CWS-night para evitar suma artificial de lifts.
Backtest: `analysis/team_daynight_override_search.py`.

| Periodo | n | Baseline reconstruido | Delta |
|---|---:|---:|---:|
| 2023 | 0 afectado | sin cambio | +0.00pp |
| 2024 | 0 afectado | sin cambio | +0.00pp |
| 2025 | 2,477 | 58.62% | +0.20pp |
| 2026 | 1,441 | 63.01% | +0.14pp |

El backtest incremental cambió 6 picks y parecía aportar +0.14pp en cada
mitad. Sin embargo, el audit aislado posterior mostró 0/1 (-1) en 30 días,
2/4 (0) en 60 días y 3/8 (-2) en 2026. Se retiró antes de producción por net
anual negativo y muestra pequeña. El badge informativo `night_elite` permanece,
pero no cambia picks.

---

## CWS en juegos nocturnos y banda cercana (ACTIVA 2026-07-12)

**Regla**: cuando participa CWS, `day_night == night` y
`0.40 <= p_home < 0.60`, aplicar `0.07` hacia CWS. Gate `season >= 2025`.

Backtest: `analysis/team_daynight_override_search.py`.

| Periodo | n | Baseline reconstruido | Delta |
|---|---:|---:|---:|
| 2023 | 0 afectado | sin cambio | +0.00pp |
| 2024 | 0 afectado | sin cambio | +0.00pp |
| 2025 | 2,477 | 58.50% | +0.12pp |
| 2026 | 1,441 | 62.73% | +0.28pp |

Cambió 10 picks en 2026. Aportó +0.56pp en la primera mitad y quedó neutral
en la segunda; no fue negativo en ninguna. Revalidar mensualmente por tratarse
de una regla específica de equipo.

---

## Volatilidad run differential L10 (WATCHLIST, NO ACTIVA 2026-07-12)

**Idea**: usar la diferencia home-away de `run_diff_std_l10` para seguir o
desvanecer al equipo más volátil, globalmente y por banda de `p_home`.

**Resultado**: `analysis/team_volatility_override_search.py` encontró solo un
candidato marginal: gate 2026, `abs(diff) >= 3.0`, seguir al lado más volátil
con bias `0.05`. Produjo +0.14pp en 2026, pero apenas 6 flippeos, +0.00pp en la
primera mitad y +0.28pp en la segunda, sin validación 2025.

**Decisión**: no implementar por muestra de frontera y falta de confirmación
cross-year. Reabrir cuando acumule al menos 12 flippeos 2026 y el net siga
positivo en ambas mitades.

---

## Aceleración pitagórica L10 vs L30 (DESCARTADA 2026-07-12)

**Idea**: usar `pyth_wpct_l10_diff - pyth_wpct_l30_diff` para seguir o revertir
cambios rápidos de calidad por diferencial de carreras.

**Resultado**: `analysis/pythagorean_acceleration_override_search.py` no
encontró ninguna regla que mejorara 2026 y mantuviera signo no negativo en
ambas mitades sin dañar 2025. El mejor candidato quedó neutral en 2026; varios
ganaban en la primera mitad y perdían más en la segunda. No se implementó.

---

## Persistencia extrema de secuenciación LOB (ACTIVA 2026-07-12)

**Regla**: construir por equipo `lob_net = lob_def_dev - lob_off_dev`. Cuando
`abs(lob_net_h - lob_net_a) >= 0.12` y `0.50 <= p_home < 0.60`, aplicar
`0.07` hacia el lado con mayor señal neta. Gate `season >= 2025`.

Backtest: `analysis/lob_sequencing_override_search.py`.

| Periodo | n | Baseline reconstruido | Delta |
|---|---:|---:|---:|
| 2023 | 0 afectado | sin cambio | +0.00pp |
| 2024 | 0 afectado | sin cambio | +0.00pp |
| 2025 | 2,477 | 58.26% | +0.24pp |
| 2026 | 1,441 | 62.46% | +0.28pp |

En 2026 cambió 8 picks y fue positivo en ambas mitades (+0.42pp y +0.14pp).

---

## Persistencia BABIP neta extrema (ACTIVA 2026-07-12)

**Regla**: cuando `abs(babip_luck_net_h - babip_luck_net_a) >= 0.07` y
`0.40 <= p_home < 0.50`, aplicar `0.02` hacia el lado con mayor señal BABIP
neta. Gate `season >= 2025`.

Aunque la hipótesis inicial era regresión, el backtest favoreció persistencia.
Se interpreta como contacto/defensa parcialmente sostenible y se mantiene bajo
auditoría mensual. Backtest: `analysis/babip_luck_override_search.py`.

| Periodo | n | Baseline reconstruido | Delta |
|---|---:|---:|---:|
| 2023 | 0 afectado | sin cambio | +0.00pp |
| 2024 | 0 afectado | sin cambio | +0.00pp |
| 2025 | 2,477 | 58.13% | +0.12pp |
| 2026 | 1,441 | 62.25% | +0.21pp |

En 2026 cambió 7 picks y fue positivo en ambas mitades (+0.14pp y +0.28pp).

---

## Regresión de suerte pitagórica L30 (RETIRADA 2026-07-17)

**Regla**: cuando `abs(pyth_minus_actual_l30_diff) >= 0.12` y
`0.50 <= p_home < 0.60`, aplicar `0.03` contra el lado cuyo récord reciente
supera más su expectativa por carreras. Gate `season >= 2025`.

La variable está disponible en `train.parquet`, es estrictamente lagged y no
forma parte de `lgb_cls.pkl::feature_names`. Backtest:
`analysis/pythagorean_luck_override_search.py`.

| Periodo | n | Baseline reconstruido | Delta |
|---|---:|---:|---:|
| 2023 | 0 afectado | sin cambio | +0.00pp |
| 2024 | 0 afectado | sin cambio | +0.00pp |
| 2025 | 2,477 | 57.93% | +0.20pp |
| 2026 | 1,441 | 61.97% | +0.28pp |

En 2026 cambió 10 picks y fue positivo en ambas mitades (+0.42pp y +0.14pp).
El baseline reconstruido quedó 0.14pp por debajo del audit independiente de
62.11%, por lo que se prioriza el delta de la regla y debe revalidarse con
`analysis/override_audit.py` en cada actualización mensual.

**Retiro 2026-07-17**: al regenerar `walkforward_preds.parquet` mes por mes para
2025-2026, se comprobo que la evaluacion anterior no era fuera de muestra. Sobre
el pipeline aplicado a probabilidades walk-forward, retirar la regla sumo `+1`
en 2025 y `+5` en 2026, repartidos `+4` H1 y `+1` H2, con 11 cambios. Los cambios
se distribuyeron entre abril-julio y la exposicion maxima fue cuatro juegos de
un equipo. La escala 0.75 tambien fue positiva en ambos anos; 1.25 perdio.
Decision: funcion desactivada en `src` y `backend`. Scripts:
`analysis/walkforward_pipeline_ablation_2526.py` y
`analysis/pythagorean_removal_walkforward_audit.py`.

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

## Incertidumbre y cobertura pregame (WATCHLIST 2026-07-13)

Se probo una familia general sin identidad de equipo usando faltantes en las
181 features del modelo, cobertura reciente del top 4 del lineup, movimiento
de linea, desacuerdo modelo-mercado y sus interacciones. El experimento
reconstruye el pipeline completo actual, incluida la regla catcher-bateria.

| Periodo | Aciertos | Juegos | Accuracy actual |
|---|---:|---:|---:|
| 2023 | 1,662 | 2,471 | 67.26% |
| 2024 | 1,641 | 2,472 | 66.38% |
| 2025 | 1,478 | 2,477 | 59.67% |
| 2026 | 939 | 1,441 | 65.16% |
| Global | 5,720 | 8,861 | 64.55% |

La busqueda general desarrollo 1,052 reglas en 2025 y uso 2026 como holdout;
ninguna tuvo soporte suficiente. La busqueda de regimen desarrollo 820 reglas
en la primera mitad de 2026 y valido en la segunda.

El unico candidato estable fue `lineup_recent_n_with_data` minimo `<=2`, banda
`0.45 <= p_home < 0.55` y bias `0.02` hacia el lado con mayor cobertura: 6
flips, neto `+2`, `+0.14pp`, con neto `+1` en cada mitad. Un bias `0.03`
mantuvo neto `+2` en 10 flips. La variante 0.40-0.60 dio `+4` en 14 flips,
pero esa expansion se descubrio mirando el holdout y no es elegible.

Decision: `WATCHLIST`, no implementada. Tres flips por mitad son insuficientes
despues de cientos de combinaciones. Repetir prospectivamente sin mover el
umbral cuando se acumulen 20-30 activaciones nuevas. Script reproducible:
`analysis/information_uncertainty_override_search.py`.

## Corrido de bases y control defensivo (DESCARTADO 2026-07-13)

Se construyeron 14 señales cronológicas desde `team_box.parquet`: presión de
robo, valor neto SB-CS, control defensivo del running game, pickoffs,
wild pitches/balks, evitación de doble play y matchup combinado, con ventanas
L10 y L20. Todas usan `shift(1)` por equipo y temporada; cobertura aproximada
de 91% después del mínimo histórico.

Se probaron 1,260 reglas generales desarrolladas en 2025 y 1,111 variantes de
régimen desarrolladas en la primera mitad de 2026. Ninguna alcanzó soporte
suficiente en el pipeline exacto. Los mejores resultados 2026 fueron `+3`
netos, pero con solo 5-7 flips; los candidatos con muestra mayor cambiaron de
signo entre mitades.

Decision: `DESCARTADO`, sin parquet runtime, bias, flag ni badge. Script:
`analysis/running_game_override_search.py`.

## Forma ajustada por fuerza de calendario (WATCHLIST 2026-07-13)

Se derivaron 19 señales pregame desde Elo y resultados anteriores: Elo medio
de rivales L5/L10/L20, residuo resultado menos expectativa Elo, desempeño
contra rivales fuertes, como underdog/favorito, en juegos cerrados y
aceleraciones L5 contra L20. No usan identidad de equipo y reinician por
temporada.

Se evaluaron 1,706 reglas generales 2025 a 2026 y 1,343 reglas de régimen H1 a
H2. La candidata principal fue:

- Gate propuesto: 2025+.
- Banda: `0.50 <= p_home < 0.60`.
- Trigger: `abs(elo_result_resid_l5_h - elo_result_resid_l5_a) >= 0.379874`;
  umbral percentil 70 aprendido en 2025.
- Acción: bias `0.05` hacia el lado con mayor residuo Elo L5.
- 2025: 41 flips, neto `+5`.
- 2026: 17 flips, neto `+3`, `+0.21pp`; H1 `+1`, H2 `+2`.
- Por mes 2026: abril `+2`, mayo `0`, junio `+1`, julio `0`.

El signo sobrevivió umbrales entre percentiles 70-85, pero no intensidades
vecinas: con bias 0.03-0.04 la mejora desapareció o se volvió negativa en
varios cortes. Decision: `WATCHLIST`, no implementada. Congelar 0.379874/0.05
y reevaluar prospectivamente tras 15-20 activaciones nuevas. Script:
`analysis/schedule_adjusted_form_search.py`.

## Carreras ajustadas por calidad Elo del rival (WATCHLIST 2026-07-13)

Se calibró con 2023-2024 una expectativa de carreras anotadas y permitidas a
partir del Elo pregame propio, Elo del rival y localía. Después se construyeron
23 señales cronológicas de residuo ofensivo, defensivo, margen total, margen
limitado, dominancia y aceleración L5/L10/L20, siempre con `shift(1)`.

El barrido evaluó 1,980 reglas generales desarrolladas en 2025 y 1,688 reglas
de régimen desarrolladas en la primera mitad de 2026. Ninguna aprobó el veto
estricto completo. La candidata 2026 congelada para seguimiento es:

- Gate: `season == 2026`.
- Trigger: `abs(offense_run_resid_l20_h - offense_run_resid_l20_a) >= 1.185760`;
  percentil 80 derivado de H1 2026.
- Acción: regresión `-sign(diff) * 0.02`, contra la ofensiva que más supera su
  expectativa Elo L20.
- Resultado 2026: 13 flips, neto `+5`, `+0.35pp`.
- H1: 4 flips, neto `+4`; H2: 9 flips, neto `+1`.

La sensibilidad percentiles 80-90 y biases 0.02-0.03 mantuvo signo no negativo;
por ejemplo, percentil 85 con 0.02 dio `+6` en 10 flips. Esa variante no es
elegible porque se descubrió mirando H2. Además, cortes equivalentes aplicados
a 2025 fueron negativos, confirmando que sería una regla de régimen, no una
señal histórica general.

Decision: `WATCHLIST`, no implementada. El holdout H2 aporta solo un acierto
neto y no basta para llamarla señal real después de 3,668 combinaciones. Congelar
el corte 1.185760 y bias 0.02; reconsiderar tras 10-15 activaciones prospectivas.
Script: `analysis/opponent_adjusted_run_margin_search.py`.

## Profundidad y eficiencia reciente del abridor (DESCARTADO 2026-07-13)

Se construyeron métricas pregame desde la línea real de pitcheo del probable
abridor: innings por salida, innings por 90 pitcheos, pitcheos por bateador y
out, salidas antes de cinco innings, frecuencia de seis innings, estabilidad y
tendencias L3 contra L10. Después de retirar ERA/K-BB y variables redundantes,
el barrido final usó 20 señales puras de profundidad/eficiencia.

Se probaron 1,738 reglas generales desarrolladas en 2025 y 989 variantes de
régimen H1 a H2. Ninguna aprobó el holdout estricto. La mejor candidata 2026 fue:

- Banda: `0.40 <= p_home < 0.50`.
- Trigger: diferencia absoluta de tendencia de innings L3-L10 `>=0.675`,
  percentil 70 de H1 2026.
- Acción: regresión `-sign(diff) * 0.05`.
- 2026: 15 flips, neto `+5`, `+0.35pp`.
- H1: 2 flips, neto `+2`; H2: 13 flips, neto `+3` (8-5).

El signo fue positivo cerca de percentiles 65-75 y biases 0.03-0.05, pero el
desarrollo H1 solo contiene dos flips. Más importante, umbrales equivalentes
aprendidos y aplicados en 2025 perdieron entre `-10` y `-18` netos. Decision:
`DESCARTADO`, sin parquet runtime, bias, flag ni badge. Script reproducible:
`analysis/starter_depth_efficiency_search.py`.

## Microestructura externa del mercado multicasa (DATOS INSUFICIENTES 2026-07-13)

Se reconstruyeron snapshots pregame de DraftKings, FanDuel, BetMGM y Pinnacle
desde `data/raw/odds_api`. Para cada evento se calculó probabilidad justa sin
vig, consenso multicasa, diferencia Pinnacle contra casas blandas, dispersión,
movimiento temporal y libros atípicos. No se usó identidad de equipo.

- Cobertura válida: 19 snapshots, 1,050 cotizaciones pregame y 136 eventos.
- Juegos terminados emparejados: 135; 91 tuvieron al menos dos snapshots.
- Pipeline productivo en la muestra: `60.00%`.
- Consenso multicasa al último snapshot: `54.81%`.
- Pinnacle: `58.33%` en 108 juegos con cobertura.
- Ninguna regla de movimiento, divergencia sharp-soft, dispersión o libro
  atípico conservó ganancia entre desarrollo y holdout cronológico.

La tabla runtime `features_market.parquet` no representa un consenso multicasa
en 2026: `market_close.py` selecciona cierres de DraftKings/ESPN BET y la fuente
histórica disponible contiene casi siempre una sola casa ese año. Por ello, las
búsquedas anteriores de modelo contra mercado no podían medir este fenómeno.

Decision: `DATOS INSUFICIENTES`, sin cambio al runtime. No hay evidencia en esta
muestra para atribuir errores a manipulación, casinos o propietarios. Repetir
sin cambiar el protocolo al llegar a 300-500 juegos emparejados, preferiblemente
con apertura y cierre consistentes, y añadir porcentajes de tickets y dinero si
se consigue una fuente fiable. Script:
`analysis/external_market_anomaly_audit.py`.

## Consenso y dispersión multicasa históricos (DESCARTADO 2026-07-13)

Se amplió el ángulo externo usando moneylines pregame no-live de ESPN. El
protocolo usó 2023 exclusivamente para aprender umbrales, 2024 como validación
y los snapshots Odds API recientes de 2026 como holdout final. Se excluyeron
proveedores `Live Odds`, agregadores y cualquier campo que revelara al ganador.

- Desarrollo 2023: 2,469 juegos, mediana de 11 casas; pipeline `67.27%` contra
  `57.15%` del consenso.
- Validación 2024: 383 juegos, mediana de 10 casas; pipeline `66.58%` contra
  `56.92%` del consenso.
- Holdout 2026: 117 juegos, mediana de 4 casas; pipeline `59.83%` contra
  `54.70%` del consenso.

Se probaron consenso mediano/promedio, votación de casas, dispersión, rango,
IQR, skew, incertidumbre de votos, confianza del mercado y divergencia contra
el modelo, además de bandas de probabilidad. Solo tres reglas sobrevivieron
2023 y 2024; las tres proponían ir contra el mercado en `0.45 <= p_home < 0.55`
con divergencia de al menos `0.048259`. En el holdout 2026, cada variante perdió
`-10` netos en 16 flips.

Decision: `DESCARTADO`, sin regla runtime. La señal histórica cambió de signo
de forma contundente en 2026. No volver a barrer cuotas históricas estáticas;
el siguiente experimento externo requiere datos nuevos de apertura-cierre y/o
porcentajes de tickets contra dinero. Script:
`analysis/historical_multibook_market_search.py`.

## Regresión H2H del local dominante en la misma sede (ACTIVA 2026-07-13)

Se auditaron todos los enfrentamientos MLB disponibles: 464 pares canónicos y
925 pares orientados home-away; la diferencia frente a 435 pares teóricos se
debe a aliases históricos como OAK/ATH. Se construyeron señales pregame de
ganador anterior, dominancia L3/L5, margen L3/L5, win rate suavizado, historial
específico de sede y portafolios de calibración aprendidos en años anteriores.

La única familia que pasó desarrollo 2023-2024, validación 2025 y holdout 2026
fue regresión contra una dominancia extrema del local en esta misma sede:

- Mínimo tres encuentros anteriores con el mismo home y away.
- Win rate local suavizado con prior Beta(2.5, 2.5) de al menos `0.70`.
- Banda del modelo: `0.50 <= p_home < 0.60`.
- Acción: bias `-0.07` a `p_home`.
- Los ocho pares H2H fijos quedan excluidos para no apilar dos ajustes.
- Todos los juegos de la misma fecha se ocultan al calcular el historial, por
  lo que una doble cartelera nunca usa un resultado todavía no disponible.

Backtest incremental exacto sobre el pipeline completo:

| Periodo | N | Baseline | Con regla | Neto | Flips |
|---|---:|---:|---:|---:|---:|
| Global | 8,864 | 64.60% | 64.77% | +15 | 43 |
| 2023 | 2,471 | 67.26% | 67.34% | +2 | 4 |
| 2024 | 2,472 | 66.38% | 66.63% | +6 | 22 |
| 2025 | 2,477 | 59.67% | 59.83% | +4 | 12 |
| 2026 | 1,444 | 65.44% | 65.65% | +3 | 5 |

En 2026 aportó `+1` en H1 y `+2` en H2. La sensibilidad encontró 53 variantes
vecinas positivas en la cola de dominancia local y cero en la cola contraria;
por eso solo se implementó el fade del local dominante. El runtime expone
`h2h_venue_regression_bias` y la UI muestra `Regresión H2H` cuando cambia el
pick. Script: `analysis/h2h_all_matchups_search.py`.

## Resaca de extrainnings y juegos cerrados (DESCARTADA 2026-07-13)

Se separaron del `burn_score` los efectos del juego anterior: extrainnings,
duracion, innings adicionales, resultado por una carrera, tiempo real de
recuperacion y viaje. El historial usa exclusivamente el ultimo juego de una
fecha anterior; una doble cartelera nunca comparte resultados entre juegos.
No se usaron identidades de equipo.

- Muestra: 8,864 juegos; 456 con exactamente un lado viniendo de extrainnings.
- Descubrimiento: 2023-2024; validacion: 2025; veto final: 2026 y sus dos
  mitades cronologicas.
- La mejor senal estable fue favorecer al lado descansado cuando el rival venia
  de extras con hasta 24 horas de recuperacion: `+2` netos en desarrollo,
  `+2` en 2025 y `+1` en 2026.
- Sin embargo, solo produjo seis flips en todo desarrollo, muy por debajo del
  minimo de 20 requerido para estimar un bias sin sobreajuste.
- Las variantes con mas cobertura cambiaron de signo; por ejemplo, extras mas
  viaje de 750 millas dio `+2` en desarrollo, `-7` en 2025 y `-2` en 2026.

Decision: `DESCARTADA`, sin cambio runtime. La muestra positiva de recuperacion
ultracorta queda como hipotesis para monitoreo, no como regla. Script:
`analysis/prior_game_hangover_search.py`.

## Forma de primera entrada para ganador (DESCARTADA 2026-07-13)

Se reconstruyeron tasas pregame de anotacion en primera entrada L20 por equipo
y carreras permitidas en primera entrada L15 por abridor. Los rolling ocultan
todos los resultados de la fecha actual. Se probaron ventaja ofensiva, ventaja
del abridor, presion lineup-abridor, blend, acuerdo entre ambas senales y nivel
total de presion; los umbrales se derivaron solo de 2023-2024.

- Cobertura: 8,864 juegos para ofensiva y 7,959 para abridores/cruces.
- El fade del acuerdo extremo en `0.45 <= p_home < 0.55` fue la unica familia
  que sostuvo desarrollo y validacion con biases exactos.
- Variante mas estable (`abs(fi_agree_adv) >= 0.20`, bias `0.05`): `+5` en
  2023, `0` en 2024 y `+4` en 2025.
- En el holdout 2026 perdio `-5` netos en 13 flips: `0` en H1 y `-5` en H2.
- Umbrales/biases vecinos tambien perdieron entre `-4` y `-12` en 2026.

Decision: `DESCARTADA`, sin cambio runtime. La relacion historica cambio de
signo en 2026 y no debe incorporarse al ganador. Script:
`analysis/first_inning_winner_signal_search.py`.

## Normalizacion condicional del peso hacia MIL (ACTIVA 2026-07-13)

La regla anterior invertia cualquier pick pre-series que fuera contra MIL en
temporadas 2025+. Se reemplazo por un gate pregame general: el flip hacia MIL
solo se permite cuando la diferencia absoluta entre la probabilidad de MIL del
modelo y la probabilidad de MIL del mercado es `<=0.07`. Si el desacuerdo es
mayor, se conserva el pick original. No se usa resultado del juego.

Backtest incremental exacto contra el pipeline previo:

| Periodo | Pick rate MIL antes | Pick rate MIL despues | Neto |
|---|---:|---:|---:|
| 2025 | 97.66% | 91.81% | 0 |
| 2026 | 85.42% | 81.25% | +2 |
| 2026 H1 | - | - | +1 |
| 2026 H2 | - | - | +1 |
| Global | - | - | +2 |

La accuracy productiva queda en `64.79%` global y `65.79%` para 2026. En los
96 juegos de MIL de 2026, la accuracy sube de `65.62%` a `67.71%`. Los cortes
vecinos `0.06` y `0.07` dieron el mismo `+2` en 2026; se eligio `0.07` por ser
el reemplazo mas conservador. Script:
`analysis/mil_conditional_normalization_search.py`.

## Normalizacion de otros equipos muy seleccionados (DESCARTADA 2026-07-13)

Se descompuso el pick rate de todos los equipos entre mercado, modelo base sin
reglas de identidad y pipeline productivo. Despues de normalizar MIL, los
candidatos con al menos 70% de seleccion historica o en 2026 fueron LAD, ATL y
PHI. Se probaron fades cuando produccion contradice al modelo/mercado, soporte
conjunto bajo y divergencias de 3-15pp, tanto globales como con gate 2025+.

| Equipo | Produccion historica | Modelo base | Mercado | Produccion 2026 |
|---|---:|---:|---:|---:|
| LAD | 85.78% | 88.05% | 83.84% | 87.63% |
| ATL | 78.02% | 81.60% | 72.74% | 62.11% |
| PHI | 72.52% | 67.88% | 71.03% | 73.20% |

LAD y ATL no estan sobreponderados por las reglas runtime: produccion los elige
menos que el modelo base. PHI recibe `+4.64pp` frente al modelo base, pero queda
a solo `+1.49pp` del mercado. Sus normalizaciones positivas en 2026 perdieron
en 2025; por ejemplo, el fade PHI con gap modelo-mercado `>=0.07` dio `-1` en
2025 y `+2` en 2026. Las variantes ATL aparentemente positivas cambiaban solo
un juego por temporada y no alcanzaron el minimo de tres flips en ambos anos.

Decision: `DESCARTADA`, sin cambio runtime. MIL sigue siendo el unico equipo
con normalizacion condicional activa. Script:
`analysis/automatic_team_normalization_search.py`.

## Calidad de victorias y juegos cerrados (DESCARTADA 2026-07-16)

Se midieron, sin usar resultados de la fecha actual, suerte en juegos de una y
dos carreras, margen recortado sin palizas, dependencia de victorias amplias,
proporcion de triunfos fragiles y diferencia media-mediana en ventanas L20/L40.
Los umbrales se aprendieron exclusivamente en 2025 y 2026 quedo como holdout.

La mejor variante exacta de 2025 (`mean_minus_median_l40`, banda 50-60%, 0.10)
sumo `+22` en 72 flips, pero perdio `-12` en 2026 (`-9` H1, `-3` H2). Ninguna
configuracion paso el veto. Decision: `DESCARTADA`, sin cambio runtime. Script:
`analysis/win_quality_close_game_search.py`.

## Ablacion reconstruida 2025/2026 (SUPERADA 2026-07-17)

**No usar para aprobar o rechazar reglas futuras.** Esta pasada aplico el modelo
guardado sobre `train.parquet`, por lo que replica el runtime retrospectivo pero
no es fuera de muestra. La ablacion mensual walk-forward posterior reemplaza
este dictamen y retiro `pythagorean_luck`.

Se reconstruyeron exactamente los 20 componentes de bias activos; el error
maximo de reconstruccion fue `2.78e-17`. Baseline: `59.83%` en 2025 y `65.79%`
en 2026. Se retiro un componente por vez y se exigio no perder en 2025, ganar
en 2026 y no perder en ninguna mitad de 2026.

Ninguna eliminacion paso. Quitar rachas situacionales perdio `-22` en 2026,
interliga `-16`, viaje `-11`, pitchers `-10` y catcher/bateria `-7`. La unica
ganancia total, retirar burn resilience (`+1`), perdio `-1` en 2025 y `-1` en
H2 2026. Decision: mantener todos los componentes. Script:
`analysis/current_pipeline_ablation_2526.py`.

## Selector online motor-mercado (DESCARTADO 2026-07-16)

Se probaron 288 selectores estrictamente walk-forward. Cada juego solo podia
consultar desacuerdos resueltos en fechas anteriores; la configuracion se
eligio en 2025 sin mirar 2026. El candidato congelado (`gap_model_side`, ventana
120 dias, minimo 60, posterior 0.55) sumo `+9` en 71 cambios de 2025, con `+6`
en H1 y `+3` en H2. En 2026 no activo ningun caso, por lo que no puede mejorar
accuracy ni justificar complejidad runtime. Decision: `DESCARTADO`, sin cambio.
Script: `analysis/online_walkforward_recalibration_search.py`.

## Ejecucion defensiva normalizada (PENDIENTE 2026-07-16)

Se preparo `analysis/defensive_execution_signal_search.py` para errores por 100
oportunidades, carreras no limpias, juegos multi-error y costo del error, con
retencion completa de la fecha actual. La pasada exacta resulto demasiado lenta
y no produjo un dictamen valido. No se aplico ninguna regla; antes de retomarla
se debe cachear la porcion determinista del pipeline.

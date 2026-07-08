# Patrones potencialmente explotables — lluvia de ideas

Documento vivo para anotar hipótesis de patrones que valga la pena investigar.
No es un plan de implementación — es el pool de ideas donde vamos escogiendo.

**Regla de oro**: antes de codear cualquier feature, primero hacer *signal check*
con `walkforward_preds.parquet` + `train.parquet` para ver si el patrón sobrevive
condicional al closing line. Si r < 0.02 con outcome, es ruido. Si el mercado
ya lo precia, no hay edge.

---

## 0. Baseline vigente (para comparar cualquier idea)
- accuracy@segmented: **0.5712**
- AUC: **0.5887**
- log_loss: **0.6771**
- Techo empírico del mercado (Pinnacle + DK closing): ~0.567
- Ya batimos al mercado por ~0.4pp

---

## 1. Contexto del día — ¿qué tan "raro" es el juego?

Categoría donde tú (Emilio) tienes intuición: días con pocos juegos = juegos raros.

### 1.1 Slate size
- **Tu observación**: días con <8-9 juegos → juegos "más raros"
- Hipótesis: menos juegos = mercado presta más atención concentrada → menos ineficiencia. O al revés: días raros son off-days de MLB con equipos B (2do lineup, day game post-nocturno).
- Cómo probarlo: dividir juegos por slate size (1-4, 5-7, 8-10, 11-13, 14+) y ver desviación del modelo vs realidad, controlando por closing line.
- Métrica a mirar: no solo accuracy, sino **volatilidad de resultados** (varianza en scores, blowouts, extra innings).

### 1.2 Doubleheader (2do juego)
- Bullpen fatigado del 1er juego, lineup con reservas → juegos más impredecibles
- Data disponible: `double_header` en games.parquet
- Podría ser feature explícito o rule para bajar confianza del modelo

### 1.3 Getaway game (último de serie antes de viajar)
- Equipo visitante checked-out mentalmente
- Data: computable de games.parquet (¿próximo juego del equipo es en otro venue?)

### 1.4 Día siguiente a extra innings del previo
- Bullpen quemado, batters cansados
- Ya tenemos `burn` como analysis pero NO como feature del modelo (falló al meterlo)
- Posible ángulo: como *post-process rule* en vez de feature (rebajar confianza si burn_score > X)

### 1.5 Home team saliendo de road trip largo (o entrando a homestand largo)
- Cambio de body clock, familias de vuelta, sueño diferente
- Computable de secuencia home/away en games.parquet

---

## 2. Dinámica del starter

### 2.1 Times-through-the-order penalty
- La 3a vez que un batter enfrenta al starter, su xwOBA sube ~30%
- Si podemos estimar hasta cuántas veces vuelve el starter (por promedio de innings recientes), podemos aportar señal
- Data: pitcher_form recientes tienen IP promedio

### 2.2 Rookie starter / MLB debut
- Data desconocida para modelo (no L15), lineup no lo ha visto → volatilidad alta
- Detectable: si starter no aparece en historia previa reciente → rookie o call-up

### 2.3 Bullpen game / opener strategy
- Cuando el "starter" solo lanza 1-2 innings intencionalmente
- Detectable: pitcher's L15 promedio de IP < 3 → opener probable
- Mercado a veces tarda en ajustar líneas de spread/total

### 2.4 Return from IL (Injured List)
- Primer juego de vuelta = incertidumbre alta
- Data: si pitcher no lanzó en 15+ días pero antes era regular → return from IL likely

### 2.5 Emergency starter (bullpen forced)
- Cuando hay cambio de último minuto (ace lesionado, cambio de rotación)
- Detectable si probable_pitcher_id publicado hoy difiere de lo esperado

---

## 3. Dinámica del bullpen (más allá de IP acumulada)

### 3.1 Closer disponibilidad
- Closer usado 3 noches seguidas → no disponible hoy
- Data: player_box con `save` outcomes en los últimos 3 días

### 3.2 Bullpen QUALITY (no solo IP)
- Team con bullpen top-5 vs bottom-5 (Reliever xFIP season aggregate)
- Ya tenemos features de IP acumulada; NO tenemos quality
- Idea: xFIP promedio de los 3 mejores relievers season-to-date

### 3.3 Bullpen match-up
- Si team con bullpen malo enfrenta un lineup potente en late innings → sesgo predecible

---

## 4. Dinámica del lineup

### 4.1 Cambio de lineup vs anterior
- Bench del star player OUT hoy
- Detectable: comparar top-4 batters de hoy vs promedio L5

### 4.2 Lineup RISP (Runners In Scoring Position) recent performance
- Clutch/no-clutch reciente
- Data disponible en pitches.parquet

### 4.3 Lineup vs THIS pitcher's arsenal (no solo hand)
- Pitcher con slider % alto contra lineup que baja mucho contra sliders
- Requiere pitch-type breakdown

### 4.4 Handedness stacking
- Cuando manager pone lineup LHH-pesado vs starter que sufre con LHH
- Ya tenemos algo pero puede refinarse

---

## 5. Motivación / situacional

### 5.1 Playoff race pressure (septiembre)
- Equipo peleando Wild Card juega distinto que uno eliminado
- Data: standings + games remaining

### 5.2 Equipo ya clasificado / eliminado
- Bajan intensidad las últimas 2 semanas
- Mercado a veces no ajusta rápido

### 5.3 Manager fired recently (dead-cat bounce)
- Bump típico +3pp win rate primeros 10 juegos post-firing
- Data: news feed necesario (no la tenemos, sería scraping)

### 5.4 Star player just returned from IL
- Boost al lineup
- Requiere injury data

### 5.5 Trade deadline reshuffle
- Rosters mezclados, chemistry no calibrada
- Timing conocido (fecha fija)

### 5.6 Rivalry games (NYY-BOS, LAD-SF)
- Intensidad extra
- Detectable por matchup

---

## 6. Weather × Park interactions (más profundo)

### 6.1 Wind blowing out at Wrigley
- Bump histórico a totales, y a batter-friendly outcomes
- Data: `weather_wind` en games.parquet, pero como raw feature falló
- Posible: rule específica solo en Wrigley + viento

### 6.2 Coors Field amplifier
- Todo se magnifica en Denver
- Puede ser interacción explícita: park_id="Coors" × weather_temp

### 6.3 Cold weather (early April, late September)
- Home run rates bajan, K rates suben
- Ya tenemos `weather_temp_f` pero interacciones son sutiles

### 6.4 Roof open vs closed at hybrid parks (Marlins, Astros, Rangers)
- Cambia physics del juego
- Data: `roof_type` disponible

---

## 7. Umpire × juego

### 7.1 Umpire tendencies × pitcher type
- Ump generoso (large zone) × K-pitcher = ↑K, ↓runs
- Ump apretado × control pitcher = poco cambio
- Ya tenemos ump features pero NO las cross features con pitcher

### 7.2 Umpire × park
- Algunos umps son "diferentes" en Fenway (foul territory small)
- Cross feature no explorado

### 7.3 Umpire fatiga (juegos consecutivos)
- Ump con 4 juegos en 5 días llama distinto
- Requiere schedule del umpire

---

## 8. Señales de mercado (donde vive la mayoría del edge)

### 8.1 Line movement en múltiples ventanas
- Ya tenemos `market_line_shift_home_pp` (open→close). +0.07pp acc.
- Falta: shift en las últimas 2h, 4h, 24h. Requiere scheduler de snapshots.

### 8.2 Reverse line movement (RLM)
- Mayoría del volumen del público en un lado pero línea se mueve contra
- Requiere data de betting % público (no la tenemos, tal vez de PickPredict/BetLabs)

### 8.3 Book divergence explícita
- `market_p_home_std` existe pero underused
- Cruzar con dirección: cuando std alto + market pica home fuerte = book split → sharp opportunity

### 8.4 Steam moves
- Múltiples books mueven líneas simultáneamente en corto tiempo = sharp money
- Requiere time-series

### 8.5 Off-market book (Pinnacle vs consensus)
- Ya lo tenemos como post-process (`pinnacle_value`)
- Podría amplificarse como feature explícita para el modelo

---

## 9. Sesgos del público (fade-the-public)

### 9.1 Marquee teams overvalued
- NYY, LAD, BOS, ATL siempre más caro de lo que merecen
- Detectable: market_p_home para esos equipos vs modelo
- Puede ser rule condicional

### 9.2 Overs bias
- Público adora overs
- Total closes shaded high vs realidad
- Ya tenemos total_shift, pero podríamos cruzar con market_over_under absoluto

### 9.3 Prime-time featured games (Sunday Night, ESPN)
- Más volumen público → más ineficiencia
- Requiere marker de "featured game" (podría scrapear ESPN)

### 9.4 Home favorite bias
- Público prefiere apostar home
- Books shade slightly
- Podría ser rule en la banda 0.52-0.58 home fav

---

## 10. Meta / arquitectura

### 10.1 Model confidence bands recalibration
- Ya tuneado, pero cada mes cambia
- Podría auto-recalibrar mensualmente con últimos 200 juegos

### 10.2 Segmented model (uno para close games, otro para blowouts)
- Modelo dedicado para market_p_home ∈ [0.42, 0.58]
- Diferentes features quizás más útiles en esa banda

### 10.3 Meta-model: predecir cuándo el modelo se equivoca
- Modelo que aprende de errores del principal
- Usar output como feature de correction

### 10.4 Ensemble con arquitecturas distintas
- Ya probamos XGB, no ayudó (mismos sesgos que LGB)
- Nuevas ideas: Random Forest, GLM logistic, Neural Net simple

---

## 11. Ideas raras / creativas (no descartar)

### 11.1 First game after West-to-East red-eye (equipo LA/SF viajando al este)
- Cross-country red-eye = jetlag
- Ya probamos travel, falló. Pero solo para esta subcondición podría funcionar.

### 11.2 Team playing 4th consecutive road game
- Fatiga acumulada específica
- Requiere secuencia home/away

### 11.3 Sunday day game after Saturday night game
- Rest corto + fin de semana
- Cross de dow + prev_daynight

### 11.4 Mid-week single game (Wed off, Thu single game vs weekend)
- Público menos engaged → market softer

### 11.5 Interleague designated hitter split
- AL team without regular DH lineup, or NL forced to DH
- Cambia dinámica del lineup

---

## 12. Ya probado y descartado (NO reproponer)

Para no perder tiempo repitiendo:
- Pythagorean como feature (post-process ok)
- luck/BABIP features
- Creative batch (streak, momentum, fatigue, linemove, circadian raw)
- Calendar (slate_size + dow) como features crudas (+0.04pp)
- Team volatility (std runs L10)
- Burn como feature (falla, pero como post-process podría ser)
- Game flow features
- bullpen_ip_min_l3
- Pitcher L5-vs-L15 divergence
- Travel/jetlag como features
- team_model_acc lagged
- Threshold recalibration
- Tiles contextuales (mes × dow, o mes × dow × hora × slate)
- Graded blend por magnitud
- Stacking XGBoost + LightGBM
- **Slate size <=9 fade home rule (2026-07-02)**: signal check original mostró home_wr 34-41% en slates chicos con residual -18pp, PERO era artefacto de first_pitch_utc NaN filtrando data. Signal check corregido con game_date: r=-0.0001 con home_won, patrón no-monotónico (chispazo en 5-7 y 10-12 que se cancelan). Rule aplicada dio -0.20pp acc en walkforward. Descartada tanto B (rule) como C (feature binaria).
  - **Lección**: siempre verificar el N total del signal check vs el walkforward preds. Si es << total, hay filtro sesgando.
- **Bullpen quality features (2026-07-02, 3 variantes)**: season aggregate (K/9, BB/9, FIP, WHIP, K-BB%), recent form last 10 team games, y form delta (recent - season). Módulo en `src/features/bullpen_quality.py` (bien construido, strictly lagged, sin data leakage). Signal checks todos con r_residual entre -0.025 y +0.021, bajo threshold de 0.03. Bucketing muestra patrones no-monotónicos con pico en "similar" (signature de ruido). Conclusión: **el mercado precia bullpen quality perfectamente** — ambos season y recent form. El módulo queda en disco por si futuro se cruza con otra idea. NO reintentar solo.

---

## Notas de priorización (mi lectura)

**Alta expectativa (probar primero)**:
- 1.1 slate size con signal check refinado (tu intuición)
- 3.2 bullpen quality (xFIP top 3 relievers)
- 8.3 book divergence cruzado con dirección
- 2.1 times-through-the-order proxy

**Media**:
- 5.1 playoff race pressure
- 6.1 Wrigley + wind rule
- 7.1 umpire × pitcher K%
- 8.1 line movement en múltiples ventanas (requiere scheduler)

**Baja / alto costo**:
- 9.x public bias (requiere data de betting %)
- 5.3 manager fired (requiere news feed)
- 10.3 meta-model (arquitectura compleja)

---

## Cómo procesar cada idea

1. **Signal check**: correr análisis de correlación con outcome, condicionando por closing line
2. **Umbral**: r > 0.05 = seguir con feature. r < 0.02 = descartar. r 0.02-0.05 = intentar como interacción/rule
3. **Diseño**: feature del modelo, post-process rule, o segmented model
4. **Walkforward**: correr con la métrica de disciplina (acc > AUC > log_loss)
5. **Decisión**: los 3 arriba → keep. Baja acc → revert. Mixto → discutir.

---

*Este archivo es para acumular ideas. Cuando probemos una, mover al final a "Ya probado" con resultado.*

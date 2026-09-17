'use client';

import { useEffect, useState } from 'react';
import './lineup-signal-audit.css';

type Stats = { n: number; wins?: number; winRate?: number; winCI95?: number[]; runsFor?: number; runsAgainst?: number; totalRuns?: number; marketResolved?: number; pushes?: number; marketOver?: number | null; baseAccuracy?: number; deltaVsBasePoints?: number; qVsBaseAllRules?: number; qTotalVs50AllRules?: number };
type Rule = { name: string; discovery: Stats; validation: Stats; selectedBeforeValidation: boolean; selectedWinner: boolean; selectedTotals: boolean; totalDirection: 'over' | 'under' };
type Metric = { games: number; correct: number; accuracy: number; brier: number };
type Pair = { games: number; before: Metric; after: Metric; deltaPoints: number; deltaCI95Points: number[] };
type Report = { asOf: string; games: number; discoveryGames: number; validationGames: number; rulesTested: number; rulesWith100DiscoveryGames: number; method: string; totalsMethod: string; rules: Rule[]; models: Record<string, Pair>; versusOldAlignmentSameGames: Record<string, Pair>; totals: { baselineMAE: number; lineupMAE: number; market?: { games: number; resolved: number; pushes: number; directionAccuracy: number; baselineDirectionAccuracy: number; baselineDirection: string; deltaCI95Points: number[]; lineMAE: number; adjustedMAE: number } } };
const pct = (n?: number | null) => n == null ? '—' : `${(100*n).toFixed(1)} %`;
const num = (n?: number | null) => n == null ? '—' : n.toFixed(2);
const label = (s: string) => s.replace(/\b(hot|solid|normal|cold|unknown|positive|dominant|trouble|Top4|Elite)\b/g, k => ({ hot: 'calientes', solid: 'sólidos', normal: 'normales', cold: 'fríos', unknown: 'sin etiqueta', positive: 'calientes + sólidos', dominant: 'dominante', trouble: 'en problemas', Top4: 'Primeros 4', Elite: 'Élite' }[k] ?? k));

export default function LineupSignalAudit() {
  const [data, setData] = useState<Report | null>(null);
  const [error, setError] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const [query, setQuery] = useState('');
  const [scope, setScope] = useState('shortlist');
  useEffect(() => {
    const controller = new AbortController();
    fetch('/data/lineup-audit.json', { signal: controller.signal }).then(r => { if (!r.ok) throw Error(); return r.json() as Promise<Report>; }).then(setData).catch(e => { if (e.name !== 'AbortError') setError(true); });
    return () => controller.abort();
  }, [attempt]);
  if (!data) return <section className="section lineup-research"><h2>Patrones de alineación</h2>{error ? <p>No se pudo cargar el estudio. <button onClick={() => { setError(false); setAttempt(n => n+1); }}>Reintentar</button></p> : <p role="status">Cargando estudio…</p>}</section>;
  const normalize = (s: string) => s.normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase();
  const rules = data.rules.filter(r => (scope !== 'shortlist' || r.selectedBeforeValidation) && normalize(label(r.name)).includes(normalize(query)));
  const old = data.versusOldAlignmentSameGames.lineup;
  return <section className="section lineup-research">
    <span className="eyebrow">LABORATORIO L30 · CORTE {data.asOf}</span><h2>¿Las rachas del lineup anticipan el resultado?</h2>
    <p>Exploración de cantidades, primeros cuatro bateadores, talento de temporada, cobertura y pitchers. “Sin etiqueta” no se cuenta como “normal”. Ninguna regla de este estudio está activada en los picks.</p>
    <div className="lineup-research-summary"><article><small>PARTIDOS</small><strong>{data.games.toLocaleString('es-MX')}</strong><span>{data.discoveryGames} descubrimiento / {data.validationGames} validación</span></article><article><small>REGLAS EXPLORADAS</small><strong>{data.rulesTested}</strong><span>{data.rulesWith100DiscoveryGames} con ≥100 juegos previos a julio</span></article><article><small>ANTES · ALINEACIÓN ACTUAL</small><strong>{pct(old.before.accuracy)}</strong><span>{old.before.correct}/{old.games} · reconstrucción histórica</span></article><article><small>DESPUÉS · MODELO DE ETIQUETAS</small><strong>{pct(old.after.accuracy)}</strong><span>{old.after.correct}/{old.games} · exactamente los mismos juegos</span></article></div>
    <div className="lineup-research-note"><strong>Sin mejora demostrada.</strong> Diferencia {num(old.deltaPoints)} puntos; intervalo 95 %: {old.deltaCI95Points.map(num).join(' a ')}. Si el intervalo incluye cero, la diferencia puede ser ruido. No convertir un patrón atractivo en una recomendación automática.</div>
    <h3>Victoria: ¿aporta más que la base?</h3><p>Modelos entrenados solo antes de julio; evaluación julio–15 de septiembre. Las tres variantes comparten muestra. El modelo de etiquetas usa regularización fija, sin ajustar parámetros con los resultados de validación.</p>
    <div className="lineup-research-scroll"><table><thead><tr><th>Modelo</th><th>Juegos</th><th>Acierto</th><th>Brier ↓</th></tr></thead><tbody>{Object.entries(data.models).map(([id, p]) => <tr key={id}><td>{{base:'Base StatsMLB',lineup:'Etiquetas + orden + talento + pitcher',base_plus_lineup:'Base + señales de alineación'}[id] ?? id}</td><td>{p.games}</td><td>{pct(p.after.accuracy)}</td><td>{p.after.brier.toFixed(4)}</td></tr>)}</tbody></table></div>
    <h3>Explorar cruces</h3><p>Enero–junio descubre; julio–corte comprueba. Las 40 reglas preseleccionadas se eligieron antes de mirar la validación. Una regla de victoria solo selecciona cuando un equipo cumple y el otro no; si ambos cumplen, se abstiene. “Ambos…” describe totales, no un favorito.</p>
    <div className="lineup-research-filters"><label>Buscar patrón<input value={query} onChange={e => setQuery(e.target.value)} placeholder="calientes, fríos, sin etiqueta, pitcher…" /></label><label>Mostrar<select value={scope} onChange={e => setScope(e.target.value)}><option value="shortlist">Preseleccionados antes de julio</option><option value="all">Todos con muestra suficiente</option></select></label><span role="status">{rules.length} patrones</span></div>
    <div className="lineup-research-scroll"><table><caption>Victorias y producción ofensiva · porcentajes descriptivos, no probabilidades del siguiente juego</caption><thead><tr><th>Patrón</th><th>Antes de julio: n / victoria</th><th>Julio–corte: n / victoria</th><th>Base, mismos juegos</th><th>Carreras a favor / contra</th><th>q ajustado vs base</th></tr></thead><tbody>{rules.map(r => <tr key={r.name}><td>{label(r.name)}{r.selectedWinner && <small>Preseleccionado para victoria</small>}</td><td>{r.discovery.n} / {pct(r.discovery.winRate)}</td><td>{r.validation.n} / {pct(r.validation.winRate)}</td><td>{pct(r.validation.baseAccuracy)}</td><td>{num(r.validation.runsFor)} / {num(r.validation.runsAgainst)}</td><td>{num(r.validation.qVsBaseAllRules)}{r.validation.deltaVsBasePoints != null && <small>Δ {num(r.validation.deltaVsBasePoints)} pts</small>}</td></tr>)}</tbody></table></div>
    <p>q ajusta por comparar muchas reglas. Un valor pequeño puede señalar una diferencia <em>negativa</em>: comprueba también el signo de Δ. Las muestras se superponen y no se deben sumar.</p>
    <h3>Altas y bajas: líneas reales, muestra separada</h3><p>{data.totalsMethod}</p>
    {data.totals.market && <div className="lineup-research-note">Modelo de totales: {pct(data.totals.market.directionAccuracy)} de acierto en {data.totals.market.resolved} resultados sin push. Referencia fija elegida antes de julio ({data.totals.market.baselineDirection === 'over' ? 'altas' : 'bajas'}): {pct(data.totals.market.baselineDirectionAccuracy)}. IC95 de la diferencia: {data.totals.market.deltaCI95Points.map(num).join(' a ')} puntos. Error medio de la línea: {num(data.totals.market.lineMAE)} → {num(data.totals.market.adjustedMAE)} carreras con ajuste.</div>}
    {data.totals.market && <p>Como comprobación descriptiva, elegir siempre el lado contrario a esa referencia habría acertado {pct(1-data.totals.market.baselineDirectionAccuracy)} en este periodo. El 54 % del modelo no demuestra por sí solo que las etiquetas aporten una ventaja estable. Solo {data.totals.market.games} juegos cuentan con línea; la cobertura de fuentes cambia entre periodos.</p>}
    <div className="lineup-research-scroll"><table><thead><tr><th>Patrón</th><th>Lado elegido antes de julio</th><th>Antes: n / altas</th><th>Después: n / altas</th><th>Push después</th><th>Total medio después</th><th>q vs 50 %</th></tr></thead><tbody>{rules.map(r => <tr key={r.name}><td>{label(r.name)}{r.selectedTotals && <small>Preseleccionado para totales</small>}</td><td>{r.totalDirection === 'over' ? 'Altas' : 'Bajas'}</td><td>{r.discovery.marketResolved ?? 0} / {pct(r.discovery.marketOver)}</td><td>{r.validation.marketResolved ?? 0} / {pct(r.validation.marketOver)}</td><td>{r.validation.pushes ?? 0}</td><td>{num(r.validation.totalRuns)}</td><td>{num(r.validation.qTotalVs50AllRules)}</td></tr>)}</tbody></table></div>
    {!rules.length && <p>No hay patrones con ese filtro.</p>}
    <h3>Límites que importan</h3><p>{data.method}</p><p>La foto histórica usa titulares del boxscore, no prueba a qué hora se confirmó el lineup. Los jugadores con poca muestra permanecen separados. Los datos ya explorados no equivalen a una prueba prospectiva; faltan nuevas predicciones archivadas antes del primer lanzamiento. No se calcularon beneficios ni rentabilidad.</p>
    <a href="/data/lineup-audit.json" download>Descargar estudio completo, reglas sin muestra y hashes de las fuentes</a>
  </section>;
}

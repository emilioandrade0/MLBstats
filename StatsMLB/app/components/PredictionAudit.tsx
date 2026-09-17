'use client';

import { useEffect, useState } from 'react';

type Metric = { games: number; correct: number; accuracy: number | null; brier: number | null };
type Pair = { before: Metric; after: Metric; deltaPoints: number; netCorrect: number; deltaCI95Points: number[] };
type Period = 'validation2026' | 'h1' | 'h2';
type Report = {
  dataThrough: string; asOf: string;
  method: { selection: string; limitations: string; interval: string };
  coverage: { games: number; maskCount: number; currentPlayerCutoff: string };
  selected: { label: string; activated: boolean; reason: string };
  beforeAfter: Record<Period, Pair>;
  probabilityCorrection: { cap: number; periods: Record<Period, Pair> };
  candidates: { id: string; label: string; periods: Record<Period, Metric> }[];
  filters: { threshold: number; coverage: number; beforeSameGames: Metric; after: Metric }[];
  ablations: { factor: string; active: boolean; deltaPoints: number }[];
  engines: { id: string; label: string; metrics: Metric; status: string }[];
};
const names: Record<string, string> = { localia: 'Localía', strength: 'Fuerza de temporada', recent: 'Forma reciente', runDiff: 'Diferencial de carreras', series: 'Series', marketSchedule: 'Mercado + descanso', bestPlayersTest: 'Mejores jugadores', coachRotationTest: 'Rotación del coach', rotationQualityTest: 'Calidad del lineup', lineupFatigueTest: 'Fatiga del lineup', opponentFormTest: 'Forma vs. rival' };
const percent = (n: number | null) => n == null ? '—' : `${(n * 100).toFixed(2)} %`;
const decimal = (n: number | null) => n == null ? '—' : n.toFixed(4);

export default function PredictionAudit({ active }: { active: Record<string, boolean> }) {
  const [report, setReport] = useState<Report | null>(null);
  const [error, setError] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const [period, setPeriod] = useState<Period>('validation2026');
  useEffect(() => {
    const controller = new AbortController();
    fetch('/data/prediction-audit.json', { signal: controller.signal }).then(r => {
      if (!r.ok) throw new Error('Audit unavailable');
      return r.json() as Promise<Report>;
    }).then(data => { setReport(data); setError(false); }).catch(e => { if (e.name !== 'AbortError') setError(true); });
    return () => controller.abort();
  }, [attempt]);
  if (!report) return <section className="section audit-section"><h2>Auditoría de motores</h2>{error ? <p>No se pudo cargar el informe. <button onClick={() => { setError(false); setAttempt(v => v + 1); }}>Reintentar</button></p> : <p role="status">Cargando medición…</p>}</section>;
  const pair = report.beforeAfter[period];
  const corrected = report.probabilityCorrection.periods[period];
  return <section className="section audit-section prediction-audit">
    <div className="section-heading"><span className="eyebrow">AUDITORÍA REPRODUCIBLE · {report.dataThrough}</span><h2>Motores, filtros y antes / después</h2><p>{report.coverage.games.toLocaleString('es-MX')} juegos revisados · {report.coverage.maskCount} combinaciones compatibles. El informe es una fotografía al {report.asOf}, no una medición en vivo.</p></div>
    <p className="audit-conclusion">Candidato en observación, NO activado. {report.selected.reason}</p>
    <label className="audit-period">Periodo de comparación <select value={period} onChange={e => setPeriod(e.target.value as Period)}><option value="validation2026">2026 hasta el corte</option><option value="h1">2026 · enero–junio</option><option value="h2">2026 · julio–corte</option></select></label>
    <div className="coach-proof-grid"><article><span>ANTES · BASE WF 45</span><strong>{percent(pair.before.accuracy)}</strong><small>{pair.before.correct} / {pair.before.games} aciertos</small></article><article><span>DESPUÉS · CANDIDATO HISTÓRICO</span><strong>{percent(pair.after.accuracy)}</strong><small>{pair.after.correct} / {pair.after.games} · mismos juegos</small></article><article><span>DIFERENCIA PAREADA</span><strong>{pair.deltaPoints.toFixed(2)} pts</strong><small>{pair.netCorrect} aciertos netos adicionales</small></article><article><span>INTERVALO 95 % · PUNTOS</span><strong>{pair.deltaCI95Points.map(v => v.toFixed(2)).join(' a ')}</strong><small>La incertidumbre no desaparece con más filtros</small></article></div>
    <p><strong>Candidato:</strong> {report.selected.label}. {report.method.selection}</p>
    <div className="audit-table-scroll"><table><caption>Calidad de las probabilidades: Brier menor es mejor</caption><thead><tr><th>Variante</th><th>Acierto</th><th>Brier</th></tr></thead><tbody><tr><td>Base walk-forward</td><td>{percent(pair.before.accuracy)}</td><td>{decimal(pair.before.brier)}</td></tr><tr><td>Candidato sin ajuste</td><td>{percent(pair.after.accuracy)}</td><td>{decimal(pair.after.brier)}</td></tr><tr><td>Candidato con techo de confianza {percent(report.probabilityCorrection.cap)}</td><td>{percent(corrected.after.accuracy)}</td><td>{decimal(corrected.after.brier)}</td></tr></tbody></table></div>
    <p>El techo se eligió únicamente en 2024–2025 entre siete opciones. Conserva los picks; no implica calibración perfecta ni está activado. Se muestran también los resultados sin ajuste para no ocultar la sobreconfianza.</p>
    <h3>Factores configurados ahora</h3><p>Los datos faltantes pueden desactivar factores por partido. Cambiar estos controles no recalcula este informe: la referencia medida sigue siendo la máscara 45.</p>
    <div className="audit-factor-list">{Object.entries(names).map(([key, label]) => <span key={key} className={active[key] ? 'is-active' : ''}>{active[key] ? '● Activo' : '○ Inactivo'} · {label}</span>)}</div>
    <h3>Combinaciones medidas en los mismos juegos</h3><div className="audit-table-scroll"><table><thead><tr><th>Motor / combinación</th><th>Juegos</th><th>Acierto</th><th>Brier</th></tr></thead><tbody>{report.candidates.map(row => <tr key={row.id}><td>{row.label}</td><td>{row.periods[period].games}</td><td>{percent(row.periods[period].accuracy)}</td><td>{decimal(row.periods[period].brier)}</td></tr>)}</tbody></table></div>
    <h3>Filtrar no es mejorar a cobertura igual</h3><p>2026 completo hasta el corte. Umbrales aplicados al candidato sin ajuste; la base se vuelve a medir solo en esos mismos partidos.</p><div className="audit-table-scroll"><table><thead><tr><th>Confianza mínima</th><th>Juegos</th><th>Cobertura</th><th>Base, mismos juegos</th><th>Candidato</th></tr></thead><tbody>{report.filters.map(row => <tr key={row.threshold}><td>{percent(row.threshold)}</td><td>{row.after.games}</td><td>{percent(row.coverage)}</td><td>{percent(row.beforeSameGames.accuracy)}</td><td>{percent(row.after.accuracy)}</td></tr>)}</tbody></table></div>
    <h3>Otros motores y límites de la evidencia</h3><p>Sus muestras son diferentes: esta tabla no es un ranking comparable.</p><div className="audit-table-scroll"><table><thead><tr><th>Motor</th><th>Juegos</th><th>Acierto observado</th><th>Validez</th></tr></thead><tbody>{report.engines.map(row => <tr key={row.id}><td>{row.label}</td><td>{row.metrics.games}</td><td>{percent(row.metrics.accuracy)}</td><td>{row.status}</td></tr>)}</tbody></table></div>
    <h3>Hallazgos que impiden certificar el rendimiento en vivo</h3><ul>
      <li>Los rankings globales del Comparador eligen ganadores usando el histórico completo. Sus resultados son retrospectivos, no una selección fuera de muestra.</li>
      <li>Value entrena con todos los resultados disponibles: su acierto histórico no demuestra capacidad predictiva futura.</li>
      <li>El calibrador estático usa resultados posteriores a partidos históricos, una máscara fija y pRaw; la web utiliza otras máscaras y pBlend.</li>
      <li>El archivo actual de jugadores declara corte {report.coverage.currentPlayerCutoff}; debe validarse contra la fecha de cada juego. La alineación reconstruida tampoco tiene hora de confirmación.</li>
      <li>Consenso, F5, rotación, series, umbrales de mercado y zona de valor generan categorías por reglas. LOCK y FUERTE no son probabilidades calibradas; tarjetas y tabla aplican reglas distintas.</li>
      <li>Movimiento de línea es contexto; los modelos de Momios predicen precios, no el ganador. Su accuracy no equivale al acierto de picks.</li>
    </ul><p>{report.method.limitations}</p><p>{report.method.interval}</p>
    <a href="/data/prediction-audit.json" download>Descargar informe JSON con cortes, máscaras mensuales y hashes de fuentes</a>
  </section>;
}

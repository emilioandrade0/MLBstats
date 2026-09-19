'use client';

import { useEffect, useState } from 'react';

type Result = { games: number; beforeCorrect: number; afterCorrect: number; beforeAccuracy: number | null; afterAccuracy: number | null; rescued: number; damaged: number; netCorrect: number; changed: number; deltaPoints: number | null; deltaCI95?: number[] };
type Report = {
  sourceGeneratedAt: string; dataThrough: string;
  selected: { label: string; reason: string; numericalGatePassed: boolean };
  evaluation: Result; homeOnlyControl: Result;
  monthly: (Result & { month: string })[];
  method: { training: string; limitations: string; interval: string };
  candidates: { id: string; label: string; design: Result }[];
  changes: { gamePk: number; date: string; away: string; home: string; before: string; after: string; winner: string; effect: string }[];
};
const pct = (n: number | null) => n == null ? '—' : `${(100*n).toFixed(1)}%`;

export default function LowConfidenceSpecialist({ enabled, sourceGeneratedAt, refreshToken }: { enabled: boolean; sourceGeneratedAt?: string; refreshToken: number }) {
  const [report, setReport] = useState<Report | null>(null);
  const [error, setError] = useState(false);
  useEffect(() => {
    if (!enabled) return;
    const controller = new AbortController();
    fetch('/data/low-confidence-specialist.json', { cache: 'no-store', signal: controller.signal })
      .then(r => { if (!r.ok) throw new Error('Report unavailable'); return r.json() as Promise<Report>; })
      .then(data => { if (!controller.signal.aborted) { setReport(data); setError(false); } }).catch(() => { if (!controller.signal.aborted) setError(true); });
    return () => controller.abort();
  }, [enabled, refreshToken]);
  if (error) return <div className="failure-note" role="status">No se pudo cargar la prueba del especialista. Pulsa «Actualizar histórico» para reintentar.</div>;
  if (!report) return enabled ? <p role="status">Cargando la prueba del especialista…</p> : null;
  const r = report.evaluation;
  const stale = sourceGeneratedAt && sourceGeneratedAt !== report.sourceGeneratedAt;
  const gain = r.netCorrect > 0;
  return <article className="specialist-experiment">
    <span className="eyebrow">EXPERIMENTO · PICKS CON CONFIANZA MENOR A 55%</span>
    <h3>¿Podemos corregirlos sin descartar partidos?</h3>
    <p><strong>{report.selected.numericalGatePassed ? 'Hay una mejora histórica que merece seguimiento.' : gain ? 'Mejoró el total, pero todavía no demuestra una mejora estable.' : 'Esta prueba no logró mejorar el nicho de forma estable.'}</strong> Se conservaron los {r.games.toLocaleString('es-MX')} partidos evaluados desde enero de 2026 hasta {report.dataThrough}.</p>
    {stale && <p className="failure-warning">Esta prueba tiene un corte distinto al histórico superior. Sus cifras corresponden únicamente al {report.dataThrough}; la actualización automática regenerará el informe.</p>}
    <div className="coach-proof-grid"><article><span>MOTOR ACTUAL · MISMO NICHO</span><strong>{pct(r.beforeAccuracy)}</strong><small>{r.beforeCorrect} / {r.games} aciertos</small></article><article><span>CON EL ESPECIALISTA</span><strong>{pct(r.afterAccuracy)}</strong><small>{r.afterCorrect} / {r.games} · conserva el 100% de los partidos</small></article><article><span>CORREGIDOS / ESTROPEADOS</span><strong>{r.rescued} / {r.damaged}</strong><small>Cambió {r.changed} picks; los demás se conservaron</small></article><article><span>CAMBIO NETO</span><strong>{r.netCorrect > 0 ? '+' : ''}{r.netCorrect}</strong><small>{Math.abs(r.netCorrect)} aciertos {r.netCorrect >= 0 ? 'más' : 'menos'} · {r.deltaPoints?.toFixed(2) ?? '—'} puntos de accuracy</small></article></div>
    <p>De sus cambios, <strong>{r.rescued}</strong> convirtieron una derrota en acierto y <strong>{r.damaged}</strong> hicieron lo contrario. {r.deltaCI95 && <>El intervalo estimado para la mejora va de <strong>{r.deltaCI95[0].toFixed(2)} a {r.deltaCI95[1].toFixed(2)} puntos</strong>; {r.deltaCI95[0] <= 0 ? 'incluye la posibilidad de no mejorar.' : 'es positivo en esta medición retrospectiva.'}</>}</p>
    <p className="failure-note"><strong>Estado: prueba, sin activar.</strong> {report.selected.reason} El especialista elegido fue «{report.selected.label}». Se eligió por su resultado en 2025 y se reentrenó cada mes con datos anteriores. No se eligió mirando qué ganaba en 2026.</p>
    <details><summary>Ver si la mejora se repite mes a mes</summary><div className="audit-table-scroll"><table><thead><tr><th>Mes</th><th>Partidos del nicho</th><th>Antes</th><th>Especialista</th><th>Corregidos</th><th>Estropeados</th><th>Aciertos netos</th></tr></thead><tbody>{report.monthly.map(m => <tr key={m.month}><td>{m.month}</td><td>{m.games}</td><td>{pct(m.beforeAccuracy)}</td><td>{pct(m.afterAccuracy)}</td><td>{m.rescued}</td><td>{m.damaged}</td><td>{m.netCorrect > 0 ? '+' : ''}{m.netCorrect}</td></tr>)}</tbody></table></div></details>
    <details><summary>Ver partidos donde cambió el pick</summary><p>Últimos 25 cambios, acertados y fallidos. Son predicciones reconstruidas para la prueba, no recomendaciones nuevas.</p><div className="audit-table-scroll"><table><thead><tr><th>Fecha</th><th>Partido</th><th>Antes</th><th>Especialista</th><th>Ganador</th><th>Efecto</th></tr></thead><tbody>{report.changes.slice(-25).reverse().map(g => <tr key={g.gamePk}><td>{g.date}</td><td>{g.away} @ {g.home}</td><td>{g.before}</td><td>{g.after}</td><td>{g.winner}</td><td className={g.effect === 'rescatado' ? 'failure-hit' : 'failure-miss'}>{g.effect === 'rescatado' ? 'Corrigió un fallo' : 'Estropeó un acierto'}</td></tr>)}</tbody></table></div>{!report.changes.length && <p>No se cambió ningún pick.</p>}</details>
    <details><summary>Cómo se eligió y qué falta comprobar</summary><p>{report.method.training}</p><div className="audit-table-scroll"><table><caption>Selección únicamente con 2025; 2024 se usa como historial inicial.</caption><thead><tr><th>Candidato</th><th>Partidos</th><th>Acierto en 2025</th><th>Aciertos netos vs. base</th></tr></thead><tbody>{report.candidates.map(c => <tr key={c.id}><td>{c.label}</td><td>{c.design.games}</td><td>{pct(c.design.afterAccuracy)}</td><td>{c.design.netCorrect}</td></tr>)}</tbody></table></div><p>Control simple en los mismos partidos de evaluación: elegir siempre al local habría acertado {pct(report.homeOnlyControl.afterAccuracy)}. Sirve para comprobar si el especialista aporta algo más que una preferencia por la localía.</p><p>{report.method.limitations}</p><p>{report.method.interval}</p><a href="/data/low-confidence-specialist.json" download>Descargar prueba completa, cambios y cortes de entrenamiento</a></details>
  </article>;
}

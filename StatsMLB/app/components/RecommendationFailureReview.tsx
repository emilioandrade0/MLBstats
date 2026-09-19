'use client';

type Row = {
  id: number; matchup: string; team: string | null; tier: string; winner: string | null;
  market: string | null; base: string | null; veryConfident: boolean;
};
const signals = [
  { label: 'Mercado elige al rival', test: (r: Row) => r.market == null ? null : r.market !== r.team },
  { label: 'Modelo base elige al rival', test: (r: Row) => r.base == null ? null : r.base !== r.team },
  { label: 'Algún motor a favor declara ≥90%', test: (r: Row) => r.veryConfident },
];
export default function RecommendationFailureReview({ rows }: { rows: Row[] }) {
  const finished = rows.filter(r => r.team && r.winner);
  const failed = finished.filter(r => r.team !== r.winner);
  const rate = (group: Row[]) => group.length ? `${group.filter(r => r.team !== r.winner).length}/${group.length} (${(100*group.filter(r => r.team !== r.winner).length/group.length).toFixed(0)}%)` : 'Sin muestra';
  return <details className="failure-method failure-comparator">
    <summary>Revisar fallos de esta jornada · {failed.length} de {finished.length} recomendaciones resueltas</summary>
    <p>Se analizan las recomendaciones que muestra esta tabla ahora. Para fechas pasadas pueden ser reconstrucciones; no certifican el pick publicado antes del partido. Se excluyen PASAR y juegos pendientes.</p>
    {!finished.length ? <p>Todavía no hay recomendaciones resueltas para comparar.</p> : <>
      <div className="audit-table-scroll"><table><caption>Tasa de fallo de recomendaciones con y sin cada señal</caption><thead><tr><th>Señal</th><th>Con señal · fallos / picks</th><th>Sin señal · fallos / picks</th><th>Sin dato</th></tr></thead><tbody>{signals.map(s => <tr key={s.label}><td>{s.label}</td><td>{rate(finished.filter(r => s.test(r) === true))}</td><td>{rate(finished.filter(r => s.test(r) === false))}</td><td>{finished.filter(r => s.test(r) == null).length}</td></tr>)}</tbody></table></div>
      <div className="audit-table-scroll"><table><caption>Detalle de recomendaciones fallidas</caption><thead><tr><th>Partido</th><th>Recomendación</th><th>Ganador</th><th>Señales presentes</th></tr></thead><tbody>{failed.map(r => <tr key={r.id}><td>{r.matchup}</td><td>{r.tier} · {r.team}</td><td>{r.winner}</td><td>{signals.filter(s => s.test(r) === true).map(s => s.label).join(' · ') || 'Ninguna de las evaluadas'}</td></tr>)}</tbody></table></div>
      {!failed.length && <p>No hubo fallos entre las recomendaciones resueltas de esta jornada.</p>}
    </>}
    <p>Un solo día no permite validar un filtro. <a href="#riesgo">Explorar patrones en el histórico del motor fijo →</a></p>
  </details>;
}

'use client';

import { useEffect, useMemo, useState } from 'react';
import { SIGNALS, compareSignal, evaluateFilter, metrics, normalizeGame, signalValue, type FailureGame, type FrozenGame, type SignalId } from '@/lib/failure-analysis';

type Manifest = { generatedAt?: string; probabilityEncoding?: string; factorKeys: string[]; dayFiles: string[] };
type Dataset = { games: FailureGame[]; generatedAt: string; retrievedAt: string; excluded: number; source: string };
const pct = (v: number | null) => v == null ? '—' : `${(100*v).toFixed(1)}%`;
const failureRate = (m: ReturnType<typeof metrics>) => m.accuracy == null ? null : 1-m.accuracy;
const interval = (m: ReturnType<typeof metrics>) => m.interval ? m.interval.map(v => pct(v)).join(' – ') : 'Sin muestra';
const dateToday = () => new Intl.DateTimeFormat('en-CA', { timeZone: 'America/Mexico_City', year: 'numeric', month: '2-digit', day: '2-digit' }).format(new Date());

async function readJson<T>(path: string, signal: AbortSignal): Promise<T> {
  const response = await fetch(path, { cache: 'no-store', signal });
  if (!response.ok) throw new Error(`No se pudo leer ${path} (${response.status}).`);
  return response.json() as Promise<T>;
}

export default function FailureComparator() {
  const [enabled, setEnabled] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [season, setSeason] = useState('2026');
  const [signal, setSignal] = useState<SignalId>('lowConfidence');
  const [outcome, setOutcome] = useState('all');
  const [page, setPage] = useState(0);
  useEffect(() => {
    const activate = () => { if (window.location.hash === '#riesgo') setEnabled(true); };
    activate(); window.addEventListener('hashchange', activate); window.addEventListener('popstate', activate);
    // Dashboard navigation uses pushState and mounts all panels; observe tab selection.
    const observer = new MutationObserver(() => activate());
    const panel = document.getElementById('panel-riesgo');
    if (panel) observer.observe(panel, { attributes: true, attributeFilter: ['hidden'] });
    return () => { observer.disconnect(); window.removeEventListener('hashchange', activate); window.removeEventListener('popstate', activate); };
  }, []);
  useEffect(() => {
    if (!enabled) return;
    const controller = new AbortController();
    async function load() {
      setLoading(true); setError('');
      try {
        const manifest = await readJson<Manifest>('/data/walkforward.json', controller.signal);
        if (!Array.isArray(manifest.dayFiles) || !manifest.dayFiles.length || manifest.dayFiles.length > 20 ||
          manifest.factorKeys?.[0] !== 'localia' || manifest.factorKeys?.[2] !== 'recent' ||
          manifest.factorKeys?.[3] !== 'runDiff' || manifest.factorKeys?.[5] !== 'marketSchedule') throw new Error('El formato del histórico cambió. No se calcularán señales con factores distintos.');
        const games = new Map<number, FailureGame>();
        const duplicates = new Set<number>();
        const today = dateToday();
        let excluded = 0;
        for (const file of [...new Set(manifest.dayFiles)]) {
          if (!/^walkforward-\d{4}\.json$/.test(file)) throw new Error('Archivo histórico no reconocido.');
          const shard = await readJson<{ days: Record<string, { games: FrozenGame[] }> }>(`/data/${file}`, controller.signal);
          if (!shard.days || typeof shard.days !== 'object') throw new Error(`Histórico incompleto: ${file}.`);
          for (const day of Object.values(shard.days)) {
            if (!Array.isArray(day.games)) throw new Error(`Jornada inválida en ${file}.`);
            for (const raw of day.games) {
              const game = raw && normalizeGame(raw, manifest.probabilityEncoding);
              if (!game || game.date > today) { excluded++; continue; }
              if (games.has(game.gamePk) || duplicates.has(game.gamePk)) { games.delete(game.gamePk); duplicates.add(game.gamePk); excluded++; continue; }
              games.set(game.gamePk, game);
            }
          }
        }
        const after = await readJson<Manifest>('/data/walkforward.json', controller.signal);
        if (after.generatedAt !== manifest.generatedAt) throw new Error('El histórico se actualizó durante la lectura. Vuelve a cargarlo para usar un único corte.');
        if (!games.size) throw new Error('No hay partidos válidos con predicción anterior al juego.');
        setDataset({ games: [...games.values()].sort((a,b) => b.date.localeCompare(a.date) || b.gamePk-a.gamePk),
          generatedAt: manifest.generatedAt ?? '', retrievedAt: new Date().toISOString(), excluded: excluded + duplicates.size, source: window.location.origin });
      } catch (e) { if (!controller.signal.aborted) setError(e instanceof Error ? e.message : 'No se pudo cargar el histórico.'); }
      finally { if (!controller.signal.aborted) setLoading(false); }
    }
    void load(); return () => controller.abort();
  }, [enabled, attempt]);
  const games = useMemo(() => dataset?.games.filter(g => season === 'all' || g.date.startsWith(season)) ?? [], [dataset, season]);
  const overall = useMemo(() => metrics(games), [games]);
  const signals = useMemo(() => SIGNALS.map(s => ({ ...s, ...compareSignal(games, s.id) })), [games]);
  // This split is fixed, never optimized on results. 2026 is already explored: retrospective only.
  const design = useMemo(() => (dataset?.games ?? []).filter(g => g.date < '2026-01-01'), [dataset]);
  const evaluation = useMemo(() => (dataset?.games ?? []).filter(g => g.date >= '2026-01-01'), [dataset]);
  const trial = useMemo(() => evaluateFilter(evaluation, signal), [evaluation, signal]);
  const designTrial = useMemo(() => evaluateFilter(design, signal), [design, signal]);
  const monthly = useMemo(() => [...new Set(evaluation.map(g => g.date.slice(0,7)))].sort().map(month => ({ month, ...evaluateFilter(evaluation.filter(g => g.date.startsWith(month)), signal) })), [evaluation, signal]);
  const visible = games.filter(g => outcome === 'all' || (outcome === 'failures' ? !g.correct : g.correct));
  const pages = Math.max(1, Math.ceil(visible.length / 25));
  const currentPage = Math.min(page, pages-1);
  const selectedSignal = SIGNALS.find(s => s.id === signal)!;
  const latest = dataset?.games[0]?.date;
  const old = latest ? Math.round((Date.parse(`${dateToday()}T12:00:00Z`) - Date.parse(`${latest}T12:00:00Z`))/86400000) > 2 : false;
  function download() {
    const payload = { method: 'Exploración retrospectiva; referencia fija WF 45. No son recomendaciones en vivo archivadas.', source: dataset?.source,
      generatedAt: dataset?.generatedAt, retrievedAt: dataset?.retrievedAt, dataThrough: latest, excluded: dataset?.excluded,
      period: season, overall, signals, filter: { signal, designThrough: '2025-12-31', evaluationFrom: '2026-01-01', design: designTrial, evaluation: trial, monthly }, games };
    const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }));
    const link = document.createElement('a'); link.href = url; link.download = `comparador-fallos-${latest}.json`; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  return <section className="section failure-comparator">
    <div className="section-heading"><span className="eyebrow">DIAGNÓSTICO · REFERENCIA FIJA WF 45</span><h2>¿Qué comparten los picks que fallan?</h2><p>Compara fallos y aciertos, revisa la confianza y mide cuántos picks perderías al aplicar un filtro.</p></div>
    <div className="failure-toolbar"><label>Periodo de diagnóstico<select value={season} onChange={e => { setSeason(e.target.value); setPage(0); }}><option value="all">Todo el histórico</option>{[...new Set(['2026', ...(dataset?.games.map(g => g.date.slice(0,4)) ?? [])])].sort().reverse().map(y => <option key={y}>{y}</option>)}</select></label><button disabled={loading} onClick={() => { setEnabled(true); setAttempt(n => n+1); }}>{loading ? 'Leyendo histórico…' : 'Actualizar histórico'}</button><button disabled={!dataset || loading || !!error} onClick={download}>Descargar análisis</button></div>
    {loading && <p role="status">Consultando los archivos publicados en esta web…</p>}
    {error && <p role="alert" className="failure-warning">{error} {dataset ? 'Los resultados anteriores permanecen visibles; no se actualizaron.' : 'No se sustituyeron por datos de otra fuente.'}</p>}
    {!dataset ? <p>El análisis estará disponible al terminar la lectura del histórico.</p> : <>
      <p className={old ? 'failure-warning' : 'failure-source'}>Último juego analizado: <strong>{latest}</strong> · Fuente: {dataset.source} · {old ? 'El histórico de predicciones tiene más de dos días de rezago, aunque la jornada en vivo pueda estar actualizada.' : 'Solo partidos finalizados con corte de entrenamiento anterior al juego.'}</p>
      <div className="coach-proof-grid"><article><span>ACIERTOS</span><strong>{pct(overall.accuracy)}</strong><small>{overall.correct} / {overall.n} picks · IC 95% {interval(overall)}</small></article><article><span>FALLOS</span><strong>{overall.failures}</strong><small>{pct(failureRate(overall))} del periodo</small></article><article><span>CONFIANZA PROMEDIO</span><strong>{pct(overall.averageConfidence)}</strong><small>Contrástala con el acierto observado</small></article><article><span>COBERTURA DEL DIAGNÓSTICO</span><strong>{overall.n.toLocaleString('es-MX')}</strong><small>Partidos válidos · {dataset.excluded} registros excluidos en la fuente completa</small></article></div>
      <p className="failure-note">Esta primera etapa analiza el motor histórico con factores fijos: localía, forma reciente, diferencial y mercado + descanso. No reconstruye los LOCK/FUERTE emitidos en vivo ni utiliza la combinación ganadora del histórico. Los porcentajes están cuantizados a intervalos de 1/256.</p>
      <h3>Señales presentes frente a señales ausentes</h3><p>Una señal resulta interesante si aumenta la tasa de fallo y se repite en otros periodos. La asociación no demuestra la causa.</p>
      <div className="audit-table-scroll"><table><caption>El porcentaje es tasa de fallo: menor es mejor. Se muestran ambos grupos.</caption><thead><tr><th>Señal</th><th>Con señal · fallos / picks</th><th>Fallo con señal</th><th>Sin señal · fallos / picks</th><th>Fallo sin señal</th><th>Diferencia</th><th>Sin dato</th></tr></thead><tbody>{signals.map(s => {
        const a = failureRate(s.withSignal), b = failureRate(s.withoutSignal);
        return <tr key={s.id}><td><strong>{s.label}</strong><small>{s.detail}{Math.min(s.withSignal.n,s.withoutSignal.n) < 100 ? ' · Muestra pequeña en uno de los grupos.' : ''}</small></td><td>{s.withSignal.failures} / {s.withSignal.n}</td><td>{pct(a)}</td><td>{s.withoutSignal.failures} / {s.withoutSignal.n}</td><td>{pct(b)}</td><td>{a == null || b == null ? '—' : `${((a-b)*100).toFixed(1)} pts`}</td><td>{s.unknown}</td></tr>;
      })}</tbody></table></div>
      <h3>Simular «pasar» cuando aparece una señal</h3>
      <label className="failure-select">Señal a filtrar<select value={signal} onChange={e => setSignal(e.target.value as SignalId)}>{SIGNALS.map(s => <option key={s.id} value={s.id}>{s.label}</option>)}</select></label>
      <p>Diseño hasta diciembre de 2025; evaluación desde enero de 2026. Estos cortes son independientes del periodo superior. Es una exploración retrospectiva sobre datos ya consultados; cambiar la regla aquí no constituye una nueva prueba independiente.</p>
      <div className="coach-proof-grid"><article><span>ANTES · TODOS LOS PICKS 2026+</span><strong>{pct(trial.before.accuracy)}</strong><small>{trial.before.correct} / {trial.before.n}</small></article><article><span>DESPUÉS · PICKS CONSERVADOS</span><strong>{pct(trial.after.accuracy)}</strong><small>{trial.after.correct} / {trial.after.n} · IC 95% {interval(trial.after)}</small></article><article><span>COBERTURA CONSERVADA</span><strong>{pct(trial.coverage)}</strong><small>{trial.skipped.n} picks descartados</small></article><article><span>EL COSTO DEL FILTRO</span><strong>{trial.skipped.failures} / {trial.skipped.correct}</strong><small>Fallos evitados / aciertos descartados</small></article></div>
      <p className="failure-note">Antes de 2026: {pct(designTrial.before.accuracy)} → {pct(designTrial.after.accuracy)} de acierto, conservando {pct(designTrial.coverage)} de {designTrial.before.n} picks. El filtro no cambia ningún ganador: sube o baja la tasa por selección de partidos. Los casos sin señal disponible se conservan. No está activado en tus recomendaciones.</p>
      <div className="audit-table-scroll"><table><caption>Estabilidad mensual del filtro seleccionado · {selectedSignal.label}</caption><thead><tr><th>Mes</th><th>Picks</th><th>Acierto antes</th><th>Acierto conservado</th><th>Cobertura</th><th>Fallos evitados</th><th>Aciertos descartados</th></tr></thead><tbody>{monthly.map(m => <tr key={m.month}><td>{m.month}</td><td>{m.before.n}</td><td>{pct(m.before.accuracy)}</td><td>{pct(m.after.accuracy)}</td><td>{pct(m.coverage)}</td><td>{m.skipped.failures}</td><td>{m.skipped.correct}</td></tr>)}</tbody></table></div>
      <h3>Revisar los partidos</h3><div className="failure-toolbar"><label>Resultado<select value={outcome} onChange={e => { setOutcome(e.target.value); setPage(0); }}><option value="all">Aciertos y fallos</option><option value="failures">Solo fallos</option><option value="hits">Solo aciertos</option></select></label><span>{visible.length} partidos · más recientes primero</span></div>
      <div className="audit-table-scroll"><table><thead><tr><th>Fecha</th><th>Partido</th><th>Pick WF 45</th><th>Confianza</th><th>Ganador</th><th>Resultado</th><th>Señales presentes</th><th>Entrenado hasta</th></tr></thead><tbody>{visible.slice(currentPage*25, (currentPage+1)*25).map(g => <tr key={g.gamePk}><td>{g.date}</td><td>{g.away} @ {g.home}</td><td>{g.pick}</td><td>{pct(g.confidence)}</td><td>{g.winner}</td><td className={g.correct ? 'failure-hit' : 'failure-miss'}>{g.correct ? 'Acierto' : 'Fallo'}</td><td>{SIGNALS.filter(s => signalValue(g,s.id) === true).map(s => s.label).join(' · ') || 'Ninguna de las evaluadas'}</td><td>{g.trainedThrough}</td></tr>)}</tbody></table></div>
      {!visible.length && <p>No hay partidos para este periodo y resultado.</p>}
      <nav className="failure-toolbar" aria-label="Páginas de partidos"><button disabled={currentPage === 0} onClick={() => setPage(currentPage-1)}>Anterior</button><span>Página {currentPage+1} de {pages}</span><button disabled={currentPage+1 >= pages} onClick={() => setPage(currentPage+1)}>Siguiente</button></nav>
      <details className="failure-method"><summary>Procedencia y límites de esta medición</summary><p>Histórico generado: {dataset.generatedAt || 'sin fecha declarada'}. Lectura: {dataset.retrievedAt}. Se excluyen empates, resultados inconsistentes, probabilidades ausentes, fechas futuras, cortes de entrenamiento inválidos y todas las copias de juegos duplicados.</p><p>Los intervalos Wilson del 95% describen cada tasa y suponen observaciones independientes; no prueban una mejora del filtro y no corrigen la exploración de múltiples reglas ni la dependencia entre partidos. Las señales pueden solaparse: no sumes sus fallos.</p><p>El histórico walk-forward es retrospectivo y no certifica que todas las entradas se capturaran antes del partido. Para entrenar un detector de fallos de LOCK/FUERTE hace falta un registro prospectivo inmutable de esas recomendaciones. Esta vista no inventa probabilidades de fallo ni activa un nuevo modelo.</p></details>
    </>}
  </section>;
}

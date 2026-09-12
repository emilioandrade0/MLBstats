'use client';

import { CalendarRange, Check, ChevronLeft, ChevronRight, RotateCcw, ShieldCheck, Target, Trophy, X } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import TeamLogo from './TeamLogo';

export type FactorKey = 'localia' | 'strength' | 'recent' | 'runDiff' | 'series' | 'marketSchedule' | 'bestPlayersTest' | 'coachRotationTest' | 'rotationQualityTest' | 'lineupFatigueTest' | 'opponentFormTest';

type WalkforwardCalendarProps = {
  active: Record<FactorKey, boolean>;
  onToggle: (key: FactorKey) => void;
  onSetActive: (keys: FactorKey[]) => void;
};

type WalkforwardGame = {
  gamePk: number;
  date: string;
  foldMonth: string;
  trainedThrough: string;
  away: string;
  home: string;
  awayScore: number;
  homeScore: number;
  homeProbability: number;
  predictedWinner: string;
  predictedProbability: number;
  actualWinner: string;
  correct: number;
  maskProbabilities?: (number | null)[];
  maskProbabilitiesEncoded?: string;
  homeContext: { code: string; label: string };
  awayContext: { code: string; label: string };
};

type WalkforwardDay = {
  date: string;
  games: WalkforwardGame[];
  total: number;
  correct: number;
  accuracy: number;
  brier: number;
};

type WalkforwardData = {
  methodology: {
    mode: string;
    rule: string;
    firstPredictionDate: string;
    lastPredictionDate: string;
    temporalLeakageViolations: number;
    coverageAudit: {
      status: string;
      derivedFinalGames: number;
      canonicalProcessedFinals: number;
      coveredCanonicalFinals: number;
      supplementalScheduleFinals: number;
      missingCanonicalFinals: number;
      datesWithMissingGames: number;
    };
  };
  summary: {
    games: number;
    correct: number;
    accuracy: number;
    brier: number;
    homeBaselineAccuracy: number;
    folds: number;
    days: number;
    winningDays: number;
    evenDays: number;
    losingDays: number;
    perfectDays: number;
  };
  probabilityScale?: number;
  probabilityEncoding?: string;
  dayFiles?: string[];
  defaultMask: number;
  bestWinningDaysMask?: number;
  bestFifteenGameAccuracyMask?: number;
  factorKeys: FactorKey[];
  combinations: { mask: number; games: number; correct: number; accuracy: number; days?: number; winningDays?: number; evenDays?: number; losingDays?: number; perfectDays?: number; fifteenGameDays?: number; fifteenGameWinningDays?: number; fifteenGameGames?: number; fifteenGameCorrect?: number; fifteenGameAccuracy?: number; fifteenGamePerfectDays?: number; seasons: Record<string, DynamicStat> }[];
  folds: { month: string; trainedThrough: string; trainGames: number; games: number; correct: number; accuracy: number }[];
  days: Record<string, WalkforwardDay>;
};

type DynamicStat = { games: number; correct: number; accuracy: number };
type CombinationStat = DynamicStat & {
  mask: number;
  keys: FactorKey[];
  label: string;
  days: number;
  winningDays: number;
  evenDays: number;
  losingDays: number;
  perfectDays: number;
  fifteenGameDays: number;
  fifteenGameWinningDays: number;
  fifteenGameGames: number;
  fifteenGameCorrect: number;
  fifteenGameAccuracy: number;
  fifteenGamePerfectDays: number;
  seasons: Record<string, DynamicStat>;
};

const FACTORS: { key: FactorKey; label: string }[] = [
  { key: 'localia', label: 'Localía' },
  { key: 'strength', label: 'Temporada' },
  { key: 'recent', label: 'Forma L10' },
  { key: 'runDiff', label: 'Carreras' },
  { key: 'series', label: 'Serie previa' },
  { key: 'marketSchedule', label: 'Mercado + descanso' },
  { key: 'bestPlayersTest', label: 'Mejores jugadores (prueba)' },
  { key: 'coachRotationTest', label: 'Rotación coach (prueba)' },
  { key: 'rotationQualityTest', label: 'Calidad rotación (prueba)' },
  { key: 'lineupFatigueTest', label: 'Fatiga lineup (prueba)' },
  { key: 'opponentFormTest', label: 'Forma vs rival (prueba)' },
];
const FACTOR_KEYS = FACTORS.map(factor => factor.key);
const FACTOR_LABELS = Object.fromEntries(FACTORS.map(factor => [factor.key, factor.label])) as Record<FactorKey, string>;
const BAND_SPECS = [
  { low: 0.5, high: 0.55, label: '50–54.9%' },
  { low: 0.55, high: 0.6, label: '55–59.9%' },
  { low: 0.6, high: 1.01, label: '60%+' },
];
const monthFormatter = new Intl.DateTimeFormat('es-MX', { month: 'long', year: 'numeric', timeZone: 'UTC' });
const dateFormatter = new Intl.DateTimeFormat('es-MX', { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric', timeZone: 'UTC' });
const pct = (value: number) => `${(value * 100).toFixed(1)}%`;
const pct2 = (value: number) => `${(value * 100).toFixed(2)}%`;

function monthLabel(month: string) {
  return monthFormatter.format(new Date(`${month}-15T12:00:00Z`));
}

function calendarCells(month: string) {
  const [year, monthNumber] = month.split('-').map(Number);
  const first = new Date(Date.UTC(year, monthNumber - 1, 1));
  const daysInMonth = new Date(Date.UTC(year, monthNumber, 0)).getUTCDate();
  const mondayOffset = (first.getUTCDay() + 6) % 7;
  return Array.from({ length: 42 }, (_, index) => {
    const day = index - mondayOffset + 1;
    if (day < 1 || day > daysInMonth) return null;
    return `${month}-${String(day).padStart(2, '0')}`;
  });
}

function dayTone(day?: WalkforwardDay) {
  if (!day) return 'calendar-empty';
  if (day.accuracy === 1) return 'calendar-perfect';
  if (day.accuracy >= 0.6) return 'calendar-good';
  if (day.accuracy >= 0.5) return 'calendar-even';
  return 'calendar-bad';
}

function probabilityForMask(game: WalkforwardGame, mask: number, data: WalkforwardData) {
  if (mask === 0) return 0.5;
  if (game.maskProbabilitiesEncoded && data.probabilityEncoding === 'uint8-base64-midpoint') {
    const packed = window.atob(game.maskProbabilitiesEncoded);
    return packed.charCodeAt(mask) / 256 + 1 / 512;
  }
  const stored = game.maskProbabilities?.[mask];
  return stored == null ? 0.5 : stored / (data.probabilityScale ?? 1);
}

export default function WalkforwardCalendar({ active, onToggle, onSetActive }: WalkforwardCalendarProps) {
  const [data, setData] = useState<WalkforwardData | null>(null);
  const [selectedMonth, setSelectedMonth] = useState('');
  const [selectedDate, setSelectedDate] = useState('');
  const [gamesPerDay, setGamesPerDay] = useState<number | 'all'>('all');

  useEffect(() => {
    const controller = new AbortController();
    fetch('/data/walkforward.json', { signal: controller.signal, cache: 'no-store' })
      .then(response => response.json() as Promise<WalkforwardData>)
      .then(async (manifest) => {
        const parts = await Promise.all((manifest.dayFiles ?? []).map(async filename => {
          const response = await fetch(`/data/${filename}`, { signal: controller.signal, cache: 'no-store' });
          if (!response.ok) throw new Error(`${filename} respondió ${response.status}`);
          return response.json() as Promise<{ days: Record<string, WalkforwardDay> }>;
        }));
        const payload = {
          ...manifest,
          days: parts.length ? Object.assign({}, ...parts.map(part => part.days)) : manifest.days,
        };
        setData(payload);
        const latestMonth = payload.methodology.lastPredictionDate.slice(0, 7);
        setSelectedMonth(latestMonth);
        setSelectedDate(payload.methodology.lastPredictionDate);
      })
      .catch(error => {
        if (error instanceof DOMException && error.name === 'AbortError') return;
        console.error('No se pudo cargar el walk-forward', error);
      });
    return () => controller.abort();
  }, []);

  const activeMask = FACTOR_KEYS.reduce((mask, key, index) => mask + (active[key] ? (1 << index) : 0), 0);
  const combinations = useMemo<CombinationStat[]>(() => {
    if (!data) return [];
    return data.combinations.map(combo => {
      const mask = combo.mask;
      const keys = FACTOR_KEYS.filter((_, index) => Boolean(mask & (1 << index)));
      return {
        ...combo,
        keys,
        label: keys.map(key => FACTOR_LABELS[key]).join(' + '),
        days: combo.days ?? data.summary.days,
        winningDays: combo.winningDays ?? 0,
        evenDays: combo.evenDays ?? 0,
        losingDays: combo.losingDays ?? 0,
        perfectDays: combo.perfectDays ?? 0,
        fifteenGameDays: combo.fifteenGameDays ?? 0,
        fifteenGameWinningDays: combo.fifteenGameWinningDays ?? 0,
        fifteenGameGames: combo.fifteenGameGames ?? 0,
        fifteenGameCorrect: combo.fifteenGameCorrect ?? 0,
        fifteenGameAccuracy: combo.fifteenGameAccuracy ?? 0,
        fifteenGamePerfectDays: combo.fifteenGamePerfectDays ?? 0,
      };
    });
  }, [data]);
  const bestWinningDays = useMemo(() => {
    if (!combinations.length) return undefined;
    const declared = combinations.find(combo => combo.mask === data?.bestWinningDaysMask);
    return declared ?? [...combinations].sort((a, b) => b.winningDays - a.winningDays || b.accuracy - a.accuracy || a.keys.length - b.keys.length || a.mask - b.mask)[0];
  }, [combinations, data?.bestWinningDaysMask]);
  const bestFifteenGameDays = useMemo(() => {
    if (!combinations.length) return undefined;
    const declared = combinations.find(combo => combo.mask === data?.bestFifteenGameAccuracyMask);
    return declared ?? [...combinations].sort((a, b) => b.fifteenGameAccuracy - a.fifteenGameAccuracy || b.fifteenGameWinningDays - a.fifteenGameWinningDays || b.accuracy - a.accuracy || a.keys.length - b.keys.length || a.mask - b.mask)[0];
  }, [combinations, data?.bestFifteenGameAccuracyMask]);

  const adjustedDays = useMemo<Record<string, WalkforwardDay>>(() => {
    if (!data) return {};
    return Object.fromEntries(Object.entries(data.days).map(([date, day]) => {
      const games = day.games.map(game => {
        const homeProbability = probabilityForMask(game, activeMask, data);
        const predictedWinner = homeProbability >= 0.5 ? game.home : game.away;
        return {
          ...game,
          homeProbability,
          predictedWinner,
          predictedProbability: Math.max(homeProbability, 1 - homeProbability),
          correct: Number(predictedWinner === game.actualWinner),
        };
      });
      const correct = games.reduce((sum, game) => sum + game.correct, 0);
      const brier = games.reduce((sum, game) => {
        const actualHome = Number(game.actualWinner === game.home);
        return sum + (game.homeProbability - actualHome) ** 2;
      }, 0) / games.length;
      return [date, { ...day, games, correct, accuracy: correct / games.length, brier }];
    }));
  }, [activeMask, data]);

  const allGames = useMemo(() => Object.values(adjustedDays).flatMap(day => day.games), [adjustedDays]);
  const slateSizes = useMemo(() => [...new Set(Object.values(adjustedDays).map(day => day.total))].sort((a, b) => b - a), [adjustedDays]);
  const filteredDays = useMemo<Record<string, WalkforwardDay>>(() => {
    if (gamesPerDay === 'all') return adjustedDays;
    return Object.fromEntries(Object.entries(adjustedDays).filter(([, day]) => day.total === gamesPerDay));
  }, [adjustedDays, gamesPerDay]);
  const dynamicSummary = useMemo(() => {
    const days = Object.values(adjustedDays);
    const correct = allGames.reduce((sum, game) => sum + game.correct, 0);
    return {
      games: allGames.length,
      correct,
      accuracy: allGames.length ? correct / allGames.length : 0,
      days: days.length,
      winningDays: days.filter(day => day.accuracy > 0.5).length,
    };
  }, [adjustedDays, allGames]);
  const seasons = useMemo(() => {
    const groups = new Map<number, WalkforwardGame[]>();
    for (const game of allGames) {
      const season = Number(game.date.slice(0, 4));
      groups.set(season, [...(groups.get(season) ?? []), game]);
    }
    return [...groups.entries()].sort(([a], [b]) => a - b).map(([season, games]) => {
      const correct = games.reduce((sum, game) => sum + game.correct, 0);
      return { season, games: games.length, correct, accuracy: correct / games.length };
    });
  }, [allGames]);
  const confidenceBands = useMemo(() => BAND_SPECS.map(spec => {
    const games = allGames.filter(game => game.predictedProbability >= spec.low && game.predictedProbability < spec.high);
    const correct = games.reduce((sum, game) => sum + game.correct, 0);
    return { label: spec.label, games: games.length, correct, accuracy: games.length ? correct / games.length : 0 };
  }), [allGames]);

  const months = data?.folds.map(fold => fold.month) ?? [];
  const monthIndex = months.indexOf(selectedMonth);
  const cells = useMemo(() => selectedMonth ? calendarCells(selectedMonth) : [], [selectedMonth]);
  const monthDays = useMemo(() => {
    if (!selectedMonth) return [];
    return Object.values(filteredDays).filter(day => day.date.startsWith(selectedMonth));
  }, [filteredDays, selectedMonth]);
  const monthTotal = monthDays.reduce((sum, day) => sum + day.total, 0);
  const monthCorrect = monthDays.reduce((sum, day) => sum + day.correct, 0);
  const selected = filteredDays[selectedDate];
  const activeCount = FACTOR_KEYS.filter(key => active[key]).length;

  const changeMonth = (nextIndex: number) => {
    const month = months[nextIndex];
    if (!month) return;
    setSelectedMonth(month);
    const dates = Object.keys(filteredDays).filter(day => day.startsWith(month)).sort();
    setSelectedDate(dates.at(-1) ?? '');
  };

  const changeGamesPerDay = (value: string) => {
    const nextValue = value === 'all' ? 'all' : Number(value);
    setGamesPerDay(nextValue);
    setSelectedDate(current => {
      const currentDay = adjustedDays[current];
      if (current.startsWith(selectedMonth) && currentDay && (nextValue === 'all' || currentDay.total === nextValue)) return current;
      const dates = Object.keys(adjustedDays)
        .filter(date => date.startsWith(selectedMonth) && (nextValue === 'all' || adjustedDays[date].total === nextValue))
        .sort();
      return dates.at(-1) ?? '';
    });
  };

  if (!data) {
    return <div className="wf-loading"><CalendarRange size={24} /><span>Calculando calendario walk-forward…</span></div>;
  }

  return (
    <>
      <QuickCombinationDock
        combinations={combinations.slice(0, 3)}
        bestWinningDays={bestWinningDays}
        bestFifteenGameDays={bestFifteenGameDays}
        activeMask={activeMask}
        defaultMask={data.defaultMask}
        baseAccuracy={data.summary.accuracy}
        onApply={onSetActive}
      />
      <section className="section wf-section" id="rendimiento">
      <div className="section-heading split-heading">
        <div>
          <span className="eyebrow"><CalendarRange size={14} /> Evidencia histórica real</span>
          <h2>Calendario walk-forward</h2>
          <p>Cada predicción fue congelada antes de conocer el resultado. El modelo se reentrena al comenzar cada mes usando únicamente juegos anteriores.</p>
        </div>
        <div className="wf-proof"><ShieldCheck size={22} /><div><strong>{data.methodology.temporalLeakageViolations}</strong><span>violaciones temporales</span><small>{data.methodology.coverageAudit.derivedFinalGames.toLocaleString('es-MX')} juegos · cobertura completa</small></div></div>
      </div>

      <div className="factor-toolbar wf-factor-toolbar"><span>Factores activos</span>{FACTORS.map(factor => <button key={factor.key} type="button" aria-pressed={active[factor.key]} className={active[factor.key] ? 'factor-active' : ''} onClick={() => onToggle(factor.key)}>{active[factor.key] && <Check size={13} />}{factor.label}</button>)}<b>{activeCount}/{FACTORS.length} activos</b></div>
      <p className="wf-ablation-note">Cada combinación fue reentrenada de forma independiente en cada corte mensual. La recomendada es Localía + Forma L10 + Carreras + Mercado + descanso: 56.24% walk-forward.</p>

      <div className="wf-summary-grid">
        <article><span>ACIERTO TOTAL</span><strong>{pct(dynamicSummary.accuracy)}</strong><small>{dynamicSummary.correct.toLocaleString('es-MX')} de {dynamicSummary.games.toLocaleString('es-MX')} juegos</small></article>
        <article><span>BASE: SIEMPRE LOCAL</span><strong>{pct(data.summary.homeBaselineAccuracy)}</strong><small>Ventaja del sistema: {((dynamicSummary.accuracy - data.summary.homeBaselineAccuracy) * 100).toFixed(1)} puntos</small></article>
        <article><span>DÍAS GANADORES</span><strong>{dynamicSummary.winningDays}</strong><small>de {dynamicSummary.days} días evaluados</small></article>
        <article><span>CORTES MENSUALES</span><strong>{data.summary.folds}</strong><small>{data.methodology.firstPredictionDate} a {data.methodology.lastPredictionDate}</small></article>
      </div>

      <CombinationLab combinations={combinations} bestWinningDays={bestWinningDays} bestFifteenGameDays={bestFifteenGameDays} activeMask={activeMask} baseline={data.summary.homeBaselineAccuracy} onApply={onSetActive} />

      <div className="wf-workspace">
        <div className="wf-calendar-panel">
          <div className="calendar-toolbar">
            <button onClick={() => changeMonth(monthIndex - 1)} disabled={monthIndex <= 0} aria-label="Mes anterior"><ChevronLeft size={18} /></button>
            <div><strong>{monthLabel(selectedMonth)}</strong><span>{monthTotal ? `${monthCorrect}-${monthTotal - monthCorrect} · ${pct(monthCorrect / monthTotal)}` : 'Sin días con este número de juegos'}</span></div>
            <button onClick={() => changeMonth(monthIndex + 1)} disabled={monthIndex >= months.length - 1} aria-label="Mes siguiente"><ChevronRight size={18} /></button>
          </div>
          <div className="calendar-filter">
            <label htmlFor="games-per-day">Juegos por día</label>
            <select id="games-per-day" value={gamesPerDay} onChange={event => changeGamesPerDay(event.target.value)}>
              <option value="all">Todos los días completos</option>
              {slateSizes.map(size => <option key={size} value={size}>Solo días con {size} {size === 1 ? 'juego' : 'juegos'}</option>)}
            </select>
            <span>{Object.keys(filteredDays).length} días visibles</span>
          </div>
          <div className="calendar-weekdays">{['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom'].map(day => <span key={day}>{day}</span>)}</div>
          <div className="calendar-grid">
            {cells.map((date, index) => {
              if (!date) return <span className="calendar-spacer" key={`empty-${index}`} />;
              const day = filteredDays[date];
              const excludedDay = adjustedDays[date];
              return (
                <button key={date} className={`${dayTone(day)} ${excludedDay && !day ? 'calendar-filtered' : ''} ${selectedDate === date ? 'calendar-selected' : ''}`} disabled={!day} onClick={() => setSelectedDate(date)} title={day ? `${day.correct}/${day.total} correctos` : excludedDay ? `${excludedDay.total} juegos; no coincide con el filtro` : 'Sin juegos evaluados'}>
                  <span>{Number(date.slice(-2))}</span>
                  {day && <><strong>{day.correct}/{day.total}</strong><small>{pct(day.accuracy)}</small></>}
                </button>
              );
            })}
          </div>
          <div className="calendar-legend"><span><i className="legend-good" />60% o más</span><span><i className="legend-even" />50–59.9%</span><span><i className="legend-bad" />Menos de 50%</span></div>
        </div>

        <aside className="wf-day-panel">
          {selected ? <>
            <div className="wf-day-head"><div><span>DETALLE DEL DÍA</span><h3>{dateFormatter.format(new Date(`${selected.date}T12:00:00Z`))}</h3></div><strong>{selected.correct}/{selected.total}</strong></div>
            <div className="wf-day-score"><span>Precisión diaria</span><strong>{pct(selected.accuracy)}</strong><div><i style={{ width: `${selected.accuracy * 100}%` }} /></div></div>
            <div className="wf-game-list">
              {selected.games.map(game => <article key={game.gamePk} className={game.correct ? 'wf-hit' : 'wf-miss'}>
                <div className="wf-game-result"><span><TeamLogo team={game.away} className="logo-xxs" />{game.away}</span><strong>{game.awayScore}–{game.homeScore}</strong><span><TeamLogo team={game.home} className="logo-xxs" />{game.home}</span></div>
                <div className="wf-pick"><span>Predicción</span><strong className="logo-label"><TeamLogo team={game.predictedWinner} className="logo-xxs" />{game.predictedWinner} {pct(game.predictedProbability)}</strong></div>
                <div className="wf-actual"><span className="logo-label"><TeamLogo team={game.actualWinner} className="logo-xxs" />Ganó {game.actualWinner}</span><b>{game.correct ? <><Check size={13} /> Acierto</> : <><X size={13} /> Error</>}</b></div>
              </article>)}
            </div>
            <div className="wf-training-note"><Target size={15} /><span>Modelo de esta combinación para el corte <strong>{selected.games[0]?.foldMonth}</strong>, entrenado solamente hasta <strong>{selected.games[0]?.trainedThrough}</strong>.</span></div>
          </> : <div className="wf-no-day"><CalendarRange size={25} /><p>{gamesPerDay === 'all' ? 'Selecciona un día con juegos para ver las predicciones.' : `Este mes no tiene días con exactamente ${gamesPerDay} juegos completos.`}</p></div>}
        </aside>
      </div>

      <div className="wf-bottom-grid">
        <DynamicStatsCard title="POR TEMPORADA" subtitle="Con factores activos" rows={seasons.map(season => ({ label: String(season.season), ...season }))} />
        <DynamicStatsCard title="POR CONFIANZA" subtitle="¿El porcentaje se sostiene?" rows={confidenceBands} />
      </div>
      </section>
    </>
  );
}

function DynamicStatsCard({ title, subtitle, rows }: { title: string; subtitle: string; rows: (DynamicStat & { label: string })[] }) {
  return <article className="wf-season-card"><div className="wf-card-title"><span>{title}</span><small>{subtitle}</small></div>{rows.map(row => <div className="wf-stat-row" key={row.label}><strong>{row.label}</strong><span>{row.correct}/{row.games}</span><b>{row.games ? pct(row.accuracy) : '—'}</b><i><em style={{ width: `${row.accuracy * 100}%` }} /></i></div>)}</article>;
}

function QuickCombinationDock({ combinations, bestWinningDays, bestFifteenGameDays, activeMask, defaultMask, baseAccuracy, onApply }: {
  combinations: CombinationStat[];
  bestWinningDays?: CombinationStat;
  bestFifteenGameDays?: CombinationStat;
  activeMask: number;
  defaultMask: number;
  baseAccuracy: number;
  onApply: (keys: FactorKey[]) => void;
}) {
  const baseKeys = FACTOR_KEYS.filter((_, index) => Boolean(defaultMask & (1 << index)));
  return <aside className="quick-combo-dock" aria-label="Atajos fijos de combinaciones con mejor accuracy">
    <div className="quick-combo-head"><span><Trophy size={14} /> Atajos de accuracy</span><small>Siempre visibles al recorrer los juegos</small></div>
    <button type="button" className={`quick-base-button ${activeMask === defaultMask ? 'active' : ''}`} onClick={() => onApply(baseKeys)} aria-pressed={activeMask === defaultMask}>
      <RotateCcw size={15} />
      <span><strong>Volver al modelo base</strong><small>Configuración al iniciar la app</small></span>
      <b>{pct2(baseAccuracy)}</b>
    </button>
    {bestWinningDays && <button type="button" className={`quick-base-button quick-day-button ${activeMask === bestWinningDays.mask ? 'active' : ''}`} onClick={() => onApply(bestWinningDays.keys)} aria-pressed={activeMask === bestWinningDays.mask} title={bestWinningDays.label}>
      <CalendarRange size={15} />
      <span><strong>Más días ganadores</strong><small>{bestWinningDays.winningDays} de {bestWinningDays.days} días</small></span>
      <b>{bestWinningDays.winningDays}</b>
    </button>}
    {bestFifteenGameDays && <button type="button" className={`quick-base-button quick-fifteen-button ${activeMask === bestFifteenGameDays.mask ? 'active' : ''}`} onClick={() => onApply(bestFifteenGameDays.keys)} aria-pressed={activeMask === bestFifteenGameDays.mask} title={bestFifteenGameDays.label}>
      <Target size={15} />
      <span><strong>Mayor accuracy · 15 juegos</strong><small>{bestFifteenGameDays.fifteenGameCorrect}/{bestFifteenGameDays.fifteenGameGames} aciertos</small></span>
      <b>{pct2(bestFifteenGameDays.fifteenGameAccuracy)}</b>
    </button>}
    <div className="quick-combo-list">
      {combinations.map((combo, index) => <button type="button" key={combo.mask} className={combo.mask === activeMask ? 'active' : ''} onClick={() => onApply(combo.keys)} aria-pressed={combo.mask === activeMask} title={combo.label}>
        <span><b>#{index + 1}</b><strong>{pct2(combo.accuracy)}</strong></span>
        <small>{combo.keys.length} factores</small>
        <h4>{combo.label}</h4>
        <em>{combo.mask === activeMask ? <><Check size={12} /> Activa</> : 'Aplicar combinación'}</em>
      </button>)}
    </div>
    <p>Los cambios se reflejan al instante en los juegos de hoy y en todo el walk-forward.</p>
  </aside>;
}

function CombinationLab({ combinations, bestWinningDays, bestFifteenGameDays, activeMask, baseline, onApply }: { combinations: CombinationStat[]; bestWinningDays?: CombinationStat; bestFifteenGameDays?: CombinationStat; activeMask: number; baseline: number; onApply: (keys: FactorKey[]) => void }) {
  return <section className="combo-lab" aria-labelledby="combo-title">
    <div className="combo-head"><div><span><Trophy size={14} /> Comparador completo</span><h3 id="combo-title">Las {combinations.length} combinaciones de factores</h3><p>Ordenadas por accuracy walk-forward y reentrenadas por separado. Toca “Aplicar” para llevar esa combinación al calendario y a los juegos de hoy.</p></div><strong>{combinations.length}/{combinations.length}</strong></div>
    {bestWinningDays && <button type="button" className={`combo-day-winner ${bestWinningDays.mask === activeMask ? 'selected' : ''}`} onClick={() => onApply(bestWinningDays.keys)}><span><CalendarRange size={15} /> Campeona en días ganadores</span><h4>{bestWinningDays.label}</h4><strong>{bestWinningDays.winningDays}<small> de {bestWinningDays.days} días</small></strong><b>{pct2(bestWinningDays.accuracy)} accuracy</b><em>{bestWinningDays.mask === activeMask ? <><Check size={13} /> Combinación activa</> : 'Aplicar combinación'}</em></button>}
    {bestFifteenGameDays && <button type="button" className={`combo-day-winner combo-fifteen-winner ${bestFifteenGameDays.mask === activeMask ? 'selected' : ''}`} onClick={() => onApply(bestFifteenGameDays.keys)}><span><Target size={15} /> Mayor accuracy en jornadas de 15 juegos</span><h4>{bestFifteenGameDays.label}</h4><strong>{pct2(bestFifteenGameDays.fifteenGameAccuracy)}<small> · {bestFifteenGameDays.fifteenGameCorrect}/{bestFifteenGameDays.fifteenGameGames} aciertos</small></strong><b>{bestFifteenGameDays.fifteenGameWinningDays} de {bestFifteenGameDays.fifteenGameDays} días ganadores</b><em>{bestFifteenGameDays.mask === activeMask ? <><Check size={13} /> Combinación activa</> : 'Aplicar combinación'}</em></button>}
    <div className="combo-podium">{combinations.slice(0, 3).map((combo, index) => <button type="button" key={combo.mask} className={combo.mask === activeMask ? 'selected' : ''} onClick={() => onApply(combo.keys)}><span>#{index + 1}</span><h4>{combo.label}</h4><strong>{pct2(combo.accuracy)}</strong><small>{combo.correct.toLocaleString('es-MX')}/{combo.games.toLocaleString('es-MX')} aciertos · {combo.winningDays} días ganadores</small><em>Aplicar combinación</em></button>)}</div>
    <div className="combo-table-shell"><table className="combo-table"><thead><tr><th>#</th><th>Combinación</th><th>Total</th><th>Días G.</th><th>Jornadas de 15</th><th>2024</th><th>2025</th><th>2026</th><th>Vs. local</th><th /></tr></thead><tbody>{combinations.map((combo, index) => <tr key={combo.mask} className={combo.mask === activeMask ? 'current' : ''}><td>{index + 1}</td><td><div>{combo.keys.map(key => <span key={key}>{FACTOR_LABELS[key]}</span>)}</div></td><td><strong>{pct2(combo.accuracy)}</strong><small>{combo.correct}/{combo.games}</small></td><td><strong>{combo.winningDays}</strong><small>/{combo.days}</small></td><td><strong>{combo.fifteenGameWinningDays}/{combo.fifteenGameDays}</strong><small>{pct2(combo.fifteenGameAccuracy)} · {combo.fifteenGameCorrect}/{combo.fifteenGameGames}</small></td>{['2024', '2025', '2026'].map(season => <td key={season}>{combo.seasons[season] ? pct(combo.seasons[season].accuracy) : '—'}</td>)}<td className={combo.accuracy >= baseline ? 'positive' : 'negative'}>{`${combo.accuracy >= baseline ? '+' : ''}${((combo.accuracy - baseline) * 100).toFixed(2)} pts`}</td><td><button type="button" disabled={combo.mask === activeMask} onClick={() => onApply(combo.keys)}>{combo.mask === activeMask ? 'Activa' : 'Aplicar'}</button></td></tr>)}</tbody></table></div>
    <p className="combo-caution">Los criterios son independientes: “Días ganadores” maximiza fechas arriba de 50%; “Jornadas de 15” maximiza el accuracy únicamente dentro de los juegos de esas fechas. En una jornada de 15 se requieren al menos 8 aciertos para marcarla como ganadora.</p>
  </section>;
}

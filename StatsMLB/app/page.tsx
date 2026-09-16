'use client';

import {
  Activity,
  ArrowDownUp,
  BarChart3,
  CircleDotDashed as Baseball,
  CalendarDays,
  CheckCircle2,
  ChevronDown,
  CircleGauge,
  ClipboardList,
  FlaskConical,
  Home,
  Info,
  RefreshCw,
  Send,
  ShieldCheck,
  Sparkles,
  TrendingUp,
  Trophy,
  X,
} from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { createPortal } from 'react-dom';
import DashboardTabs from './components/DashboardTabs';
import TeamLogo from './components/TeamLogo';
import WalkforwardCalendar from './components/WalkforwardCalendar';

type TeamMetric = {
  team: string;
  name: string;
  games: number;
  wins: number;
  losses: number;
  winPct: number;
  last10Wins: number;
  last10Losses: number;
  last10Pct: number;
  runDiff: number;
  runDiffPerGame: number;
  lastGameDate: string;
};

type SerializedModel = {
  kind: 'constant' | 'logistic';
  probability?: number;
  inputFeatures: string[];
  imputeValues?: number[];
  indicatorFeatures?: number[];
  means?: number[];
  scales?: number[];
  coefficients?: number[];
  intercept?: number;
};

type SeriesContext = {
  code: string;
  label: string;
  wins?: number;
  losses?: number;
  opponent?: string;
  runDiff?: number;
};

type RankingRow = {
  team: string;
  team_name: string;
  three_game_series: number;
  sweeps: number;
  sweep_rate: number;
  times_swept: number;
  swept_rate: number;
  net_sweeps: number;
  net_sweep_rate: number;
  baseline_win_rate: number;
  after_sweep_games: number;
  after_sweep_wins: number;
  after_sweep_win_rate: number | null;
  after_swept_games: number;
  after_swept_wins: number;
  after_swept_win_rate: number | null;
};

type IndicationPairRow = {
  key: string;
  awayLabel: string;
  homeLabel: string;
  games: number;
  awayWins: number;
  homeWins: number;
  leader: 'home' | 'away' | 'even';
  leaderWinRate: number;
  games2026: number;
  awayWins2026: number;
  homeWins2026: number;
  walkforwardGames: number;
  walkforwardCorrect: number;
  walkforwardAccuracy: number | null;
  walkforwardHomeAccuracy: number | null;
  deltaVsHomePoints: number | null;
};

type IndicationPairSignal = {
  title: string;
  eligibleGames: number;
  pairCount: number;
  summary: {
    games: number;
    correct: number;
    accuracy: number | null;
    homeBaselineAccuracy: number | null;
    deltaVsHomePoints: number | null;
  };
  rows: IndicationPairRow[];
};

type SweepRate = { games: number; sweeps: number; rate: number | null };
type SweepStageMetrics = {
  events: number;
  sweeps: number;
  observedRate: number | null;
  averageProbability: number | null;
  brier: number | null;
  accuracyAt50: number | null;
  correctAt50: number;
  seasons?: Record<string, SweepStageMetrics>;
};

type StatsData = {
  generatedAt: string;
  cutoffDate: string;
  playerNames?: Record<string, string>;
  coverage: { final_regular_games: number; first_final_date: string; last_final_date: string };
  seriesSummary: { eligible_three_game_series: number; swept_three_game_series: number; league_sweep_frequency: number };
  afterSweepSummary: { sweeps_with_next_game: number; next_game_wins: number; next_game_win_rate: number };
  afterSweptSummary: { events_with_next_game: number; next_game_wins: number; next_game_win_rate: number };
  rankings: Record<string, RankingRow[]>;
  scenarios: {
    bothSweep: { games: number; homeWins: number; awayWins: number; homeWinRate: number };
    bothSwept: { games: number; homeWins: number; awayWins: number; homeWinRate: number };
    sweptVsLost12: {
      games: number;
      sweptWins: number;
      lost12Wins: number;
      sweptWinRate: number;
      sweptAtHome: { games: number; wins: number };
      sweptAway: { games: number; wins: number };
    };
  };
  teams: Record<string, TeamMetric>;
  todayContext: Record<string, SeriesContext>;
  model: {
    defaultMask: number;
    factorKeys: FactorKey[];
    factorGroups: Record<string, string[]>;
    modelsByMask: Record<string, SerializedModel>;
    metrics: {
      trainRange: string;
      holdoutRange: string;
      trainGames: number;
      holdoutGames: number;
      accuracy: number;
      brier: number;
      homeBaselineAccuracy: number;
    };
  };
  previousGameAudit: {
    method: string;
    cutoffDate: string;
    games: number;
    baseline: { accuracy: number; correct: number; accuracy2026: number };
    candidates: { name: string; accuracy: number; correct: number; deltaPoints: number; accuracy2026: number; delta2026Points: number; ci95Points?: number[]; status: string; reason: string }[];
    descriptive: { label: string; games: number; nextWinRate: number; nextWinRate2026: number }[];
  };
  bestPlayersAudit: {
    method: string;
    cutoffDate: string;
    coverage: { teamGames: number; lineupQualityGames: number };
    games: number;
    baseline: { accuracy: number; correct: number; accuracy2026: number };
    candidates: { name: string; accuracy: number; correct: number; deltaPoints: number; accuracy2026: number; delta2026Points: number; ci95Points: number[]; status: string; reason: string; seasons: Record<string, number> }[];
    descriptive: { label: string; games: number; nextWinRate: number; nextWinRate2026: number }[];
    latestTeamSignals: Record<string, { gamePk: number; playedAt: string; signals: Record<string, number> }>;
    recommendation: string;
  };
  coachRotationAudit: {
    method: string;
    cutoffDate: string;
    coverage: { teamGames: number; rotationComparisons: number; averageLineupSize: number };
    descriptive: { label: string; previousResult: 'loss' | 'win'; rotation: string; games: number; wins: number; winRate: number; winRate2026: number }[];
    latestTeamStates: Record<string, { gamePk: number; playedAt: string; win: number; players: number[]; slots: Record<string, number>; top4: number[] }>;
    candidate: {
      mask: number; games: number; correct: number; accuracy: number; deltaPoints: number; brier: number;
      baseline: { mask: number; correct: number; accuracy: number; brier: number };
      seasons: Record<string, { games: number; correct: number; accuracy: number }>;
      delta2026Points: number; ci95Points: number[]; foldsBetter: number; foldsEqual: number; foldsWorse: number; status: string;
    };
    recommendation: string;
  };
  rotationQualityAudit: {
    method: string;
    cutoffDate: string;
    coverage: { teamGames: number; qualityComparisons: number; ratedPlayers: number };
    descriptive: { label: string; games: number; wins: number; winRate: number; winRate2026: number }[];
    latestPlayerRatings: Record<string, Record<string, number>>;
    candidate: {
      mask: number; games: number; correct: number; accuracy: number; deltaPoints: number; brier: number;
      baseline: { mask: number; correct: number; accuracy: number; brier: number };
      seasons: Record<string, { games: number; correct: number; accuracy: number }>;
      delta2026Points: number; ci95Points: number[]; foldsBetter: number; foldsEqual: number; foldsWorse: number; status: string;
    };
    combinedWithCoach: {
      mask: number; games: number; correct: number; accuracy: number; deltaPoints: number;
      seasons: Record<string, { games: number; correct: number; accuracy: number }>;
    };
    recommendation: string;
  };
  lineupFatigueAudit: {
    method: string;
    cutoffDate: string;
    coverage: { teamGames: number; lineupComparisons: number; trackedPlayers: number };
    descriptive: { label: string; games: number; wins: number; winRate: number; winRate2026: number }[];
    latestPlayerWorkloads: Record<string, Record<string, { streak: number; events: [string, number, number][]; quality: number }>>;
    latestTeamStates: Record<string, { win: number; extraInnings: number; doubleheaderGame: number; playedAt: string }>;
    candidate: {
      mask: number; games: number; correct: number; accuracy: number; deltaPoints: number; brier: number;
      baseline: { mask: number; correct: number; accuracy: number; brier: number };
      seasons: Record<string, { games: number; correct: number; accuracy: number }>;
      delta2026Points: number; ci95Points: number[]; foldsBetter: number; foldsEqual: number; foldsWorse: number;
      design_2024_2025: { games: number; accuracy: number; deltaPoints: number };
      h1_2026: { games: number; accuracy: number; deltaPoints: number };
      h2_2026: { games: number; accuracy: number; deltaPoints: number };
      passesGate: boolean; selectionRule: string; status: string;
    };
    combinedWithRotationQuality: { accuracy: number; deltaPoints: number };
    recommendation: string;
  };
  opponentFormAudit: {
    method: string;
    cutoffDate: string;
    coverage: { teamGames: number; teams: number };
    descriptive: { label: string; games: number; wins: number; winRate: number; winRate2026: number }[];
    latestTeamSignals: Record<string, { opponentStrengthL10: number; opponentStrengthL20: number }>;
    candidate: {
      mask: number; games: number; correct: number; accuracy: number; deltaPoints: number; brier: number;
      baseline: { mask: number; correct: number; accuracy: number; brier: number };
      seasons: Record<string, { games: number; correct: number; accuracy: number }>;
      delta2026Points: number; ci95Points: number[]; foldsBetter: number; foldsEqual: number; foldsWorse: number;
      design_2024_2025: { games: number; accuracy: number; deltaPoints: number };
      h1_2026: { games: number; accuracy: number; deltaPoints: number };
      h2_2026: { games: number; accuracy: number; deltaPoints: number };
      passesGate: boolean; selectionRule: string; status: string;
    };
    recommendation: string;
  };
  indicationPairAudit: {
    method: string;
    cutoffDate: string;
    minimumPriorGames: number;
    signals: Record<'coachRotation' | 'rotationQuality' | 'lineupFatigue', IndicationPairSignal>;
  };
  seriesSweepAudit: {
    status: string;
    cutoffDate: string;
    method: string;
    temporalLeakageViolations: number;
    historical: {
      beforeSeries: SweepRate;
      afterGame1: SweepRate;
      afterGame2: SweepRate;
      afterPreviousSweep: SweepRate;
      doubleSweepUnconditional: { pairs: number; doubleSweeps: number; rate: number };
    };
    walkforward: {
      stages: Record<'beforeSeries' | 'afterGame1' | 'afterGame2', SweepStageMetrics>;
      doubleSweep: SweepStageMetrics;
    };
    estimator: Record<'afterGame1' | 'afterGame2', {
      overall: SweepRate;
      byLocation: Record<string, SweepRate>;
      byTeam: Record<string, SweepRate>;
      byOpponent: Record<string, SweepRate>;
    }>;
    latestCompleted: Record<string, { endDate: string; opponent: string; swept: boolean; wins: number; losses: number }>;
    teams: { team: string; name: string; series: number; sweeps: number; sweepRate: number; stage2Games: number; stage2Sweeps: number; stage2Rate: number | null; repeatAttempts: number; repeatSweeps: number; repeatRate: number | null }[];
    recommendation: string;
  };
  additionalAnalyses: { name: string; status: string; detail: string }[];
};

type SeasonL10Rate = {
  games: number;
  wins: number;
  winRate: number | null;
  seasons: Record<string, { games: number; wins: number; winRate: number }>;
};

type SeasonL10Audit = {
  generatedAt: string;
  cutoffDate: string;
  status: string;
  method: string;
  temporalLeakageViolations: number;
  coverage: { games: number; firstDate: string; lastDate: string; minimumPriorSeasonGamesPerTeam: number; requiredPriorGamesL10: number };
  exactScenario: { label: string; broad: SeasonL10Rate; strict: SeasonL10Rate; strictDefinition: string };
  patterns: (SeasonL10Rate & { key: string; label: string })[];
  walkforward: {
    folds: number;
    games: number;
    gainVsSeasonPoints: number;
    gainVsSeason2026Points: number;
    models: { key: string; label: string; games: number; correct: number; accuracy: number; brier: number; seasons: Record<string, { games: number; correct: number; accuracy: number }> }[];
  };
  productionModel: null | {
    withL10: { games: number; correct: number; accuracy: number; seasons: Record<string, { games: number; correct: number; accuracy: number }> };
    withoutL10: { games: number; correct: number; accuracy: number; seasons: Record<string, { games: number; correct: number; accuracy: number }> };
    deltaPoints: number;
    delta2026Points: number;
    pairedMasks: { improved: number; tied: number; worsened: number };
  };
  recommendation: string;
};

type ScheduleGame = {
  gamePk: number;
  gameDate: string;
  officialDate: string;
  seriesGameNumber?: number;
  gamesInSeries?: number;
  seriesDescription?: string;
  status: string;
  abstractState: string;
  venue: string;
  away: { team: string; name: string; score?: number; pitcher?: string; lineup?: ConfirmedLineup | null };
  home: { team: string; name: string; score?: number; pitcher?: string; lineup?: ConfirmedLineup | null };
};

type ConfirmedLineup = { confirmed: boolean; players: { playerId: number; battingOrder: number; name?: string }[] };

type ScheduleData = {
  sourceState: string;
  source: string;
  retrievedAt: string;
  selectedDate?: string;
  warning?: string;
  games: ScheduleGame[];
};

type FrozenWalkforwardGame = {
  gamePk: number;
  foldMonth: string;
  trainedThrough: string;
  homeProbability: number;
  maskProbabilitiesEncoded?: string;
};

type FrozenWalkforwardDay = {
  date: string;
  games: FrozenWalkforwardGame[];
};

type FrozenGamePrediction = {
  homeProbability: number;
  foldMonth: string;
  trainedThrough: string;
};

type OddsModelStat = { n: number; mae_american: number | null; mae_decimal: number | null; direction_accuracy: number | null };
type OddsSeasonBlock = { games: number; models: Record<string, OddsModelStat> };
type OddsEnginePayload = {
  generatedAt: string;
  minHistory: number;
  totalTeamGameRows: number;
  overall: OddsSeasonBlock;
  seasons: Record<string, OddsSeasonBlock>;
};
type OddsPredictionGame = {
  gamePk: number;
  gameDate: string;
  awayTeam: string;
  homeTeam: string;
  awayPredicted: number | null;
  homePredicted: number | null;
  awayMarket: number | null;
  homeMarket: number | null;
  awayDelta: number | null;
  homeDelta: number | null;
  awayLast: number | null;
  homeLast: number | null;
  awayRoll5: number | null;
  homeRoll5: number | null;
  awayGamesSeen: number;
  homeGamesSeen: number;
};
type OddsPredictionsPayload = { generatedAt: string; model: string; note?: string; games: OddsPredictionGame[] };
type OddsHistoryEntry = { marketPHome: number; gameDate: string };
type OddsHistoryPayload = { generatedAt: string; games: Record<string, OddsHistoryEntry> };

type CardEvidenceProfile = {
  conditions: string[];
  sampleSize: number;
  wins: number;
  winRate: number;
  baselineWinRate: number;
  delta: number;
  lower95: number;
};

type PlayerPerformanceEntry = {
  kind: 'batter' | 'pitcher';
  name: string;
  games: number;
  label: string;
  pa?: number;
  avg?: number | null;
  ops?: number | null;
  hr?: number;
  rbi?: number;
  runs?: number;
  sb?: number;
  kPct?: number | null;
  bbPct?: number | null;
  ip?: number | null;
  era?: number | null;
  whip?: number | null;
  k9?: number | null;
  bb9?: number | null;
  so?: number;
  seasonOps?: number | null;
  talentTier?: 'elite' | 'regular' | 'weak';
};

type PlayerPerformancePayload = {
  generatedAt: string;
  windowDays: number;
  cutoffDate: string;
  players: Record<string, PlayerPerformanceEntry>;
};

type HistoryTeamState = {
  gamePk: number;
  playedAt: string;
  win: number;
  players: number[];
  slots: Record<string, number>;
  top4: number[];
};

type HistorySnapshot = {
  date: string;
  windowDays: number;
  cutoffDate: string;
  generatedAt: string;
  players: Record<string, PlayerPerformanceEntry>;
  teamStates: Record<string, HistoryTeamState>;
};

type HistoryIndex = {
  dates: string[];
  season: number;
  windowDays: number;
};

type CardEvidenceSignal = {
  code: string;
  label: string;
  direction: 'supports' | 'challenges' | 'context';
  sampleSize: number;
  winRate: number;
  baselineWinRate: number;
  delta: number;
  lower95: number;
  upper95: number;
};

type CardEvidencePayload = {
  generatedAt: string;
  baselinePickWinRate?: number;
  minimumSampleSize?: number;
  minimumValidation?: number;
  signals?: CardEvidenceSignal[];
  profiles: CardEvidenceProfile[];
};

type BestOdds = { book: string; american: number; decimal: number };
type CurrentOddsGame = {
  eventId: string;
  commenceTime: string;
  awayTeam: string;
  homeTeam: string;
  lastUpdated: string;
  bestAway: BestOdds | null;
  bestHome: BestOdds | null;
  books?: { name: string; awayDecimal: number | null; homeDecimal: number | null }[];
  marketMoveHomePp?: number | null;
  marketMoveBooks?: number;
  marketMoveFrom?: string | null;
};
type OddsData = {
  sourceState: string;
  source: string;
  generatedAt: string;
  sourceUpdatedAt?: string;
  selectionRule?: string;
  games: CurrentOddsGame[];
};

type RefreshSummary = {
  status: 'success' | 'partial' | 'error';
  finishedAt: string;
  yesterdayGames: number;
  todayGames: number;
  tomorrowGames: number;
  finalGames: number;
  oddsEvents: number;
  oddsBooks: number;
  errors: string[];
};

type RefreshProgress = {
  percent: number;
  stage: string;
  message: string;
};

type RefreshStreamEvent =
  | { type: 'progress'; progress: RefreshProgress }
  | { type: 'complete'; result: RefreshSummary }
  | { type: 'error'; error: string };

type FactorKey = 'localia' | 'strength' | 'recent' | 'runDiff' | 'series' | 'marketSchedule' | 'bestPlayersTest' | 'coachRotationTest' | 'rotationQualityTest' | 'lineupFatigueTest' | 'opponentFormTest';

const FACTORS: { key: FactorKey; label: string; detail: string }[] = [
  { key: 'localia', label: 'Localía', detail: 'Ventaja histórica del equipo local' },
  { key: 'strength', label: 'Temporada', detail: 'Récord actual con suavizado' },
  { key: 'recent', label: 'Forma L10', detail: 'Resultados de los últimos 10 juegos' },
  { key: 'runDiff', label: 'Carreras', detail: 'Diferencial de carreras por juego' },
  { key: 'series', label: 'Serie previa', detail: 'Barrida, barrido o derrota 1-2 inmediata' },
  { key: 'marketSchedule', label: 'Mercado + descanso', detail: 'Consenso sin margen disponible antes del juego y diferencia de días de descanso' },
  { key: 'bestPlayersTest', label: 'Mejores jugadores (prueba)', detail: 'Uso de los mejores bateadores en el juego anterior, separado por victoria o derrota; señal experimental' },
  { key: 'coachRotationTest', label: 'Rotación coach (prueba)', detail: 'Cambios de titulares, orden al bate y top 4 según el resultado anterior; requiere ambas alineaciones confirmadas' },
  { key: 'rotationQualityTest', label: 'Calidad rotación (prueba)', detail: 'Valora la calidad previa de quienes entran, salen y permanecen en el lineup; requiere ambas alineaciones confirmadas' },
  { key: 'lineupFatigueTest', label: 'Fatiga lineup (prueba)', detail: 'Titularidades y apariciones de plato acumuladas en 3/7 días, más descanso menor a 30 horas; requiere ambas alineaciones confirmadas' },
  { key: 'opponentFormTest', label: 'Forma vs rival (prueba)', detail: 'Fuerza previa de los rivales enfrentados en los últimos 10 y 20 juegos; validado únicamente junto con Fatiga + Calidad' },
];

const pct = (value?: number | null, digits = 1) => value == null ? '—' : `${(value * 100).toFixed(digits)}%`;
const alias = (team: string) => team === 'OAK' ? 'ATH' : team;
const logistic = (value: number) => 1 / (1 + Math.exp(-value));
const normalizeTeamName = (value: string) => value.normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase().replace(/[^a-z0-9]/g, '');
const gameTime = (value: string) => new Intl.DateTimeFormat('es-MX', { timeZone: 'America/Mexico_City', hour: '2-digit', minute: '2-digit', hourCycle: 'h23' }).format(new Date(value));
const gameTime12 = (value: string) => new Intl.DateTimeFormat('en-US', { timeZone: 'America/Mexico_City', hour: 'numeric', minute: '2-digit', hour12: true }).format(new Date(value));
const shortDate = (value: string) => new Intl.DateTimeFormat('es-MX', { timeZone: 'UTC', weekday: 'short', day: 'numeric' }).format(new Date(`${value}T12:00:00Z`)).replace('.', '');

function mexicoIsoDate(offsetDays = 0) {
  const parts = new Intl.DateTimeFormat('en-CA', { timeZone: 'America/Mexico_City', year: 'numeric', month: '2-digit', day: '2-digit' }).formatToParts(new Date());
  const values = Object.fromEntries(parts.map(part => [part.type, part.value]));
  const date = new Date(`${values.year}-${values.month}-${values.day}T12:00:00Z`);
  date.setUTCDate(date.getUTCDate() + offsetDays);
  return date.toISOString().slice(0, 10);
}

function activeFactorMask(active: Record<FactorKey, boolean>) {
  return FACTORS.reduce((mask, factor, index) => mask + (active[factor.key] ? 1 << index : 0), 0);
}

function maskToActive(mask: number): Record<FactorKey, boolean> {
  const active = {} as Record<FactorKey, boolean>;
  FACTORS.forEach((factor, index) => { active[factor.key] = Boolean(mask & (1 << index)); });
  return active;
}

type StrikecastPick = {
  gamePk: number;
  away: string;
  home: string;
  pHome: number | null;
  pick: string | null;
  confidenceTier: string | null;
  confidencePp: number | null;
  f5PHome: number | null;
  seriesFlip: boolean | null;
  teamFlip: boolean | null;
  isPlayed: boolean | null;
  homeScore: number | null;
  awayScore: number | null;
};

type ValuePick = {
  gamePk: number;
  home: string;
  away: string;
  homeMl: number | null;
  awayMl: number | null;
  pHomeWin: number | null;
  pHomeFirst: number | null;
  impHome: number | null;
  edgeWin: number | null;
  edgeFirst: number | null;
  pickCode: string | null;
  pickProb: number | null;
  tier: 'VALOR-FUERTE' | 'VALOR' | 'NEUTRO';
};

type ComparisonMaskSet = {
  base: number;
  topAccuracy: number;
  fifteenGames: number;
  winningDays: number;
  best2026: number;
};

function frozenPredictionForMask(game: FrozenWalkforwardGame | undefined, mask: number): FrozenGamePrediction | undefined {
  if (!game) return undefined;
  let homeProbability = game.homeProbability;
  if (game.maskProbabilitiesEncoded && typeof window !== 'undefined') {
    const packed = window.atob(game.maskProbabilitiesEncoded);
    if (mask >= 0 && mask < packed.length) homeProbability = packed.charCodeAt(mask) / 256 + 1 / 512;
  }
  return { homeProbability, foldMonth: game.foldMonth, trainedThrough: game.trainedThrough };
}

function findCurrentOdds(game: ScheduleGame, odds: OddsData | null) {
  if (!odds) return undefined;
  const away = normalizeTeamName(game.away.name);
  const home = normalizeTeamName(game.home.name);
  const matches = odds.games.filter(item => normalizeTeamName(item.awayTeam) === away && normalizeTeamName(item.homeTeam) === home);
  const closest = matches.sort((a, b) => Math.abs(new Date(a.commenceTime).getTime() - new Date(game.gameDate).getTime()) - Math.abs(new Date(b.commenceTime).getTime() - new Date(game.gameDate).getTime()))[0];
  if (!closest) return undefined;
  return Math.abs(new Date(closest.commenceTime).getTime() - new Date(game.gameDate).getTime()) <= 12 * 60 * 60 * 1000 ? closest : undefined;
}

type PickType = 'ML' | 'F5_ML' | 'RL_MINUS_1_5' | 'RL_PLUS_1_5' | 'NRFI' | 'YRFI' | 'OVER' | 'UNDER';
const PICK_TYPE_OPTIONS: { value: PickType; label: string; needsTeam: boolean; needsTotal: boolean }[] = [
  { value: 'ML', label: 'Money Line (ganador)', needsTeam: true, needsTotal: false },
  { value: 'F5_ML', label: 'F5 ML (primeras 5 entradas)', needsTeam: true, needsTotal: false },
  { value: 'RL_MINUS_1_5', label: 'Run Line -1.5', needsTeam: true, needsTotal: false },
  { value: 'RL_PLUS_1_5', label: 'Run Line +1.5', needsTeam: true, needsTotal: false },
  { value: 'NRFI', label: 'NRFI (sin carrera 1er inning)', needsTeam: false, needsTotal: false },
  { value: 'YRFI', label: 'YRFI (carrera en 1er inning)', needsTeam: false, needsTotal: false },
  { value: 'OVER', label: 'OVER (más de X carreras)', needsTeam: false, needsTotal: true },
  { value: 'UNDER', label: 'UNDER (menos de X carreras)', needsTeam: false, needsTotal: true },
];

function pickLineText(type: PickType, pickTeam: string, totalPoint: string): string {
  switch (type) {
    case 'ML': return `${pickTeam || '—'} ML`;
    case 'F5_ML': return `${pickTeam || '—'} ML (F5)`;
    case 'RL_MINUS_1_5': return `${pickTeam || '—'} -1.5 (Run Line)`;
    case 'RL_PLUS_1_5': return `${pickTeam || '—'} +1.5 (Run Line)`;
    case 'NRFI': return `NRFI (sin carrera 1er inning)`;
    case 'YRFI': return `YRFI (carrera en 1er inning)`;
    case 'OVER': return `OVER ${totalPoint || '—'}`;
    case 'UNDER': return `UNDER ${totalPoint || '—'}`;
  }
}

function telegramPreview(pickNumber: number, game: ScheduleGame, pickLine: string, decimalOdds?: number, live?: boolean) {
  const liveBadge = live ? `\n🔴 EN VIVO` : '';
  return `🚨 PICK #${pickNumber || '—'} | MLB${liveBadge}\n\n⚾️ ${game.away.name} vs ${game.home.name}\n🕣 Hora: ${gameTime(game.gameDate)}\n\n🎯 Pick: ${pickLine}\n💰 Momio: ${decimalOdds !== undefined && Number.isFinite(decimalOdds) ? decimalOdds.toFixed(2) : '—'}`;
}

function marketPickFromCurrentOdds(game: ScheduleGame, currentOdds?: CurrentOddsGame, oddsHistory?: OddsHistoryPayload | null): { code: string; prob: number } | null {
  const homeCode = alias(game.home.team);
  const awayCode = alias(game.away.team);
  if (currentOdds) {
    const probs = (currentOdds.books ?? []).flatMap(book => {
      if (!book.homeDecimal || !book.awayDecimal) return [];
      const rawHome = 1 / book.homeDecimal;
      const rawAway = 1 / book.awayDecimal;
      return [rawHome / (rawHome + rawAway)];
    });
    if (probs.length) {
      const pHome = probs.reduce((s, v) => s + v, 0) / probs.length;
      return pHome >= 0.5 ? { code: homeCode, prob: pHome } : { code: awayCode, prob: 1 - pHome };
    }
  }
  // Fallback histórico (walkforward_preds.parquet vía odds-history.json)
  const histEntry = oddsHistory?.games[String(game.gamePk)];
  if (histEntry) {
    const pHome = histEntry.marketPHome;
    return pHome >= 0.5 ? { code: homeCode, prob: pHome } : { code: awayCode, prob: 1 - pHome };
  }
  return null;
}

function marketSnapshot(odds?: CurrentOddsGame) {
  const probabilities = (odds?.books ?? []).flatMap(book => {
    if (!book.homeDecimal || !book.awayDecimal) return [];
    const rawHome = 1 / book.homeDecimal;
    const rawAway = 1 / book.awayDecimal;
    return [rawHome / (rawHome + rawAway)];
  });
  if (!probabilities.length) return { logit: null, providers: null, std: null };
  const mean = probabilities.reduce((sum, value) => sum + value, 0) / probabilities.length;
  const variance = probabilities.reduce((sum, value) => sum + (value - mean) ** 2, 0) / probabilities.length;
  return { logit: Math.log(mean / (1 - mean)), providers: probabilities.length, std: Math.sqrt(variance) };
}

function daysBetween(gameDate: string, lastGameDate?: string) {
  if (!lastGameDate) return null;
  return Math.round((new Date(`${gameDate}T12:00:00Z`).getTime() - new Date(`${lastGameDate}T12:00:00Z`).getTime()) / 86_400_000);
}

async function fetchWithTimeout(input: RequestInfo | URL, init: RequestInit = {}, timeoutMs = 10_000) {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(input, { ...init, signal: controller.signal });
  } finally {
    window.clearTimeout(timer);
  }
}

async function readJsonResponse<T>(response: Response) {
  const body = await response.text();
  if (!body.trim()) throw new Error(`El servidor respondió vacío (${response.status})`);
  try {
    return JSON.parse(body) as T;
  } catch {
    throw new Error(response.ok
      ? 'El servidor devolvió una respuesta incompleta. Intenta nuevamente.'
      : `El servidor tuvo un error temporal (${response.status}).`);
  }
}

async function requestDailyRefresh(onProgress: (progress: RefreshProgress) => void) {
  let lastError: Error | null = null;
  for (let attempt = 0; attempt < 2; attempt += 1) {
    let receivedProgress = false;
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 60_000);
    try {
      const response = await fetch('/api/refresh?stream=1', {
        method: 'POST',
        cache: 'no-store',
        headers: { Accept: 'application/x-ndjson' },
        signal: controller.signal,
      });
      if (!response.ok) {
        const errorBody = await response.text();
        throw new Error(errorBody || `La actualización respondió ${response.status}`);
      }
      if (!response.body) throw new Error('El servidor no pudo abrir el canal de progreso');

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      const resultRef: { current: RefreshSummary | null } = { current: null };
      let streamError = '';

      const consumeLine = (line: string) => {
        if (!line.trim()) return;
        const event = JSON.parse(line) as RefreshStreamEvent;
        if (event.type === 'progress') {
          receivedProgress = true;
          onProgress(event.progress);
        } else if (event.type === 'complete') {
          resultRef.current = event.result;
        } else {
          streamError = event.error;
        }
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';
        lines.forEach(consumeLine);
      }
      buffer += decoder.decode();
      consumeLine(buffer);

      if (streamError) throw new Error(streamError);
      const result = resultRef.current;
      if (!result) throw new Error('La actualización terminó sin un resumen verificable');
      if (result.status === 'error') throw new Error(result.errors?.join(' · ') || 'No se pudo actualizar');
      return result;
    } catch (error) {
      lastError = error instanceof Error ? error : new Error('No se pudo actualizar');
      if (attempt === 0 && !receivedProgress) await new Promise(resolve => window.setTimeout(resolve, 1_000));
      else throw lastError;
    } finally {
      window.clearTimeout(timeout);
    }
  }
  throw lastError ?? new Error('No se pudo actualizar');
}

function coachRotationSignals(lineup: ConfirmedLineup | null | undefined, previous?: StatsData['coachRotationAudit']['latestTeamStates'][string]) {
  if (!lineup?.confirmed || lineup.players.length < 8 || !previous) return null;
  const currentPlayers = new Set(lineup.players.map(player => player.playerId));
  const previousPlayers = new Set(previous.players);
  const currentSlots = Object.fromEntries(lineup.players.map(player => [String(player.playerId), player.battingOrder]));
  const overlap = lineup.players.filter(player => previousPlayers.has(player.playerId));
  const orderMoves = overlap.flatMap(player => previous.slots[String(player.playerId)] == null ? [] : [Math.abs(player.battingOrder - previous.slots[String(player.playerId)])]);
  const currentTop4 = new Set(lineup.players.filter(player => player.battingOrder <= 4).map(player => player.playerId));
  const incomingIds = [...currentPlayers].filter(player => !previousPlayers.has(player));
  const outgoingIds = [...previousPlayers].filter(player => !currentPlayers.has(player));
  const changes = incomingIds.length;
  const top4Changed = [...currentTop4].filter(player => !previous.top4.includes(player)).length;
  const prevLoss = Number(previous.win === 0);
  const incoming = incomingIds.map(id => ({ id, name: lineup.players.find(p => p.playerId === id)?.name, battingOrder: currentSlots[String(id)] }));
  const outgoing = outgoingIds.map(id => ({ id, name: undefined as string | undefined, battingOrder: previous.slots[String(id)] }));
  return {
    prevLoss,
    changes,
    orderMove: orderMoves.length ? orderMoves.reduce((sum, value) => sum + value, 0) / orderMoves.length : 0,
    top4Changed,
    lossRotation: prevLoss * changes,
    lossOrderMove: prevLoss * (orderMoves.length ? orderMoves.reduce((sum, value) => sum + value, 0) / orderMoves.length : 0),
    lossTop4Changed: prevLoss * top4Changed,
    lineupSize: Object.keys(currentSlots).length,
    incoming,
    outgoing,
  };
}

function rotationQualitySignals(
  lineup: ConfirmedLineup | null | undefined,
  previous: StatsData['coachRotationAudit']['latestTeamStates'][string] | undefined,
  ratings: Record<string, number> | undefined,
) {
  if (!lineup?.confirmed || lineup.players.length < 8 || !previous || !ratings) return null;
  const rated = lineup.players.filter(player => ratings[String(player.playerId)] != null);
  if (rated.length < 7) return null;
  const rating = (playerId: number) => ratings[String(playerId)] ?? 0.35;
  const currentPlayers = new Set(lineup.players.map(player => player.playerId));
  const previousPlayers = new Set(previous.players);
  const incoming = [...currentPlayers].filter(player => !previousPlayers.has(player));
  const outgoing = [...previousPlayers].filter(player => !currentPlayers.has(player));
  const mean = (players: number[], fallback = 0.35) => players.length ? players.reduce((sum, player) => sum + rating(player), 0) / players.length : fallback;
  const currentMean = mean([...currentPlayers]);
  const previousMean = mean([...previousPlayers]);
  const incomingMean = mean(incoming, currentMean);
  const outgoingMean = mean(outgoing, previousMean);
  const previousWeight = [...previousPlayers].reduce((sum, player) => sum + Math.max(rating(player), 0.01), 0);
  const retainedWeight = [...currentPlayers].filter(player => previousPlayers.has(player)).reduce((sum, player) => sum + Math.max(rating(player), 0.01), 0);
  const currentTop4 = lineup.players.filter(player => player.battingOrder <= 4).map(player => player.playerId);
  const qualityDelta = currentMean - previousMean;
  const replacementDelta = incoming.length || outgoing.length ? incomingMean - outgoingMean : 0;
  const retentionQuality = previousWeight ? retainedWeight / previousWeight : 1;
  const top4QualityDelta = mean(currentTop4, currentMean) - mean(previous.top4, previousMean);
  const loss = Number(previous.win === 0);
  return {
    qualityDelta,
    replacementDelta,
    retentionQuality,
    top4QualityDelta,
    lossQualityDelta: loss * qualityDelta,
    lossReplacementDelta: loss * replacementDelta,
    lossRetentionQuality: loss * retentionQuality,
    lossTop4QualityDelta: loss * top4QualityDelta,
    ratedPlayers: rated.length,
  };
}

function qualityDirection(value?: number | null) {
  if (value == null) return 'sin valoración';
  if (value > .003) return 'mejoró';
  if (value < -.003) return 'bajó';
  return 'similar';
}

function fatigueDirection(value?: number | null) {
  if (value == null) return 'sin dato';
  if (value <= 1.2) return 'carga baja';
  if (value <= 2.0) return 'carga media';
  return 'carga alta';
}

function lineupFatigueSignals(
  lineup: ConfirmedLineup | null | undefined,
  workloads: StatsData['lineupFatigueAudit']['latestPlayerWorkloads'][string] | undefined,
  gameDate: string,
) {
  if (!lineup?.confirmed || lineup.players.length < 8 || !workloads) return null;
  const currentTime = new Date(gameDate).getTime();
  const available = lineup.players.flatMap(player => {
    const state = workloads[String(player.playerId)];
    if (!state) return [];
    const events = state.events.map(([playedAt, started, pa]) => ({ at: new Date(playedAt).getTime(), started, pa }));
    const within = (event: { at: number }, hours: number) => currentTime >= event.at && currentTime - event.at <= hours * 3_600_000;
    const recent3 = events.filter(event => within(event, 72));
    const recent7 = events.filter(event => within(event, 168));
    const last = events.at(-1);
    const consecutive = state.streak ?? 0;
    const quality = Math.max(state.quality ?? .35, .01);
    const startsLast3Days = recent3.reduce((sum, event) => sum + event.started, 0);
    const paLast3Days = recent3.reduce((sum, event) => sum + event.pa, 0);
    const paLast7Days = recent7.reduce((sum, event) => sum + event.pa, 0);
    const shortRest = Number(Boolean(last && within(last, 30)));
    return [{
      startsLast3Days,
      paLast3Days,
      paLast7Days,
      shortRest,
      quality,
      fatigue: .32 * consecutive + .28 * startsLast3Days + .20 * (paLast3Days / 4.3) + .12 * (paLast7Days / 12.9) + .65 * shortRest,
    }];
  });
  if (available.length < 7) return null;
  const average = (key: 'startsLast3Days' | 'paLast3Days' | 'paLast7Days') => (
    available.reduce((sum, player) => sum + player[key], 0) / available.length
  );
  const qualityWeight = available.reduce((sum, player) => sum + player.quality, 0);
  return {
    startsLast3Days: average('startsLast3Days'),
    paLast3Days: average('paLast3Days'),
    paLast7Days: average('paLast7Days'),
    shortRestCount: available.reduce((sum, player) => sum + player.shortRest, 0),
    weightedFatigue: available.reduce((sum, player) => sum + player.fatigue * player.quality, 0) / qualityWeight,
    trackedPlayers: available.length,
  };
}

function buildFeatures(game: ScheduleGame, stats: StatsData, odds?: CurrentOddsGame) {
  const homeCode = alias(game.home.team);
  const awayCode = alias(game.away.team);
  const home = stats.teams[homeCode];
  const away = stats.teams[awayCode];
  const homeContext = stats.todayContext[`${homeCode}|${awayCode}`] ?? { code: 'none', label: 'Sin serie exacta de 3 previa' };
  const awayContext = stats.todayContext[`${awayCode}|${homeCode}`] ?? { code: 'none', label: 'Sin serie exacta de 3 previa' };
  const hc = homeContext.code;
  const ac = awayContext.code;
  const homeStrength = home ? (home.wins + 10) / (home.games + 20) : 0.5;
  const awayStrength = away ? (away.wins + 10) / (away.games + 20) : 0.5;
  const homeRecent = home ? (home.last10Wins + 2.5) / 15 : 0.5;
  const awayRecent = away ? (away.last10Wins + 2.5) / 15 : 0.5;
  const market = marketSnapshot(odds);
  const homeRest = daysBetween(game.officialDate, home?.lastGameDate);
  const awayRest = daysBetween(game.officialDate, away?.lastGameDate);
  const homeStars = stats.bestPlayersAudit.latestTeamSignals?.[homeCode]?.signals ?? {};
  const awayStars = stats.bestPlayersAudit.latestTeamSignals?.[awayCode]?.signals ?? {};
  const starEdge = (key: string) => (homeStars[key] ?? 0) - (awayStars[key] ?? 0);
  const homeCoach = coachRotationSignals(game.home.lineup, stats.coachRotationAudit.latestTeamStates?.[homeCode]);
  const awayCoach = coachRotationSignals(game.away.lineup, stats.coachRotationAudit.latestTeamStates?.[awayCode]);
  const coachRotationAvailable = Boolean(homeCoach && awayCoach);
  const homeRotationQuality = rotationQualitySignals(
    game.home.lineup,
    stats.coachRotationAudit.latestTeamStates?.[homeCode],
    stats.rotationQualityAudit.latestPlayerRatings?.[homeCode],
  );
  const awayRotationQuality = rotationQualitySignals(
    game.away.lineup,
    stats.coachRotationAudit.latestTeamStates?.[awayCode],
    stats.rotationQualityAudit.latestPlayerRatings?.[awayCode],
  );
  const rotationQualityAvailable = Boolean(homeRotationQuality && awayRotationQuality);
  const qualityEdge = (key: keyof NonNullable<typeof homeRotationQuality>) => (
    (homeRotationQuality?.[key] ?? 0) - (awayRotationQuality?.[key] ?? 0)
  );
  const homeLineupFatigue = lineupFatigueSignals(
    game.home.lineup, stats.lineupFatigueAudit.latestPlayerWorkloads?.[homeCode], game.gameDate,
  );
  const awayLineupFatigue = lineupFatigueSignals(
    game.away.lineup, stats.lineupFatigueAudit.latestPlayerWorkloads?.[awayCode], game.gameDate,
  );
  const lineupFatigueAvailable = Boolean(homeLineupFatigue && awayLineupFatigue);
  const fatigueEdge = (key: 'startsLast3Days' | 'paLast3Days' | 'paLast7Days' | 'shortRestCount') => (
    (homeLineupFatigue?.[key] ?? 0) - (awayLineupFatigue?.[key] ?? 0)
  );
  const homeOpponentForm = stats.opponentFormAudit.latestTeamSignals?.[homeCode];
  const awayOpponentForm = stats.opponentFormAudit.latestTeamSignals?.[awayCode];
  return {
    home,
    away,
    homeContext,
    awayContext,
    homeCoach,
    awayCoach,
    coachRotationAvailable,
    homeRotationQuality,
    awayRotationQuality,
    rotationQualityAvailable,
    homeLineupFatigue,
    awayLineupFatigue,
    lineupFatigueAvailable,
    values: {
      strength_diff: homeStrength - awayStrength,
      recent_diff: homeRecent - awayRecent,
      run_diff_diff: Math.max(-3, Math.min(3, (home?.runDiffPerGame ?? 0) - (away?.runDiffPerGame ?? 0))),
      home_sweep: Number(hc === 'sweep'),
      away_sweep: Number(ac === 'sweep'),
      home_swept: Number(hc === 'swept'),
      away_swept: Number(ac === 'swept'),
      home_lost_1_2: Number(hc === 'lost_1_2'),
      away_lost_1_2: Number(ac === 'lost_1_2'),
      both_sweep: Number(hc === 'sweep' && ac === 'sweep'),
      both_swept: Number(hc === 'swept' && ac === 'swept'),
      home_swept_vs_away_lost_1_2: Number(hc === 'swept' && ac === 'lost_1_2'),
      away_swept_vs_home_lost_1_2: Number(ac === 'swept' && hc === 'lost_1_2'),
      market_logit_p_home: market.logit,
      market_n_providers: market.providers,
      market_p_home_std: market.std,
      days_since_last_game_diff: homeRest == null || awayRest == null ? null : homeRest - awayRest,
      edge_prev_top5_ge4_win: starEdge('prev_top5_ge4_win'),
      edge_prev_top5_ge4_loss: starEdge('prev_top5_ge4_loss'),
      edge_prev_core8_ge7_win: starEdge('prev_core8_ge7_win'),
      edge_prev_core8_ge7_loss: starEdge('prev_core8_ge7_loss'),
      edge_prev_star_pa_ge50_win: starEdge('prev_star_pa_ge50_win'),
      edge_prev_star_pa_ge50_loss: starEdge('prev_star_pa_ge50_loss'),
      edge_prev_star_weight_ge80_win: starEdge('prev_star_weight_ge80_win'),
      edge_prev_star_weight_ge80_loss: starEdge('prev_star_weight_ge80_loss'),
      home_prevLoss: homeCoach?.prevLoss ?? null,
      away_prevLoss: awayCoach?.prevLoss ?? null,
      home_changes: homeCoach?.changes ?? null,
      away_changes: awayCoach?.changes ?? null,
      home_orderMove: homeCoach?.orderMove ?? null,
      away_orderMove: awayCoach?.orderMove ?? null,
      home_top4Changed: homeCoach?.top4Changed ?? null,
      away_top4Changed: awayCoach?.top4Changed ?? null,
      home_lossRotation: homeCoach?.lossRotation ?? null,
      away_lossRotation: awayCoach?.lossRotation ?? null,
      home_lossOrderMove: homeCoach?.lossOrderMove ?? null,
      away_lossOrderMove: awayCoach?.lossOrderMove ?? null,
      home_lossTop4Changed: homeCoach?.lossTop4Changed ?? null,
      away_lossTop4Changed: awayCoach?.lossTop4Changed ?? null,
      edge_qualityDelta: rotationQualityAvailable ? qualityEdge('qualityDelta') : null,
      edge_replacementDelta: rotationQualityAvailable ? qualityEdge('replacementDelta') : null,
      edge_retentionQuality: rotationQualityAvailable ? qualityEdge('retentionQuality') : null,
      edge_top4QualityDelta: rotationQualityAvailable ? qualityEdge('top4QualityDelta') : null,
      edge_lossQualityDelta: rotationQualityAvailable ? qualityEdge('lossQualityDelta') : null,
      edge_lossReplacementDelta: rotationQualityAvailable ? qualityEdge('lossReplacementDelta') : null,
      edge_lossRetentionQuality: rotationQualityAvailable ? qualityEdge('lossRetentionQuality') : null,
      edge_lossTop4QualityDelta: rotationQualityAvailable ? qualityEdge('lossTop4QualityDelta') : null,
      edge_startsLast3Days: lineupFatigueAvailable ? fatigueEdge('startsLast3Days') : null,
      edge_paLast3Days: lineupFatigueAvailable ? fatigueEdge('paLast3Days') : null,
      edge_paLast7Days: lineupFatigueAvailable ? fatigueEdge('paLast7Days') : null,
      edge_shortRestCount: lineupFatigueAvailable ? fatigueEdge('shortRestCount') : null,
      edge_opponentStrengthL10: (homeOpponentForm?.opponentStrengthL10 ?? 0.5) - (awayOpponentForm?.opponentStrengthL10 ?? 0.5),
      edge_opponentStrengthL20: (homeOpponentForm?.opponentStrengthL20 ?? 0.5) - (awayOpponentForm?.opponentStrengthL20 ?? 0.5),
    },
  };
}

function estimateGame(game: ScheduleGame, stats: StatsData, active: Record<FactorKey, boolean>, odds?: CurrentOddsGame) {
  const prepared = buildFeatures(game, stats, odds);
  const mask = FACTORS.reduce((value, factor, index) => {
    const available = factor.key === 'coachRotationTest'
      ? prepared.coachRotationAvailable
      : factor.key === 'rotationQualityTest'
        ? prepared.rotationQualityAvailable
        : factor.key === 'lineupFatigueTest'
          ? prepared.lineupFatigueAvailable
          : factor.key === 'opponentFormTest'
            ? active.rotationQualityTest && active.lineupFatigueTest
              && prepared.rotationQualityAvailable && prepared.lineupFatigueAvailable
          : true;
    return value + (active[factor.key] && available ? 1 << index : 0);
  }, 0);
  const model = stats.model.modelsByMask[String(mask)];
  if (!model) return { ...prepared, homeProbability: 0.5 };
  if (model.kind === 'constant') return { ...prepared, homeProbability: model.probability ?? 0.5 };
  const missing: boolean[] = [];
  const imputed = model.inputFeatures.map((feature, index) => {
    const raw = prepared.values[feature as keyof typeof prepared.values];
    const isMissing = raw == null || !Number.isFinite(raw);
    missing.push(isMissing);
    return isMissing ? (model.imputeValues?.[index] ?? 0) : raw;
  });
  const transformed = [...imputed, ...(model.indicatorFeatures ?? []).map(index => Number(missing[index]))];
  const logit = transformed.reduce((sum, value, index) => {
    const z = (value - (model.means?.[index] ?? 0)) / (model.scales?.[index] || 1);
    return sum + z * (model.coefficients?.[index] ?? 0);
  }, model.intercept ?? 0);
  return { ...prepared, homeProbability: logistic(logit) };
}

function smoothedSweepRate(base: SweepRate, sample?: SweepRate, strength = 18) {
  const baseRate = base.rate ?? 0.5;
  if (!sample?.games) return baseRate;
  return (sample.sweeps + strength * baseRate) / (sample.games + strength);
}

function conditionalSweepRate(stats: StatsData, stage: 'afterGame1' | 'afterGame2', team: string, opponent: string, location: 'Casa' | 'Visitante') {
  const estimator = stats.seriesSweepAudit.estimator[stage];
  const base = estimator.byLocation[location] ?? estimator.overall;
  const teamRate = smoothedSweepRate(base, estimator.byTeam[alias(team)]);
  const opponentRate = smoothedSweepRate(base, estimator.byOpponent[alias(opponent)]);
  return 0.55 * teamRate + 0.45 * opponentRate;
}

type SweepForecast = {
  game: ScheduleGame;
  stage: number;
  sides: { team: string; name: string; probability: number; consecutiveAttempt: boolean }[];
};

function buildSweepForecasts(
  schedule: ScheduleData,
  previousGames: ScheduleGame[],
  stats: StatsData,
  active: Record<FactorKey, boolean>,
  odds: OddsData | null,
): SweepForecast[] {
  const forecasts: SweepForecast[] = [];
  for (const game of schedule.games) {
    if (game.gamesInSeries !== 3 || !game.seriesGameNumber || game.seriesGameNumber < 1 || game.seriesGameNumber > 3) continue;
    const stage = game.seriesGameNumber;
    const earlier = previousGames
      .filter(item => item.abstractState === 'Final'
        && item.gamesInSeries === 3
        && item.away.team === game.away.team
        && item.home.team === game.home.team
        && (item.seriesGameNumber ?? 0) < stage)
      .sort((a, b) => (a.seriesGameNumber ?? 0) - (b.seriesGameNumber ?? 0));
    if (stage > 1 && earlier.length < stage - 1) continue;
    const estimate = estimateGame(game, stats, active, findCurrentOdds(game, odds));
    const currentProbabilities: Record<string, number> = {
      [game.away.team]: 1 - estimate.homeProbability,
      [game.home.team]: estimate.homeProbability,
    };
    const sides = [game.away, game.home].flatMap(side => {
      const wonEveryPrevious = earlier.every(item => {
        const sideWasAway = item.away.team === side.team;
        const sideScore = sideWasAway ? item.away.score : item.home.score;
        const opponentScore = sideWasAway ? item.home.score : item.away.score;
        return sideScore != null && opponentScore != null && sideScore > opponentScore;
      });
      if (stage > 1 && !wonEveryPrevious) return [];
      const opponent = side.team === game.away.team ? game.home.team : game.away.team;
      const location = side.team === game.home.team ? 'Casa' : 'Visitante';
      const currentWinProbability = currentProbabilities[side.team] ?? .5;
      const probability = stage === 1
        ? currentWinProbability * conditionalSweepRate(stats, 'afterGame1', side.team, opponent, location)
        : stage === 2
          ? currentWinProbability * conditionalSweepRate(stats, 'afterGame2', side.team, opponent, location)
          : currentWinProbability;
      return [{
        team: side.team,
        name: side.name,
        probability,
        consecutiveAttempt: Boolean(stats.seriesSweepAudit.latestCompleted[alias(side.team)]?.swept),
      }];
    });
    if (sides.length) forecasts.push({ game, stage, sides });
  }
  return forecasts;
}

function SourceBadge({ schedule }: { schedule: ScheduleData }) {
  const live = schedule.sourceState.includes('LIVE');
  return <span className={`source-badge ${live ? 'source-live' : 'source-fallback'}`}><span className="source-dot" /> {live ? 'CALENDARIO EN VIVO + GUARDADO' : 'RESPALDO GUARDADO'}</span>;
}

function GameCard({ game, stats, active, odds, pickNumber, telegramConfigured, historical = false, frozenPrediction, comparisonMasks, strikecastPick, valuePick, oddsHistory, evidenceProfiles = [], evidenceSignals = [], playerPerformance = null, historicalSnapshot }: { game: ScheduleGame; stats: StatsData; active: Record<FactorKey, boolean>; odds?: CurrentOddsGame; pickNumber: number; telegramConfigured: boolean; historical?: boolean; frozenPrediction?: FrozenGamePrediction; comparisonMasks?: ComparisonMaskSet | null; strikecastPick?: StrikecastPick; valuePick?: ValuePick; oddsHistory?: OddsHistoryPayload | null; evidenceProfiles?: CardEvidenceProfile[]; evidenceSignals?: CardEvidenceSignal[]; playerPerformance?: PlayerPerformancePayload | null; historicalSnapshot?: HistorySnapshot }) {
  const estimate = estimateGame(game, stats, active, odds);
  const homeP = historical ? (frozenPrediction?.homeProbability ?? 0.5) : estimate.homeProbability;
  const awayP = 1 - homeP;
  const favorite = homeP >= 0.5 ? game.home : game.away;
  const favoriteProbability = Math.max(homeP, awayP);
  const start = gameTime(game.gameDate);
  const played = game.abstractState === 'Final' || game.abstractState === 'Live';
  const gap = Math.abs(homeP - 0.5);
  const confidence = gap >= 0.1 ? 'Señal clara' : gap >= 0.05 ? 'Ventaja moderada' : 'Partido cerrado';
  const lineupAvailable = estimate.coachRotationAvailable || estimate.rotationQualityAvailable || estimate.lineupFatigueAvailable;
  const [modalOpen, setModalOpen] = useState(false);
  const [performanceOpen, setPerformanceOpen] = useState(false);
  const [selectedSide, setSelectedSide] = useState<'away' | 'home'>(homeP >= 0.5 ? 'home' : 'away');
  const [numberInput, setNumberInput] = useState(String(pickNumber));
  const [sending, setSending] = useState(false);
  const [feedback, setFeedback] = useState<{ kind: 'success' | 'error'; text: string } | null>(null);
  const [pickType, setPickType] = useState<PickType>('ML');
  const [isLive, setIsLive] = useState(false);
  const [manualOddsInput, setManualOddsInput] = useState('');
  const [totalPointInput, setTotalPointInput] = useState('8.5');
  const pickTypeMeta = PICK_TYPE_OPTIONS.find(o => o.value === pickType)!;
  const selectedTeam = selectedSide === 'home' ? game.home.name : game.away.name;
  const marketOdds = selectedSide === 'home' ? odds?.bestHome : odds?.bestAway;
  // Momio efectivo: para ML puro pre-llena con mercado, pero el usuario puede sobrescribir.
  // Para el resto de tipos no hay mercado en la data, así que es 100% manual.
  const manualDecimal = manualOddsInput.trim() ? Number(manualOddsInput.replace(',', '.')) : NaN;
  const effectiveDecimal: number | undefined = (() => {
    if (Number.isFinite(manualDecimal) && manualDecimal >= 1.01) return manualDecimal;
    if (pickType === 'ML' && marketOdds?.decimal) return marketOdds.decimal;
    return undefined;
  })();
  const effectiveBook: string | undefined = (() => {
    if (Number.isFinite(manualDecimal) && manualDecimal >= 1.01) return 'Manual';
    if (pickType === 'ML' && marketOdds?.book) return marketOdds.book;
    return undefined;
  })();
  const numericPick = Number(numberInput);
  const preview = telegramPreview(numericPick, game, pickLineText(pickType, selectedTeam, totalPointInput), effectiveDecimal, isLive);

  useEffect(() => {
    if (!modalOpen) return;
    const previousOverflow = document.body.style.overflow;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setModalOpen(false);
    };
    document.body.style.overflow = 'hidden';
    document.addEventListener('keydown', closeOnEscape);
    return () => {
      document.body.style.overflow = previousOverflow;
      document.removeEventListener('keydown', closeOnEscape);
    };
  }, [modalOpen]);

  const openTelegram = () => {
    setSelectedSide(homeP >= 0.5 ? 'home' : 'away');
    setNumberInput(String(pickNumber));
    setPickType('ML');
    setIsLive(false);
    setManualOddsInput('');
    setTotalPointInput('8.5');
    setFeedback(null);
    setModalOpen(true);
  };

  const canSend = (() => {
    if (!Number.isInteger(numericPick) || numericPick < 1) return false;
    if (!effectiveDecimal || effectiveDecimal < 1.01) return false;
    if (pickTypeMeta.needsTotal) {
      const total = Number(totalPointInput.replace(',', '.'));
      if (!Number.isFinite(total) || total <= 0 || total > 30) return false;
    }
    return true;
  })();

  const sendPick = async () => {
    if (!canSend || !effectiveDecimal) return;
    setSending(true);
    setFeedback(null);
    try {
      const response = await fetch('/api/telegram', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          pickNumber: numericPick,
          awayTeam: game.away.name,
          homeTeam: game.home.name,
          gameDate: game.gameDate,
          time: start,
          pickTeam: pickTypeMeta.needsTeam ? selectedTeam : undefined,
          decimalOdds: effectiveDecimal,
          pickType,
          live: isLive,
          totalPoint: pickTypeMeta.needsTotal ? Number(totalPointInput.replace(',', '.')) : undefined,
        }),
      });
      const result = await response.json() as { sent?: boolean; error?: string };
      if (!response.ok || !result.sent) throw new Error(result.error ?? 'No se pudo enviar el pick.');
      setFeedback({ kind: 'success', text: 'Pick publicado correctamente en Telegram.' });
    } catch (error) {
      setFeedback({ kind: 'error', text: error instanceof Error ? error.message : 'No se pudo enviar el pick.' });
    } finally {
      setSending(false);
    }
  };

  return (
    <article className="game-card game-card-redesigned">
      <GameComparisonLayout game={game} stats={stats} active={active} odds={odds} estimate={estimate} homeP={homeP} awayP={awayP} favorite={favorite} favoriteProbability={favoriteProbability} historical={historical} comparisonMasks={comparisonMasks} strikecastPick={strikecastPick} valuePick={valuePick} oddsHistory={oddsHistory} evidenceProfiles={evidenceProfiles} evidenceSignals={evidenceSignals} playerPerformance={playerPerformance} historicalSnapshot={historicalSnapshot} telegramConfigured={telegramConfigured} onOpenTelegram={openTelegram} onOpenPerformance={() => setPerformanceOpen(true)} />
      <div className="game-meta"><span>{start} · {game.venue || 'Sede por confirmar'}</span><span className={game.abstractState === 'Live' ? 'status-live' : ''}>{game.status}</span></div>
      <div className="game-matchup">
        <div className="matchup-team"><TeamLogo team={game.away.team} name={game.away.name} className="matchup-logo" /><div><p className="team-code">{game.away.team}</p><p className="team-name">{game.away.name}</p><p className="team-record">{historical ? 'Resultado histórico' : estimate.away ? `${estimate.away.wins}-${estimate.away.losses} · L10 ${estimate.away.last10Wins}-${estimate.away.last10Losses}` : 'Sin registro local'}</p></div></div>
        <div className="score-box">{historical && game.away.score != null && game.home.score != null ? `${game.away.score}–${game.home.score}` : played && game.away.score != null ? game.away.score : 'vs'}</div>
        <div className="matchup-team home-team"><TeamLogo team={game.home.team} name={game.home.name} className="matchup-logo" /><div><p className="team-code">{game.home.team}</p><p className="team-name">{game.home.name}</p><p className="team-record">{historical ? 'Resultado histórico' : estimate.home ? `${estimate.home.wins}-${estimate.home.losses} · L10 ${estimate.home.last10Wins}-${estimate.home.last10Losses}` : 'Sin registro local'}</p></div></div>
      </div>
      <div className="pitchers"><span>{game.away.pitcher || 'Abridor por confirmar'}</span><span>{game.home.pitcher || 'Abridor por confirmar'}</span></div>
      {historical && !frozenPrediction ? <div className="historical-no-prediction"><ShieldCheck size={15} /><span>Este juego no tiene una predicción walk-forward congelada disponible.</span></div> : <>
        <div className="probability-head"><div><span className="eyebrow">{historical ? 'Predicción congelada WF' : 'Estimación StatsMLB'}</span><strong className="logo-label"><TeamLogo team={favorite.team} name={favorite.name} className="logo-xs" />{favorite.team} {pct(favoriteProbability)}</strong></div><span className="confidence">{confidence}</span></div>
        <div className="probability-bar" aria-label={`Probabilidad ${game.away.team} ${pct(awayP)}, ${game.home.team} ${pct(homeP)}`}><div className="away-bar" style={{ width: `${awayP * 100}%` }} /><div className="home-bar" style={{ width: `${homeP * 100}%` }} /></div>
        <div className="probability-labels"><span><TeamLogo team={game.away.team} name={game.away.name} className="logo-xxs" />{game.away.team} {pct(awayP)}</span><span><TeamLogo team={game.home.team} name={game.home.name} className="logo-xxs" />{game.home.team} {pct(homeP)}</span></div>
        {!historical && <div className="game-insight-grid" aria-label="Contexto real del partido">
          <div><span>FORMA L10</span><strong>{estimate.away ? `${estimate.away.last10Wins}-${estimate.away.last10Losses}` : '-'} · {estimate.home ? `${estimate.home.last10Wins}-${estimate.home.last10Losses}` : '-'}</strong><small>{game.away.team} · {game.home.team}</small></div>
          <div><span>MERCADO</span><strong>{odds?.bestAway?.decimal.toFixed(2) ?? '-'} · {odds?.bestHome?.decimal.toFixed(2) ?? '-'}</strong><small>{odds ? 'mejor precio disponible' : 'sin momios enlazados'}</small></div>
          <div><span>SERIE PREVIA</span><strong>{estimate.awayContext.code === 'none' ? 'Sin señal' : estimate.awayContext.label}</strong><small>{game.away.team} · {estimate.homeContext.code === 'none' ? 'sin señal' : estimate.homeContext.label}</small></div>
          <div><span>LINEUP</span><strong>{lineupAvailable ? 'Con cobertura' : 'Pendiente'}</strong><small>{lineupAvailable ? 'cambios, calidad y carga aplicados' : 'se conserva la combinación base'}</small></div>
        </div>}
      </>}
      {historical ? <div className="historical-result-row"><div><span>RESULTADO FINAL</span><strong>{game.away.score != null && game.home.score != null ? `${game.away.team} ${game.away.score}–${game.home.score} ${game.home.team}` : 'Marcador pendiente'}</strong></div>{frozenPrediction && <small><ShieldCheck size={13} /> Corte {frozenPrediction.foldMonth} · entrenado hasta {frozenPrediction.trainedThrough}</small>}</div> : <>
        <div className="odds-pick-row"><div><span>MEJOR MOMIO</span>{odds ? <strong className="odds-team-line"><span><TeamLogo team={game.away.team} name={game.away.name} className="logo-xxs" />{game.away.team} {odds.bestAway?.decimal.toFixed(2) ?? '—'}</span><i>·</i><span><TeamLogo team={game.home.team} name={game.home.name} className="logo-xxs" />{game.home.team} {odds.bestHome?.decimal.toFixed(2) ?? '—'}</span></strong> : <strong>Sin momios enlazados</strong>}</div><button type="button" onClick={openTelegram} title={played ? 'El partido ya inició; el pick se enviará de todos modos.' : undefined}><Send size={14} /> {telegramConfigured ? 'Enviar pick' : 'Preparar Telegram'}</button></div>
        <div className="context-row"><span title={estimate.awayContext.label}><TeamLogo team={game.away.team} name={game.away.name} className="logo-xxs" />{game.away.team}: {estimate.awayContext.code === 'none' ? 'sin señal de serie' : estimate.awayContext.label}</span><span title={estimate.homeContext.label}><TeamLogo team={game.home.team} name={game.home.name} className="logo-xxs" />{game.home.team}: {estimate.homeContext.code === 'none' ? 'sin señal de serie' : estimate.homeContext.label}</span></div>
        {(active.coachRotationTest || active.rotationQualityTest || active.lineupFatigueTest) && <div className="lineup-insights"><div className="lineup-insights-head"><span>CAMBIOS DE LINEUP</span><small>{lineupAvailable ? 'La señal disponible ya está incorporada en la estimación.' : 'Esperando ambas alineaciones confirmadas; se conserva la combinación base.'}</small></div>
          {active.coachRotationTest && <div className={`coach-live-signal ${estimate.coachRotationAvailable ? 'available' : 'pending'}`}><span>Rotación coach</span><strong>{estimate.coachRotationAvailable ? `${game.away.team} ${estimate.awayCoach?.changes ?? 0} cambios · ${game.home.team} ${estimate.homeCoach?.changes ?? 0} cambios` : 'Sin confirmación de ambas alineaciones'}</strong></div>}
          {active.rotationQualityTest && <div className={`coach-live-signal quality-live-signal ${estimate.rotationQualityAvailable ? 'available' : 'pending'}`}><span>Calidad rotación</span><strong>{estimate.rotationQualityAvailable ? `${game.away.team} ${qualityDirection(estimate.awayRotationQuality?.qualityDelta)} · ${game.home.team} ${qualityDirection(estimate.homeRotationQuality?.qualityDelta)}` : 'Sin valoraciones suficientes de ambos lineups'}</strong></div>}
          {active.lineupFatigueTest && <div className={`coach-live-signal ${estimate.lineupFatigueAvailable ? 'available' : 'pending'}`}><span>Carga 72 h</span><strong>{estimate.lineupFatigueAvailable ? `${game.away.team} ${fatigueDirection(estimate.awayLineupFatigue?.weightedFatigue)} (${estimate.awayLineupFatigue?.paLast3Days.toFixed(1)} PA/bat) · ${game.home.team} ${fatigueDirection(estimate.homeLineupFatigue?.weightedFatigue)} (${estimate.homeLineupFatigue?.paLast3Days.toFixed(1)} PA/bat)` : 'Sin historial suficiente de ambos lineups'}</strong></div>}
        </div>}
      </>}
      {modalOpen && typeof document !== 'undefined' && createPortal(<div className="telegram-modal-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) setModalOpen(false); }}>
        <section className="telegram-modal" role="dialog" aria-modal="true" aria-labelledby={`telegram-title-${game.gamePk}`}>
          <header><div><span><Send size={14} /> Telegram</span><h3 id={`telegram-title-${game.gamePk}`}>Preparar pick MLB</h3></div><button type="button" onClick={() => setModalOpen(false)} aria-label="Cerrar"><X size={18} /></button></header>
          <div className="telegram-form-row">
            <label>Número de pick<input type="number" min="1" max="999" value={numberInput} onChange={(event) => setNumberInput(event.target.value)} /></label>
            <label>Tipo de pick
              <select value={pickType} onChange={(event) => { setPickType(event.target.value as PickType); setManualOddsInput(''); }}>
                {PICK_TYPE_OPTIONS.map(opt => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
              </select>
            </label>
            <label className="telegram-live-toggle"><input type="checkbox" checked={isLive} onChange={(event) => setIsLive(event.target.checked)} /> <span>🔴 En vivo</span></label>
            <div><span>Hora CDMX</span><strong>{start}</strong></div>
          </div>
          {pickTypeMeta.needsTeam && (
            <fieldset><legend>Selecciona el equipo</legend><div className="telegram-side-grid">
              <button type="button" className={selectedSide === 'away' ? 'selected' : ''} onClick={() => setSelectedSide('away')}><TeamLogo team={game.away.team} name={game.away.name} className="telegram-team-logo" /><span>{game.away.name}</span>{pickType === 'ML' ? <><strong>{odds?.bestAway?.decimal.toFixed(2) ?? '—'}</strong><small>{odds?.bestAway?.book ?? 'Sin momio'}</small></> : <><strong>—</strong><small>momio manual</small></>}</button>
              <button type="button" className={selectedSide === 'home' ? 'selected' : ''} onClick={() => setSelectedSide('home')}><TeamLogo team={game.home.team} name={game.home.name} className="telegram-team-logo" /><span>{game.home.name}</span>{pickType === 'ML' ? <><strong>{odds?.bestHome?.decimal.toFixed(2) ?? '—'}</strong><small>{odds?.bestHome?.book ?? 'Sin momio'}</small></> : <><strong>—</strong><small>momio manual</small></>}</button>
            </div></fieldset>
          )}
          <div className="telegram-form-row">
            {pickTypeMeta.needsTotal && (
              <label>Total (line)<input type="number" step="0.5" min="0.5" max="30" value={totalPointInput} onChange={(event) => setTotalPointInput(event.target.value)} placeholder="8.5" /></label>
            )}
            <label>Momio decimal{pickType === 'ML' ? ' (opcional, sobrescribe mercado)' : ''}<input type="text" inputMode="decimal" value={manualOddsInput} onChange={(event) => setManualOddsInput(event.target.value)} placeholder={pickType === 'ML' && marketOdds?.decimal ? marketOdds.decimal.toFixed(2) : '1.85'} /></label>
          </div>
          <div className="telegram-preview"><span>VISTA PREVIA EXACTA</span><pre>{preview}</pre></div>
          <div className="telegram-provenance">
            <span className={effectiveBook === 'Manual' ? 'missing' : effectiveBook ? 'real' : 'missing'}>{effectiveBook === 'Manual' ? 'MOMIO MANUAL' : effectiveBook ? 'REAL DATA' : 'SIN MOMIO'}</span>
            <p>{effectiveBook === 'Manual' ? 'Momio ingresado manualmente. Verifica el precio con tu casa antes de mandar.' : effectiveBook ? `Precio de ${effectiveBook} · actualizado ${odds ? new Intl.DateTimeFormat('es-MX', { timeZone: 'America/Mexico_City', hour: '2-digit', minute: '2-digit' }).format(new Date(odds.lastUpdated)) : '—'}` : 'Ingresa el momio manualmente para poder mandar el pick.'}</p>
          </div>
          {!telegramConfigured && <p className="telegram-setup-warning">El envío quedará habilitado cuando conectes el bot y el canal.</p>}
          {feedback && <p className={`telegram-feedback ${feedback.kind}`}>{feedback.text}</p>}
          <footer><button type="button" className="telegram-cancel" onClick={() => setModalOpen(false)}>Cancelar</button><button type="button" className="telegram-send" disabled={!telegramConfigured || !canSend || sending || feedback?.kind === 'success'} onClick={() => void sendPick()}><Send size={15} /> {sending ? 'Enviando…' : feedback?.kind === 'success' ? 'Enviado' : 'Enviar a Telegram'}</button></footer>
        </section>
      </div>, document.body)}
      {performanceOpen && typeof document !== 'undefined' && createPortal(<PlayerPerformanceModal game={game} playerPerformance={playerPerformance} historicalSnapshot={historicalSnapshot} onClose={() => setPerformanceOpen(false)} />, document.body)}
    </article>
  );
}

function alignmentPickForGame(game: ScheduleGame, estimate: ReturnType<typeof estimateGame>, performanceLookup: Record<string, PlayerPerformanceEntry>) {
  const homeGames = (estimate.home?.wins ?? 0) + (estimate.home?.losses ?? 0);
  const awayGames = (estimate.away?.wins ?? 0) + (estimate.away?.losses ?? 0);
  const homePct = homeGames > 0 ? (estimate.home?.wins ?? 0) / homeGames : .5;
  const awayPct = awayGames > 0 ? (estimate.away?.wins ?? 0) / awayGames : .5;
  if (homePct >= .580 && awayPct >= .580) return null;
  if (!game.home.lineup?.confirmed || !game.away.lineup?.confirmed) return null;
  const scoreSide = (players: { playerId: number }[]) => {
    let tilt = 0;
    let counted = 0;
    for (const player of players) {
      const entry = performanceLookup[String(player.playerId)];
      if (!entry || entry.kind !== 'batter') continue;
      const tier = entry.talentTier ?? 'regular';
      const label = entry.label;
      counted += 1;
      if (tier === 'elite') {
        if (label === 'racha caliente') tilt += .030;
        else if (label === 'sólido') tilt += .012;
        else if (label === 'racha fría') tilt -= .025;
      } else if (tier === 'weak') {
        if (label === 'racha caliente') tilt -= .050;
        else if (label === 'sólido') tilt -= .015;
        else if (label === 'racha fría') tilt += .035;
      }
    }
    return { tilt, counted };
  };
  const home = scoreSide(game.home.lineup.players);
  const away = scoreSide(game.away.lineup.players);
  if (home.counted < 5 || away.counted < 5) return null;
  const rawProb = .5 + (home.tilt - away.tilt);
  const clampedProb = Math.max(.25, Math.min(.75, rawProb));
  return pickFromProbability(clampedProb, game);
}


function GameComparisonLayout({ game, stats, active, odds, estimate, homeP, awayP, favorite, favoriteProbability, historical, comparisonMasks, strikecastPick, valuePick, oddsHistory, evidenceProfiles, evidenceSignals, playerPerformance, historicalSnapshot, telegramConfigured, onOpenTelegram, onOpenPerformance }: { game: ScheduleGame; stats: StatsData; active: Record<FactorKey, boolean>; odds?: CurrentOddsGame; estimate: ReturnType<typeof estimateGame>; homeP: number; awayP: number; favorite: ScheduleGame['home']; favoriteProbability: number; historical: boolean; comparisonMasks?: ComparisonMaskSet | null; strikecastPick?: StrikecastPick; valuePick?: ValuePick; oddsHistory?: OddsHistoryPayload | null; evidenceProfiles: CardEvidenceProfile[]; evidenceSignals: CardEvidenceSignal[]; playerPerformance: PlayerPerformancePayload | null; historicalSnapshot?: HistorySnapshot; telegramConfigured: boolean; onOpenTelegram: () => void; onOpenPerformance: () => void }) {
  const modelPick = (mask?: number) => {
    if (mask == null) return null;
    const probability = estimateGame(game, stats, maskToActive(mask), odds).homeProbability;
    return pickFromProbability(probability, game);
  };
  const marketPick = marketPickFromCurrentOdds(game, odds, oddsHistory);
  const value = valuePick && valuePick.tier !== 'NEUTRO' && valuePick.pickCode && valuePick.pickProb != null ? { code: alias(valuePick.pickCode), prob: valuePick.pickProb } : null;
  const performanceLookup = historicalSnapshot?.players ?? playerPerformance?.players ?? {};
  const alignmentTile = alignmentPickForGame(game, estimate, performanceLookup);
  const tiles = [
    { label: 'BASE', pick: comparisonMasks ? modelPick(comparisonMasks.base) : pickFromProbability(homeP, game) },
    { label: 'DÍAS G.', pick: modelPick(comparisonMasks?.winningDays) },
    { label: 'JORNADA 15', pick: modelPick(comparisonMasks?.fifteenGames) },
    { label: 'STRIKECAST', pick: strikecastPick?.pick && strikecastPick.pHome != null ? { code: alias(strikecastPick.pick), prob: strikecastPick.pick === strikecastPick.home ? strikecastPick.pHome : 1 - strikecastPick.pHome } : null },
    { label: 'VALUE', pick: value },
    { label: 'MERCADO', pick: marketPick },
    { label: 'ALINEACIÓN', pick: alignmentTile },
  ];
  const present = tiles.filter((tile): tile is { label: string; pick: { code: string; prob: number } } => tile.pick != null);
  const agreement = present.filter(tile => tile.pick.code === favorite.team).length;
  const tier = agreement === present.length && present.length >= 5 && favoriteProbability >= .60 ? 'LOCK' : agreement >= Math.max(4, present.length - 1) ? 'FUERTE' : agreement >= 3 ? 'JUEGA' : 'PASAR';
  const strength = tier === 'LOCK' ? '●●●' : tier === 'FUERTE' ? '●●○' : tier === 'JUEGA' ? '●○○' : '○○○';
  const swept = estimate.awayContext.code === 'swept' ? game.away.team : estimate.homeContext.code === 'swept' ? game.home.team : null;
  const marketAlreadyPaid = marketPick?.code === favorite.team && marketPick.prob >= favoriteProbability + .025;
  const contextNote = swept
    ? `${swept} llega barrido 0-3${marketAlreadyPaid ? '; el mercado ya pagó parte del favoritismo.' : '.'}`
    : marketAlreadyPaid
      ? `El mercado confirma a ${favorite.team}; parte del favoritismo ya está incorporado en el precio.`
      : agreement === present.length && present.length >= 5
        ? `Consenso unánime y ${marketPick ? `mercado ${marketPick.code === favorite.team ? 'alineado' : 'en desacuerdo'}` : 'sin mercado disponible'}.`
        : `Consenso parcial: ${agreement}/${present.length} modelos favorecen a ${favorite.team}.`;
  const favoriteHome = favorite.team === game.home.team;
  const direction = favoriteHome ? 1 : -1;
  const favoriteMarketMove = odds?.marketMoveHomePp == null ? null : odds.marketMoveHomePp * direction;
  const l30Diff = ((estimate.home?.last10Pct ?? .5) - (estimate.away?.last10Pct ?? .5)) * direction;
  const runDiffDelta = ((estimate.home?.runDiffPerGame ?? 0) - (estimate.away?.runDiffPerGame ?? 0)) * direction;
  const restDelta = (estimate.values.days_since_last_game_diff ?? 0) * direction;
  const marketMoveDelta = favoriteMarketMove ?? 0;
  const marketPriceEdge = marketPick ? (marketPick.code === favorite.team ? marketPick.prob - .5 : .5 - marketPick.prob) : 0;
  const modelMinusMarket = marketPick ? (favoriteProbability - marketPick.prob) : 0;
  const liveSignals: Record<string, boolean> = {
    fav_home: favoriteHome,
    fav_road: !favoriteHome,
    market_agrees: Boolean(marketPick && marketPick.code === favorite.team),
    market_disagrees: Boolean(marketPick && marketPick.code !== favorite.team),
    model_edge_3: modelMinusMarket >= .03,
    model_edge_5: modelMinusMarket >= .05,
    market_prices_favorite: marketPriceEdge >= .10,
    favorite_l30_hot: l30Diff >= .08,
    favorite_l30_cold: l30Diff <= -.08,
    favorite_run_diff_strong: runDiffDelta >= .35,
    favorite_run_diff_weak: runDiffDelta <= -.35,
    favorite_rest_advantage: restDelta >= 1,
    favorite_short_rest: restDelta <= -1,
    market_move_toward_favorite: marketMoveDelta >= 2,
    market_move_against_favorite: marketMoveDelta <= -2,
    // starter / bullpen / elo / park / lineup: not exposed live; treat as unknown
    favorite_starter_edge: false,
    favorite_starter_weak: false,
    favorite_bullpen_fresher: false,
    favorite_bullpen_tired: false,
    favorite_elo_edge: false,
    favorite_park_boost: false,
    favorite_lineup_ops_edge: false,
    favorite_lineup_ops_deficit: false,
  };
  const triggeredSignals = evidenceSignals.filter(signal => liveSignals[signal.code] === true);
  triggeredSignals.sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));
  const alignmentSignals: CardEvidenceSignal[] = (() => {
    const homeGames = (estimate.home?.wins ?? 0) + (estimate.home?.losses ?? 0);
    const awayGames = (estimate.away?.wins ?? 0) + (estimate.away?.losses ?? 0);
    const homePct = homeGames > 0 ? (estimate.home?.wins ?? 0) / homeGames : .5;
    const awayPct = awayGames > 0 ? (estimate.away?.wins ?? 0) / awayGames : .5;
    if (homePct >= .580 && awayPct >= .580) return [];
    if (!game.home.lineup?.confirmed || !game.away.lineup?.confirmed) return [];
    const bucket = (players: { playerId: number }[]) => {
      const b = { elite_hot: 0, elite_cold: 0, weak_hot: 0, weak_cold: 0 };
      for (const p of players) {
        const e = performanceLookup[String(p.playerId)];
        if (!e || e.kind !== 'batter') continue;
        const tier = e.talentTier ?? 'regular';
        if (tier === 'elite' && e.label === 'racha caliente') b.elite_hot++;
        else if (tier === 'elite' && e.label === 'racha fría') b.elite_cold++;
        else if (tier === 'weak' && e.label === 'racha caliente') b.weak_hot++;
        else if (tier === 'weak' && e.label === 'racha fría') b.weak_cold++;
      }
      return b;
    };
    const fav = favoriteHome ? bucket(game.home.lineup.players) : bucket(game.away.lineup.players);
    const opp = favoriteHome ? bucket(game.away.lineup.players) : bucket(game.home.lineup.players);
    const signals: CardEvidenceSignal[] = [];
    if (fav.elite_hot >= 2) signals.push({ code: 'favorite_elite_bats_hot', label: `Bateadores élite del favorito calientes (${fav.elite_hot})`, direction: 'supports', sampleSize: 449, winRate: .548, baselineWinRate: .529, delta: .019, lower95: .5, upper95: .59 });
    if (fav.weak_hot >= 2) signals.push({ code: 'favorite_weak_bats_hot_regression', label: `Bateadores débiles del favorito calientes (${fav.weak_hot}) — riesgo de regresión`, direction: 'challenges', sampleSize: 477, winRate: .480, baselineWinRate: .529, delta: -.049, lower95: .43, upper95: .52 });
    if (opp.weak_hot >= 2) signals.push({ code: 'opponent_weak_bats_hot_regression', label: `Bateadores débiles del rival calientes (${opp.weak_hot}) — regresión a favor`, direction: 'supports', sampleSize: 550, winRate: .556, baselineWinRate: .529, delta: .063, lower95: .51, upper95: .60 });
    return signals;
  })();
  const combinedSignals = [...triggeredSignals, ...alignmentSignals].sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));
  const topSignals = combinedSignals.slice(0, 3);
  const supportsCount = topSignals.filter(s => s.direction === 'supports').length;
  const challengesCount = topSignals.filter(s => s.direction === 'challenges').length;
  const mixedSignal = supportsCount > 0 && challengesCount > 0;
  const evidenceEmpty = topSignals.length === 0;
  const bestOdds = favorite.team === game.home.team ? odds?.bestHome : odds?.bestAway;
  const line = (team: typeof game.home, coach: typeof estimate.homeCoach, quality: typeof estimate.homeRotationQuality, fatigue: typeof estimate.homeLineupFatigue) => <div className="lineup-table-row" key={team.team}><strong>{team.team}</strong><span>{coach ? `${coach.changes ?? 0} cambio${(coach.changes ?? 0) === 1 ? '' : 's'}` : 'sin datos'}</span><span>{coach ? (coach.top4Changed ? 'cambios en top 4' : 'top 4 intacto') : '-'}</span><span>{quality?.qualityDelta != null ? `${quality.qualityDelta >= 0 ? '+' : ''}${(quality.qualityDelta * 100).toFixed(1)}` : '-'}</span><span>{fatigue?.paLast3Days != null ? `${fatigue.paLast3Days.toFixed(1)} PA/bat` : '-'}</span></div>;
  return <div className="game-comparison-layout">
    <div className="game-comparison-matchup"><div className="game-meta"><span>{gameTime(game.gameDate)} · {game.venue || 'Sede por confirmar'}</span></div><div className="comparison-team"><TeamLogo team={game.away.team} name={game.away.name} className="matchup-logo" /><strong>{game.away.team}</strong><small>{estimate.away ? `${estimate.away.wins}-${estimate.away.losses}` : '-'}</small><b>{pct(awayP)}</b></div><div className="comparison-probability"><i style={{ width: `${homeP * 100}%` }} /></div><div className="comparison-team"><TeamLogo team={game.home.team} name={game.home.name} className="matchup-logo" /><strong>{game.home.team}</strong><small>{estimate.home ? `${estimate.home.wins}-${estimate.home.losses}` : '-'}</small><b>{pct(homeP)}</b></div><div className="comparison-pitchers"><span>{game.away.pitcher || 'Abridor por confirmar'}</span><span>{game.home.pitcher || 'Abridor por confirmar'}</span></div><LineupChangesDetails game={game} homeCoach={estimate.homeCoach} awayCoach={estimate.awayCoach} playerNames={stats.playerNames} historicalSnapshot={historicalSnapshot} /></div>
    <div className="game-comparison-models"><p><span>COMPARACIÓN</span><strong>{agreement}/{present.length} de acuerdo</strong></p><div>{tiles.map(tile => <article key={tile.label} className={tile.pick && tile.pick.code !== favorite.team ? 'model-disagrees' : ''}><span>{tile.label}</span><strong>{tile.pick?.code ?? '—'}</strong><b>{pct(tile.pick?.prob)}</b></article>)}</div><small>{contextNote}</small>{favoriteMarketMove != null && Math.abs(favoriteMarketMove) >= 2 && <small className="market-move-note">Mercado: {favoriteMarketMove > 0 ? `+${favoriteMarketMove.toFixed(1)} pp hacia ${favorite.team}` : `${favoriteMarketMove.toFixed(1)} pp contra ${favorite.team}`} desde la primera captura ({odds?.marketMoveBooks ?? 0} casas).</small>}<div className="game-evidence"><span>EVIDENCIA ENCONTRADA</span>{evidenceEmpty ? <p className="caution"><i />Sin señales históricas con muestra suficiente para este perfil de partido.</p> : topSignals.map(signal => <p key={signal.code} className={signal.direction === 'challenges' ? 'caution' : 'support'}><i />{signal.label}: {pct(signal.winRate)} de aciertos históricos (n={signal.sampleSize.toLocaleString('es-MX')}, {signal.delta >= 0 ? '+' : ''}{(signal.delta * 100).toFixed(1)} pp {signal.delta >= 0 ? 'sobre' : 'bajo'} la base).</p>)}{mixedSignal && <p className="caution"><i />Señal mixta: {supportsCount} factor{supportsCount === 1 ? '' : 'es'} respalda{supportsCount === 1 ? '' : 'n'} y {challengesCount} cuestiona{challengesCount === 1 ? '' : 'n'} el pick.</p>}</div></div>
    <div className="game-comparison-pick"><div><span className="comparison-tier">{tier}</span><b>{strength}</b></div><p><TeamLogo team={favorite.team} name={favorite.name} className="matchup-logo" /><strong>{favorite.team}</strong><small>{pct(favoriteProbability)} estimado</small></p><div className="comparison-best-odds"><span>Mejor momio</span><strong>{bestOdds?.decimal.toFixed(2) ?? '—'}</strong></div>{!historical && <button type="button" onClick={onOpenTelegram}><Send size={15} />{telegramConfigured ? 'Enviar pick' : 'Preparar Telegram'}</button>}<button type="button" className="performance-button" onClick={onOpenPerformance}><Activity size={15} />Rendimiento de jugadores</button></div>
    {!historical && <div className="game-comparison-lineup"><p><span>CAMBIOS DE LINEUP</span><small>{estimate.coachRotationAvailable || estimate.rotationQualityAvailable || estimate.lineupFatigueAvailable ? 'Rotación coach, calidad del lineup y carga de 72 h ya aplicadas a la estimación.' : 'Esperando ambas alineaciones confirmadas; se conserva la combinación base.'}</small></p><div className="lineup-table-head"><span>EQUIPO</span><span>TITULARES</span><span>ORDEN AL BATE</span><span>CALIDAD</span><span>CARGA 72 H</span></div>{line(game.away, estimate.awayCoach, estimate.awayRotationQuality, estimate.awayLineupFatigue)}{line(game.home, estimate.homeCoach, estimate.homeRotationQuality, estimate.homeLineupFatigue)}</div>}
  </div>;
}

function computeLineupChanges(lineup: ConfirmedLineup | null | undefined, previous: HistoryTeamState | { players: number[]; slots: Record<string, number> } | undefined) {
  if (!lineup?.confirmed || lineup.players.length < 8 || !previous) return null;
  const currentIds = new Set(lineup.players.map(p => p.playerId));
  const previousIds = new Set(previous.players);
  const incoming = lineup.players.filter(p => !previousIds.has(p.playerId)).map(p => ({ id: p.playerId, name: p.name, battingOrder: p.battingOrder }));
  const outgoing = [...previousIds].filter(id => !currentIds.has(id)).map(id => ({ id, name: undefined as string | undefined, battingOrder: previous.slots[String(id)] }));
  return { incoming, outgoing };
}

function LineupChangesDetails({ game, homeCoach, awayCoach, playerNames, historicalSnapshot }: { game: ScheduleGame; homeCoach: ReturnType<typeof coachRotationSignals>; awayCoach: ReturnType<typeof coachRotationSignals>; playerNames?: Record<string, string>; historicalSnapshot?: HistorySnapshot }) {
  const resolveName = (id: number, fallback?: string) => fallback || playerNames?.[String(id)] || `Jugador #${id}`;
  const derive = (side: ScheduleGame['home'], liveCoach: ReturnType<typeof coachRotationSignals>) => {
    if (historicalSnapshot) {
      const prev = historicalSnapshot.teamStates[side.team];
      const changes = computeLineupChanges(side.lineup, prev);
      return changes;
    }
    if (!liveCoach) return null;
    return { incoming: liveCoach.incoming ?? [], outgoing: liveCoach.outgoing ?? [] };
  };
  const awayChanges = derive(game.away, awayCoach);
  const homeChanges = derive(game.home, homeCoach);
  const section = (side: ScheduleGame['home'], changes: ReturnType<typeof computeLineupChanges> | null) => {
    if (!changes) return null;
    if (changes.incoming.length === 0 && changes.outgoing.length === 0) return null;
    return <div className="lineup-changes-team" key={side.team}>
      <h5><TeamLogo team={side.team} name={side.name} className="lineup-changes-logo" /><strong>{side.team}</strong><small>{side.name}</small></h5>
      {changes.incoming.length > 0 && <div><span className="lineup-changes-label lineup-changes-in">Nuevos ({changes.incoming.length})</span><ul>{changes.incoming.sort((a, b) => (a.battingOrder ?? 99) - (b.battingOrder ?? 99)).map(player => <li key={player.id}><b>{player.battingOrder ?? '—'}</b><span>{resolveName(player.id, player.name)}</span></li>)}</ul></div>}
      {changes.outgoing.length > 0 && <div><span className="lineup-changes-label lineup-changes-out">Ausentes ({changes.outgoing.length})</span><ul>{changes.outgoing.sort((a, b) => (a.battingOrder ?? 99) - (b.battingOrder ?? 99)).map(player => <li key={player.id}><b>{player.battingOrder ?? '—'}</b><span>{resolveName(player.id, player.name)}</span></li>)}</ul></div>}
    </div>;
  };
  const awaySection = section(game.away, awayChanges);
  const homeSection = section(game.home, homeChanges);
  if (!awaySection && !homeSection) return null;
  const totalIn = (awayChanges?.incoming.length ?? 0) + (homeChanges?.incoming.length ?? 0);
  const totalOut = (awayChanges?.outgoing.length ?? 0) + (homeChanges?.outgoing.length ?? 0);
  return <details className="lineup-changes-details"><summary><ChevronDown size={14} /><span>Ver ausentes ({totalOut}) y nuevos ({totalIn})</span></summary><div className="lineup-changes-body">{awaySection}{homeSection}</div></details>;
}

function PlayerPerformanceModal({ game, playerPerformance, historicalSnapshot, onClose }: { game: ScheduleGame; playerPerformance: PlayerPerformancePayload | null; historicalSnapshot?: HistorySnapshot; onClose: () => void }) {
  useEffect(() => {
    const handler = (event: KeyboardEvent) => { if (event.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose]);
  const lookup = historicalSnapshot?.players ?? playerPerformance?.players ?? {};
  const windowDays = historicalSnapshot?.windowDays ?? playerPerformance?.windowDays ?? 30;
  const cutoffDate = historicalSnapshot?.cutoffDate ?? playerPerformance?.cutoffDate ?? '—';
  const buildSide = (side: ScheduleGame['home']) => {
    const lineupPlayers = side.lineup?.confirmed ? side.lineup.players : [];
    const batters = lineupPlayers.map(player => ({ id: player.playerId, name: player.name, battingOrder: player.battingOrder, entry: lookup[String(player.playerId)] }));
    const pitcherEntry = Object.values(lookup).find(entry => entry.kind === 'pitcher' && entry.name === side.pitcher);
    return { team: side.team, name: side.name, pitcher: side.pitcher, pitcherEntry, batters };
  };
  const away = buildSide(game.away);
  const home = buildSide(game.home);
  const labelClass = (label: string) => {
    const key = label.normalize('NFD').replace(/[̀-ͯ]/g, '').toLowerCase().replace(/\s+/g, '-');
    return `performance-tag performance-tag-${key}`;
  };
  const batterRow = (row: { name?: string; battingOrder?: number; entry?: PlayerPerformanceEntry }, index: number) => <div key={`${row.name ?? index}-${index}`} className="performance-row">
    <div className="performance-slot"><b>{row.battingOrder ?? '—'}</b><span className="performance-name">{row.entry?.name ?? row.name ?? 'Sin nombre'}</span></div>
    {row.entry ? <>
      <span className={labelClass(row.entry.label)}>{row.entry.label}</span>
      <div className="performance-stats"><i>PA</i><b>{row.entry.pa ?? '—'}</b><i>AVG</i><b>{row.entry.avg != null ? row.entry.avg.toFixed(3).replace(/^0/, '') : '—'}</b><i>OPS</i><b>{row.entry.ops != null ? row.entry.ops.toFixed(3).replace(/^0/, '') : '—'}</b><i>HR</i><b>{row.entry.hr ?? '—'}</b><i>K%</i><b>{row.entry.kPct != null ? `${(row.entry.kPct * 100).toFixed(0)}%` : '—'}</b></div>
    </> : <span className="performance-tag performance-tag-sin-datos">sin datos L30</span>}
  </div>;
  const pitcherBlock = (label: string, name: string | null | undefined, entry: PlayerPerformanceEntry | undefined) => name ? <div className="performance-row performance-row-pitcher">
    <div className="performance-slot"><b>SP</b><span className="performance-name">{name}</span></div>
    {entry ? <>
      <span className={labelClass(entry.label)}>{entry.label}</span>
      <div className="performance-stats"><i>IP</i><b>{entry.ip ?? '—'}</b><i>ERA</i><b>{entry.era != null ? entry.era.toFixed(2) : '—'}</b><i>WHIP</i><b>{entry.whip != null ? entry.whip.toFixed(2) : '—'}</b><i>K/9</i><b>{entry.k9 != null ? entry.k9.toFixed(1) : '—'}</b><i>BB/9</i><b>{entry.bb9 != null ? entry.bb9.toFixed(1) : '—'}</b></div>
    </> : <span className="performance-tag performance-tag-sin-datos">sin datos L30</span>}
  </div> : null;
  const sideBlock = (data: ReturnType<typeof buildSide>) => <section className="performance-team">
    <header><TeamLogo team={data.team} name={data.name} className="performance-team-logo" /><div><strong>{data.team}</strong><small>{data.name}</small></div></header>
    {pitcherBlock('Abridor', data.pitcher, data.pitcherEntry)}
    {data.batters.length ? <>
      <h5>LINEUP CONFIRMADO</h5>
      {data.batters.sort((a, b) => (a.battingOrder ?? 99) - (b.battingOrder ?? 99)).map(batterRow)}
    </> : <p className="performance-waiting">Alineación aún no confirmada. Se mostrará al publicarse.</p>}
  </section>;
  return <div className="telegram-modal-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
    <section className="telegram-modal performance-modal" role="dialog" aria-modal="true" aria-labelledby={`performance-title-${game.gamePk}`}>
      <header><div><span><Activity size={14} /> Rendimiento L{windowDays}{historicalSnapshot ? ` · congelado a ${historicalSnapshot.cutoffDate}` : ''}</span><h3 id={`performance-title-${game.gamePk}`}>Convocados de {game.away.team} vs {game.home.team}</h3></div><button type="button" onClick={onClose} aria-label="Cerrar"><X size={18} /></button></header>
      <div className="performance-body"><aside className="performance-legend"><h5>GLOSARIO</h5><dl><dt>PA</dt><dd>Apariciones al plato (turnos, incluye BB y HBP).</dd><dt>AVG</dt><dd>Promedio de bateo: hits ÷ turnos oficiales.</dd><dt>OPS</dt><dd>On-base + Slugging. Referencia: <b>.700 normal</b>, <b>.800+ élite</b>.</dd><dt>HR</dt><dd>Home runs conectados en L30.</dd><dt>K%</dt><dd>Ponches por aparición. Menor es mejor (&lt; 20% élite).</dd><dt>IP</dt><dd>Entradas lanzadas. Cada out = ⅓ de entrada.</dd><dt>ERA</dt><dd>Carreras limpias por 9 IP. <b>≤ 3.5 élite</b>, <b>≥ 5 débil</b>.</dd><dt>WHIP</dt><dd>(BB + Hits) ÷ IP. <b>≤ 1.15 élite</b>, <b>≥ 1.4 riesgo</b>.</dd><dt>K/9</dt><dd>Ponches por 9 IP. <b>≥ 9 dominante</b>.</dd><dt>BB/9</dt><dd>Bases por bolas por 9 IP. Menor es mejor.</dd></dl><h5>ETIQUETAS</h5><ul className="performance-legend-tags"><li><span className="performance-tag performance-tag-racha-caliente">racha caliente</span><small>OPS ≥ .820 con K% ≤ 28%</small></li><li><span className="performance-tag performance-tag-solido">sólido</span><small>OPS ≥ .720 · ERA ≤ 4.20</small></li><li><span className="performance-tag performance-tag-normal">normal</span><small>Entre las bandas</small></li><li><span className="performance-tag performance-tag-racha-fria">racha fría</span><small>OPS ≤ .580 o K% muy alto</small></li><li><span className="performance-tag performance-tag-dominante">dominante</span><small>ERA ≤ 3.20 y WHIP ≤ 1.15</small></li><li><span className="performance-tag performance-tag-en-problemas">en problemas</span><small>ERA ≥ 5.50 o WHIP ≥ 1.50</small></li></ul><p className="performance-legend-note">Todos los valores son de los últimos 30 días de acción — no de temporada completa.</p></aside>{sideBlock(away)}{sideBlock(home)}</div>
      <footer><small>Ventana: últimos {windowDays} días · cutoff {cutoffDate}. Etiquetas basadas en OPS/K% para bateadores y ERA/WHIP para abridores.</small><button type="button" className="telegram-cancel" onClick={onClose}>Cerrar</button></footer>
    </section>
  </div>;
}

function PairIndicationTable({ signal }: { signal: IndicationPairSignal }) {
  const delta = signal.summary.deltaVsHomePoints;
  return (
    <article className="pair-signal-card">
      <header><div><span>CRUCE VISITANTE → LOCAL</span><h3>{signal.title}</h3><small>{signal.eligibleGames.toLocaleString('es-MX')} juegos con ambas indicaciones</small></div><div className="pair-summary"><strong>{pct(signal.summary.accuracy, 2)}</strong><span>accuracy WF</span><small className={(delta ?? 0) >= 0 ? 'positive' : 'negative'}>{delta == null ? '—' : `${delta >= 0 ? '+' : ''}${delta.toFixed(2)} pts`} vs elegir local</small></div></header>
      <div className="pair-table-shell"><table><thead><tr><th>Cruce</th><th>Ganó más</th><th>Histórico V-L</th><th>Accuracy WF</th><th>Vs local</th><th>2026 V-L</th><th>Muestra WF</th></tr></thead><tbody>{signal.rows.map(row => {
        const leader = row.leader === 'home' ? `Local · ${row.homeLabel}` : row.leader === 'away' ? `Visitante · ${row.awayLabel}` : 'Empate';
        return <tr key={row.key}><td><strong>{row.awayLabel}</strong><i>→</i><strong>{row.homeLabel}</strong></td><td><b>{leader}</b><small>{pct(row.leaderWinRate)}</small></td><td>{row.awayWins.toLocaleString('es-MX')}-{row.homeWins.toLocaleString('es-MX')}</td><td><b>{pct(row.walkforwardAccuracy, 2)}</b></td><td><span className={(row.deltaVsHomePoints ?? 0) >= 0 ? 'positive' : 'negative'}>{row.deltaVsHomePoints == null ? '—' : `${row.deltaVsHomePoints >= 0 ? '+' : ''}${row.deltaVsHomePoints.toFixed(2)} pts`}</span></td><td>{row.awayWins2026}-{row.homeWins2026}</td><td>{row.walkforwardCorrect}/{row.walkforwardGames}</td></tr>;
      })}</tbody></table></div>
    </article>
  );
}

function RankingTable({ rows, view, ascending, showAll }: { rows: RankingRow[]; view: string; ascending: boolean; showAll: boolean }) {
  const sorted = [...rows].sort((a, b) => {
    const field = view === 'sweep' ? 'net_sweep_rate' : view === 'vulnerable' ? 'swept_rate' : view === 'afterSweep' ? 'after_sweep_win_rate' : 'after_swept_win_rate';
    const av = a[field] ?? -1;
    const bv = b[field] ?? -1;
    return ascending ? av - bv : bv - av;
  });
  const visibleRows = showAll ? sorted : sorted.slice(0, 10);
  return (
    <div className="table-shell"><table><thead><tr><th>#</th><th>Equipo</th><th>{view === 'sweep' ? 'Balance barridas' : view === 'vulnerable' ? 'Veces barrido' : 'Siguiente juego'}</th><th>Muestra</th><th>Récord base</th></tr></thead><tbody>
      {visibleRows.map((row, index) => {
        const rate = view === 'sweep' ? row.net_sweep_rate : view === 'vulnerable' ? row.swept_rate : view === 'afterSweep' ? row.after_sweep_win_rate : row.after_swept_win_rate;
        const sample = view === 'sweep' || view === 'vulnerable' ? row.three_game_series : view === 'afterSweep' ? row.after_sweep_games : row.after_swept_games;
        const record = view === 'afterSweep' ? `${row.after_sweep_wins}-${row.after_sweep_games - row.after_sweep_wins}` : view === 'afterSwept' ? `${row.after_swept_wins}-${row.after_swept_games - row.after_swept_wins}` : `${row.sweeps}-${row.times_swept}`;
        return <tr key={row.team}><td>{index + 1}</td><td><div className="ranking-team"><TeamLogo team={row.team} name={row.team_name} className="ranking-team-logo" /><div><strong>{row.team}</strong><span>{row.team_name}</span></div></div></td><td><strong>{pct(rate)}</strong><span>{record}</span></td><td>{sample}</td><td>{pct(row.baseline_win_rate)}</td></tr>;
      })}
    </tbody></table></div>
  );
}

function SeriesSweepAuditSection({
  stats, schedule, previousGames, active, odds,
}: {
  stats: StatsData;
  schedule: ScheduleData;
  previousGames: ScheduleGame[];
  active: Record<FactorKey, boolean>;
  odds: OddsData | null;
}) {
  const audit = stats.seriesSweepAudit;
  const forecasts = useMemo(
    () => buildSweepForecasts(schedule, previousGames, stats, active, odds),
    [schedule, previousGames, stats, active, odds],
  );
  const stages: { key: 'beforeSeries' | 'afterGame1' | 'afterGame2'; label: string; detail: string }[] = [
    { key: 'beforeSeries', label: 'Antes de la serie', detail: 'Primer juego + conversión histórica restante' },
    { key: 'afterGame1', label: 'Después de 1-0', detail: 'Segundo juego + probabilidad del tercero' },
    { key: 'afterGame2', label: 'Después de 2-0', detail: 'Probabilidad congelada del tercer juego' },
  ];
  return (
    <section className="section series-wf-section" id="barridas-walkforward">
      <div className="section-heading split-heading">
        <div><span className="eyebrow"><Trophy size={14} /> Barridas de serie · prueba</span><h2>La barrida, juego por juego</h2><p>Calcula para todos los equipos la posibilidad de terminar 3-0 y registra aparte quién intenta barrer dos series consecutivas. Las probabilidades históricas quedan congeladas antes de cada partido.</p></div>
        <div className="audit-baseline"><strong>{pct(audit.historical.afterGame2.rate, 2)}</strong><span>Convirtieron después de 2-0</span><small>{audit.historical.afterGame2.sweeps}/{audit.historical.afterGame2.games} oportunidades</small></div>
      </div>
      <div className="series-proof-grid">
        <article><span>BARRIDA ANTES DE EMPEZAR</span><strong>{pct(audit.historical.beforeSeries.rate, 2)}</strong><small>{audit.historical.beforeSeries.sweeps}/{audit.historical.beforeSeries.games} lados de serie</small></article>
        <article><span>DESPUÉS DE GANAR EL PRIMERO</span><strong>{pct(audit.historical.afterGame1.rate, 2)}</strong><small>{audit.historical.afterGame1.games.toLocaleString('es-MX')} oportunidades</small></article>
        <article><span>DOS BARRIDAS SEGUIDAS</span><strong>{pct(audit.historical.afterPreviousSweep.rate, 2)}</strong><small>{audit.historical.afterPreviousSweep.sweeps}/{audit.historical.afterPreviousSweep.games} intentos</small></article>
        <article><span>SIN FUGA TEMPORAL</span><strong>{audit.temporalLeakageViolations}</strong><small>violaciones detectadas</small></article>
      </div>
      <div className="series-live-head"><div><span>EN JUEGO AHORA</span><h3>Probabilidad dinámica de barrida</h3></div><small>Se recalcula con los factores activos; no altera el pick del partido.</small></div>
      {forecasts.length ? <div className="series-live-grid">{forecasts.map(forecast => (
        <article key={forecast.game.gamePk} className="series-live-card">
          <header><span>Juego {forecast.stage} de 3</span><b>{gameTime(forecast.game.gameDate)} CDMX</b></header>
          <h4>{forecast.game.away.team} vs {forecast.game.home.team}</h4>
          <div>{forecast.sides.map(side => <div key={side.team} className="series-side"><TeamLogo team={side.team} name={side.name} className="series-team-logo" /><span><strong>{side.name}</strong><small>{forecast.stage === 3 ? 'Necesita ganar este juego' : `Debe ganar ${4 - forecast.stage} juegos contando el de hoy`}</small>{side.consecutiveAttempt && <em>Busca segunda barrida seguida</em>}</span><b>{pct(side.probability, 1)}</b></div>)}</div>
          <footer>Probabilidad prepartido · serie exacta de tres</footer>
        </article>
      ))}</div> : <div className="series-empty"><Info size={18} /><span>No hay una serie exacta de tres con posibilidad de barrida identificada en los juegos de hoy.</span></div>}
      <div className="series-wf-head"><div><span>VALIDACIÓN CRONOLÓGICA</span><h3>Qué tan bien quedaron calibradas</h3></div><small>{audit.method}</small></div>
      <div className="series-stage-grid">{stages.map(stage => {
        const metrics = audit.walkforward.stages[stage.key];
        return <article key={stage.key}><span>{stage.label}</span><strong>{pct(metrics.averageProbability, 2)} <i>pronosticado</i></strong><b>{pct(metrics.observedRate, 2)} ocurrió</b><small>{stage.detail} · Brier {metrics.brier?.toFixed(3) ?? '—'} · {metrics.events.toLocaleString('es-MX')} casos</small>{stage.key === 'afterGame2' && <em>Accuracy del juego: {pct(metrics.accuracyAt50, 2)}</em>}</article>;
      })}<article className="repeat-stage"><span>Doble barrida</span><strong>{pct(audit.walkforward.doubleSweep.averageProbability, 2)} <i>pronosticado</i></strong><b>{pct(audit.walkforward.doubleSweep.observedRate, 2)} ocurrió</b><small>Brier {audit.walkforward.doubleSweep.brier?.toFixed(3) ?? '—'} · {audit.walkforward.doubleSweep.events} series posteriores a barrida</small></article></div>
      <div className="series-team-table"><div><span>EQUIPO</span><b>Barridas</b><b>Convierte 2-0</b><b>Repite barrida</b><b>Muestra repetición</b></div>{audit.teams.map(row => <div key={row.team}><span><TeamLogo team={row.team} name={row.name} className="series-table-logo" /><strong>{row.team}</strong><small>{row.name}</small></span><b>{pct(row.sweepRate)}</b><b>{pct(row.stage2Rate)}</b><b>{pct(row.repeatRate)}</b><em>{row.repeatSweeps}/{row.repeatAttempts}</em></div>)}</div>
      <p className="audit-conclusion audit-watch"><ShieldCheck size={16} /> {audit.recommendation}</p>
    </section>
  );
}

function SeasonL10AuditSection({ audit }: { audit: SeasonL10Audit }) {
  const aligned = audit.patterns.find(row => row.key === 'aligned');
  const largeAligned = audit.patterns.find(row => row.key === 'largeAligned');
  const strict = audit.exactScenario.strict;
  const productionDelta = audit.productionModel?.deltaPoints ?? null;
  const signedPoints = (value: number | null) => value == null ? '—' : `${value >= 0 ? '+' : ''}${value.toFixed(2)} pts`;
  return (
    <section className="section audit-section coach-audit" id="auditoria-temporada-l10">
      <div className="section-heading split-heading">
        <div><span className="eyebrow"><TrendingUp size={14} /> Auditoría de forma reciente</span><h2>Temporada contra los últimos 10</h2><p>Compara el récord acumulado y el L10 que cada equipo tenía antes de jugar. También prueba si juntarlos mejora la predicción fuera de muestra.</p></div>
        <div className="audit-baseline"><strong>{pct(strict.winRate, 2)}</strong><span>caso 5-5 vs 4-6</span><small>{strict.wins}/{strict.games} ganó el equipo de mejor temporada</small></div>
      </div>
      <div className="pair-audit-note"><ShieldCheck size={16} /><span>{audit.method} Corte {audit.cutoffDate}; {audit.temporalLeakageViolations} fugas temporales detectadas.</span></div>
      <div className="coach-proof-grid">
        <article><span>MEJOR TEMPORADA + MEJOR L10</span><strong>{pct(aligned?.winRate, 2)}</strong><small>{aligned?.wins.toLocaleString('es-MX')}/{aligned?.games.toLocaleString('es-MX')} juegos</small></article>
        <article><span>VENTAJA GRANDE Y CONFIRMADA</span><strong>{pct(largeAligned?.winRate, 2)}</strong><small>temporada ≥15 pts y L10 ≥2 victorias</small></article>
        <article><span>APORTE L10 EN PRUEBA AISLADA</span><strong>{signedPoints(audit.walkforward.gainVsSeasonPoints)}</strong><small>vs usar solo temporada · {audit.walkforward.games.toLocaleString('es-MX')} OOS</small></article>
        <article><span>EFECTO EN EL MODELO BASE ACTUAL</span><strong className={productionDelta != null && productionDelta > 0 ? 'positive' : 'negative'}>{signedPoints(productionDelta)}</strong><small>{audit.productionModel ? `${audit.productionModel.withL10.correct - audit.productionModel.withoutL10.correct} aciertos al activar L10` : 'comparación no disponible'}</small></article>
      </div>
      <div className="audit-descriptive">
        <div><span>PATRÓN OBSERVADO</span><b>Ganó mejor récord</b><b>2026</b><b>Muestra</b></div>
        {audit.patterns.map(row => <div key={row.key}><strong>{row.label}</strong><b>{pct(row.winRate, 2)}</b><b>{pct(row.seasons['2026']?.winRate, 2)}</b><span>{row.games.toLocaleString('es-MX')}</span></div>)}
        <div><strong>{audit.exactScenario.label}</strong><b>{pct(audit.exactScenario.broad.winRate, 2)}</b><b>{pct(audit.exactScenario.broad.seasons['2026']?.winRate, 2)}</b><span>{audit.exactScenario.broad.games.toLocaleString('es-MX')}</span></div>
      </div>
      <div className="audit-descriptive">
        <div><span>MODELO WALK-FORWARD</span><b>Accuracy OOS</b><b>2026</b><b>Muestra</b></div>
        {audit.walkforward.models.map(model => <div key={model.key}><strong>{model.label}</strong><b>{pct(model.accuracy, 2)}</b><b>{pct(model.seasons['2026']?.accuracy, 2)}</b><span>{model.correct.toLocaleString('es-MX')}/{model.games.toLocaleString('es-MX')}</span></div>)}
      </div>
      <p className="audit-conclusion audit-watch"><Info size={16} /> {audit.recommendation}</p>
    </section>
  );
}

const COMPARISON_CONFIGS = [
  { key: 'base' as const, label: 'Modelo base', short: 'Base' },
  { key: 'topAccuracy' as const, label: 'Combinación #1', short: '#1' },
  { key: 'fifteenGames' as const, label: 'Mayor accuracy · 15', short: 'A15' },
  { key: 'winningDays' as const, label: 'Más días ganadores', short: 'MDG' },
  { key: 'best2026' as const, label: 'Mejor 2026', short: '26' },
];

function formatRelativeTime(iso: string | null): string {
  if (!iso) return '';
  const ts = Date.parse(iso);
  if (!Number.isFinite(ts)) return '';
  const diffSec = Math.max(0, Math.round((Date.now() - ts) / 1000));
  if (diffSec < 60) return 'hace unos segundos';
  const mins = Math.round(diffSec / 60);
  if (mins < 60) return `hace ${mins} min`;
  const hrs = Math.round(mins / 60);
  if (hrs < 24) return `hace ${hrs} h`;
  const days = Math.round(hrs / 24);
  return `hace ${days} d`;
}

function pickFromProbability(homeP: number | null, game: ScheduleGame): { code: string; prob: number } | null {
  if (homeP == null || !Number.isFinite(homeP)) return null;
  const homeCode = alias(game.home.team);
  const awayCode = alias(game.away.team);
  return homeP >= 0.5 ? { code: homeCode, prob: homeP } : { code: awayCode, prob: 1 - homeP };
}

function ComparisonSection({ stats, odds, scheduleArchive, schedule, tomorrowSchedule, recentWalkforward, comparisonDate, onChangeDate, strikecast, strikecastLoading, strikecastError, valuePicks, valuePicksUpdatedAt, masks, oddsHistory, calibratorBuckets, playerPerformance }: {
  stats: StatsData;
  odds: OddsData | null;
  scheduleArchive: Record<string, ScheduleData>;
  schedule: ScheduleData;
  tomorrowSchedule: ScheduleData | null;
  recentWalkforward: Record<string, FrozenWalkforwardDay>;
  comparisonDate: string;
  onChangeDate: (value: string) => void;
  strikecast: StrikecastPick[];
  strikecastLoading: boolean;
  strikecastError: string | null;
  valuePicks: ValuePick[];
  valuePicksUpdatedAt: string | null;
  masks: ComparisonMaskSet | null;
  oddsHistory: OddsHistoryPayload | null;
  calibratorBuckets: { key: string; n: number; winRate: number }[] | null;
  playerPerformance: PlayerPerformancePayload | null;
}) {
  const todayDate = mexicoIsoDate();
  const tomorrowDate = mexicoIsoDate(1);
  const quickDates = [
    { key: 'hoy', label: 'Hoy', date: todayDate },
    { key: 'ayer', label: 'Ayer', date: mexicoIsoDate(-1) },
    { key: 'antier', label: 'Antier', date: mexicoIsoDate(-2) },
    { key: 'week', label: 'Hace 1 semana', date: mexicoIsoDate(-7) },
  ];
  const scheduleForDate = scheduleArchive[comparisonDate] ?? (comparisonDate === todayDate ? schedule : comparisonDate === tomorrowDate ? tomorrowSchedule ?? undefined : undefined);
  const games = scheduleForDate?.games ?? [];
  const isHistorical = comparisonDate < todayDate;
  const frozenDay = recentWalkforward[comparisonDate];
  const frozenGames = new Map((frozenDay?.games ?? []).map(game => [game.gamePk, game]));
  const strikecastByPk = new Map(strikecast.map(item => [item.gamePk, item]));
  const valueByPk = new Map(valuePicks.map(item => [item.gamePk, item]));
  const label = new Intl.DateTimeFormat('es-MX', { timeZone: 'UTC', weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' }).format(new Date(`${comparisonDate}T12:00:00Z`));

  const rows = games.map(game => {
    const strikecastGame = strikecastByPk.get(game.gamePk);
    const strikecastPick = strikecastGame?.pick && strikecastGame.pHome != null
      ? { code: alias(strikecastGame.pick), prob: strikecastGame.pick === strikecastGame.home ? strikecastGame.pHome : 1 - strikecastGame.pHome }
      : null;

    const valueRaw = valueByPk.get(game.gamePk);
    const valuePick = valueRaw && valueRaw.pickCode && valueRaw.pickProb != null && valueRaw.tier !== 'NEUTRO'
      ? { code: alias(valueRaw.pickCode), prob: valueRaw.pickProb, tier: valueRaw.tier, edge: valueRaw.edgeWin }
      : null;

    const configs = COMPARISON_CONFIGS.map(config => {
      const mask = masks?.[config.key];
      if (mask == null) return { ...config, pick: null as ReturnType<typeof pickFromProbability> };
      let homeP: number | null = null;
      if (isHistorical) {
        const frozen = frozenPredictionForMask(frozenGames.get(game.gamePk), mask);
        homeP = frozen?.homeProbability ?? null;
      } else {
        const estimate = estimateGame(game, stats, maskToActive(mask), findCurrentOdds(game, odds));
        homeP = estimate.homeProbability;
      }
      return { ...config, pick: pickFromProbability(homeP, game) };
    });

    const currentOddsForGame = findCurrentOdds(game, odds);
    const marketPick = marketPickFromCurrentOdds(game, currentOddsForGame, oddsHistory);
    // Historical dates must not use today's player performance as a pregame pick.
    const alignmentPick = !isHistorical && playerPerformance
      ? alignmentPickForGame(game, estimateGame(game, stats, maskToActive(0), currentOddsForGame), playerPerformance.players)
      : null;
    const allPicks: { code: string }[] = [
      ...configs.map(c => c.pick).filter((p): p is { code: string; prob: number } => Boolean(p)),
      ...(strikecastPick ? [strikecastPick] : []),
      ...(valuePick ? [{ code: valuePick.code }] : []),
      ...(marketPick ? [marketPick] : []),
      ...(alignmentPick ? [alignmentPick] : []),
    ];
    const tally = new Map<string, number>();
    allPicks.forEach(pick => tally.set(pick.code, (tally.get(pick.code) ?? 0) + 1));
    let consensus: { code: string; count: number; total: number } | null = null;
    if (allPicks.length) {
      const [code, count] = [...tally.entries()].sort((a, b) => b[1] - a[1])[0];
      consensus = { code, count, total: allPicks.length };
    }

    const isFinal = game.abstractState === 'Final' && game.home.score != null && game.away.score != null && game.home.score !== game.away.score;
    const winnerCode = isFinal ? (game.home.score! > game.away.score! ? alias(game.home.team) : alias(game.away.team)) : null;

    const byName = Object.fromEntries(configs.map(c => [c.key, c.pick])) as Record<'base' | 'topAccuracy' | 'fifteenGames' | 'winningDays' | 'best2026', { code: string; prob: number } | null>;
    const scPick = strikecastPick;
    const smAllEqual = byName.base && byName.topAccuracy && byName.fifteenGames && byName.winningDays
      && byName.base.code === byName.topAccuracy.code
      && byName.topAccuracy.code === byName.fifteenGames.code
      && byName.fifteenGames.code === byName.winningDays.code
      ? byName.base.code : null;
    const allFiveEqual = smAllEqual && scPick && scPick.code === smAllEqual ? smAllEqual : null;
    const avgProb = allFiveEqual ? (
      [byName.base!, byName.topAccuracy!, byName.fifteenGames!, byName.winningDays!, scPick!].reduce((sum, p) => sum + p.prob, 0) / 5
    ) : null;
    const trioBEqual = byName.topAccuracy && byName.winningDays && scPick
      && byName.topAccuracy.code === byName.winningDays.code
      && byName.winningDays.code === scPick.code
      ? byName.topAccuracy.code : null;

    // === Calibrador empírico (bucket walk-forward 4-dim: side|conf|scAgree|scMag).
    // Fallback a `side|conf|nosc` cuando StrikeCast no está disponible.
    // Ver scripts/stacked_calibrator.ps1.
    let calibratedWinRate: number | null = null;
    let calibratedBucket: string | null = null;
    let calibratedBucketN: number | null = null;
    if (calibratorBuckets && byName.topAccuracy) {
      const homeCode = alias(game.home.team);
      const top1IsHome = byName.topAccuracy.code === homeCode;
      const top1Conf = byName.topAccuracy.prob;
      const c = top1Conf * 100;
      const confBucket = c < 55 ? '50-55' : c < 60 ? '55-60' : c < 65 ? '60-65' : c < 70 ? '65-70' : '70+';
      const side = top1IsHome ? 'home' : 'away';
      let key: string;
      if (strikecastGame?.pHome != null) {
        const scPHome = strikecastGame.pHome;
        const scIsHome = scPHome >= 0.5;
        const scAgree = ((scIsHome && top1IsHome) || (!scIsHome && !top1IsHome)) ? 'y' : 'n';
        const scPickProb = top1IsHome ? scPHome : 1 - scPHome;
        const scMag = scPickProb < 0.45 ? 'w' : scPickProb < 0.55 ? 'm' : 's';
        key = `${side}|${confBucket}|${scAgree}|${scMag}`;
      } else {
        key = `${side}|${confBucket}|nosc`;
      }
      const entry = calibratorBuckets.find(b => b.key === key);
      if (entry && entry.n >= 40) {
        calibratedWinRate = entry.winRate;
        calibratedBucket = key;
        calibratedBucketN = entry.n;
      }
    }

    // Bandera informativa (no demota el pick): F5 discrepa del pick unánime
    const scRaw = strikecastGame;
    let f5DisagreesPick = false;
    if (allFiveEqual && scRaw?.f5PHome != null && scRaw.pHome != null) {
      const f5FavorsHome = scRaw.f5PHome >= 0.5;
      const pickIsHome = allFiveEqual === alias(game.home.team);
      f5DisagreesPick = f5FavorsHome !== pickIsHome;
    }

    // Señales de rotación + serie previa: solo hoy/mañana
    let rotationWarning: string | null = null;
    let rotationBoost: string | null = null;
    let rotationSignal: { team: string; note: string; strong: boolean } | null = null;
    let rotationChangesLabel: string | null = null;
    let seriesWarning: string | null = null;
    let seriesBoost: string | null = null;
    let seriesSignal: { team: string; note: string; strong: boolean } | null = null;
    if (!isHistorical) {
      const rotEst = estimateGame(game, stats, maskToActive(0), findCurrentOdds(game, odds));
      // Serie previa: usa homeContext/awayContext ('sweep' | 'swept' | 'lost_1_2' | 'none')
      const hc = rotEst.homeContext?.code ?? 'none';
      const ac = rotEst.awayContext?.code ?? 'none';
      const homeAlias = alias(game.home.team), awayAlias = alias(game.away.team);
      if (hc === 'sweep' && ac === 'sweep') seriesSignal = { team: homeAlias, note: `Ambos barrieron su serie previa — local 68% histórico (n=38)`, strong: true };
      const pickTeam = allFiveEqual ?? trioBEqual;
      if (pickTeam) {
        const isHomePick = pickTeam === homeAlias;
        const pickCtx = isHomePick ? hc : ac;
        const rivalCtx = isHomePick ? ac : hc;
        if (pickCtx === 'sweep' && rivalCtx === 'sweep' && isHomePick) seriesBoost = `Ambos barrieron — local 68% histórico`;
        else if (pickCtx === 'sweep' && rivalCtx === 'sweep' && !isHomePick) seriesWarning = `Ambos barrieron — visita 32% histórico`;
        else if (pickCtx === 'swept') seriesWarning = `Viene de ser barrido 0-3 (histórico 47%)`;
        else if (pickCtx === 'sweep' && rivalCtx !== 'sweep') seriesBoost = `Viene de barrer 3-0 (histórico 53.5%)`;
        else if (pickCtx === 'swept' && rivalCtx === 'lost_1_2') seriesWarning = `Barrido vs rival que perdió 1-2 (histórico 49%)`;
      }
      if (rotEst.coachRotationAvailable && rotEst.homeCoach && rotEst.awayCoach) {
        const homeCh = rotEst.homeCoach.changes ?? 0;
        const awayCh = rotEst.awayCoach.changes ?? 0;
        rotationChangesLabel = `${alias(game.away.team)} ${awayCh}c · ${alias(game.home.team)} ${homeCh}c`;
        // Determinar la señal de rotación
        let stableTeam: string | null = null, stableCh = 0, chaoticCh = 0;
        if (homeCh <= 1 && awayCh >= 3) { stableTeam = alias(game.home.team); stableCh = homeCh; chaoticCh = awayCh; }
        else if (awayCh <= 1 && homeCh >= 3) { stableTeam = alias(game.away.team); stableCh = awayCh; chaoticCh = homeCh; }
        else if (homeCh <= 1 && awayCh === 2) { stableTeam = alias(game.home.team); stableCh = homeCh; chaoticCh = awayCh; }
        else if (awayCh <= 1 && homeCh === 2) { stableTeam = alias(game.away.team); stableCh = awayCh; chaoticCh = homeCh; }

        if (stableTeam) {
          const strong = chaoticCh >= 3;
          rotationSignal = { team: stableTeam, note: `Lineup estable (${stableCh}) vs rival con ${chaoticCh} cambios${strong ? ' — histórico 60-68%' : ' — histórico ~55%'}`, strong };
        }
        // Caso simétrico caótico 4+/4+: home gana 58.8% (2026: 60%)
        else if (homeCh >= 4 && awayCh >= 4) {
          rotationSignal = { team: alias(game.home.team), note: `Ambos con 4+ cambios (${awayCh}·${homeCh}) — local histórico 58.8% (2026: 60%)`, strong: true };
        }

        // Marca sobre el pick recomendado si coincide
        const pickTeam = allFiveEqual ?? trioBEqual;
        if (pickTeam) {
          const isHomePick = pickTeam === alias(game.home.team);
          const pickChanges = isHomePick ? homeCh : awayCh;
          const rivalChanges = isHomePick ? awayCh : homeCh;
          if (pickChanges <= 1 && rivalChanges >= 3) rotationBoost = `Lineup estable (${pickChanges}) vs rival con ${rivalChanges} cambios`;
          else if (pickChanges <= 1 && rivalChanges === 2) rotationBoost = `Estable vs rival con 2 cambios`;
          else if (pickChanges >= 3 && rivalChanges <= 1) rotationWarning = `Cambió ${pickChanges} vs rival estable`;
          else if (pickChanges >= 4 && rivalChanges === 2) rotationWarning = `Cambió ${pickChanges} vs rival con 2 (histórico ~48%)`;
          else if (pickChanges >= 4 && rivalChanges >= 4 && isHomePick) rotationBoost = `Ambos con 4+ cambios — local (60% hist.)`;
          else if (pickChanges >= 4 && rivalChanges >= 4 && !isHomePick) rotationWarning = `Ambos con 4+ cambios — visita (histórico 40%)`;
          // Nuevo: pick es visitante con 3 cambios contra local con 4+ → visitante gana 52.6% histórico (+6.6pp vs esperado)
          else if (!isHomePick && pickChanges === 3 && rivalChanges >= 4) rotationBoost = `Visita cambió 3 · local cambió ${rivalChanges} (histórico 53%)`;
          else if (isHomePick && pickChanges >= 4 && rivalChanges === 3) rotationWarning = `Local cambió ${pickChanges} contra visita con 3 (histórico 47%)`;
        }
      }
    }

    // 4/5 consenso (StatsMLB de acuerdo, StrikeCast disiente) — no tier por sí solo pero puede aliarse con rotación
    const fourFive = smAllEqual && scPick && scPick.code !== smAllEqual ? smAllEqual : null;

    // Nuevos tiers apilados. F5 se considera "OK" si concuerda o si no hay dato disponible.
    const f5Ok = !f5DisagreesPick; // F5 concuerda o no está disponible
    const scTier = scRaw?.confidenceTier ?? null;
    const tierNotSoft = scTier !== 'moderado';
    const tierStrong = scTier === 'lock' || scTier === 'fuerte';

    let recommendation: { tier: 'lock' | 'strong' | 'play' | 'playB' | 'rot' | 'skip'; team: string | null; label: string; note: string; warning?: string; boost?: string } = { tier: 'skip', team: null, label: 'Pasar', note: 'Sin consenso claro' };

    // Voz del mercado (voz #6): única regla validada en backtest — cuando el pick califica como
    // FUERTE y el mercado le da ≥58% al mismo lado, se promueve a LOCK (win-rate histórico 67.3%,
    // muy cerca del 69.7% del LOCK base, con +23% más volumen de picks LOCK).
    const proposedPick = allFiveEqual ?? trioBEqual ?? fourFive;
    let marketUpgrade = false;
    let marketNote: string | null = null;
    if (marketPick && proposedPick && marketPick.code === proposedPick && marketPick.prob >= 0.58) {
      marketUpgrade = true;
      marketNote = `Mercado confirma con ${Math.round(marketPick.prob * 100)}%`;
    }
    const combineWarnings = (base: string | undefined) => [base, rotationWarning ?? undefined, seriesWarning ?? undefined].filter(Boolean).join(' · ') || undefined;
    const combineBoosts = () => [rotationBoost ?? undefined, seriesBoost ?? undefined, marketNote ?? undefined].filter(Boolean).join(' · ') || undefined;

    if (allFiveEqual && f5Ok && avgProb != null && (avgProb >= 0.65 || tierStrong)) {
      recommendation = { tier: 'lock', team: allFiveEqual, label: 'LOCK', note: `5/5 + F5 concuerda + ${avgProb >= 0.65 ? `prob ${Math.round(avgProb * 1000) / 10}%` : `SC ${scTier}`} (histórico 72-73%)`, boost: combineBoosts(), warning: combineWarnings(undefined) };
    } else if (allFiveEqual && f5Ok) {
      recommendation = { tier: 'strong', team: allFiveEqual, label: 'FUERTE', note: `5/5 unánime + F5 concuerda (histórico 70%)`, boost: combineBoosts(), warning: combineWarnings(undefined) };
    } else if (trioBEqual && f5Ok) {
      recommendation = { tier: 'play', team: trioBEqual, label: 'JUEGA', note: `Trío top + F5 concuerda (histórico 70%)`, boost: combineBoosts(), warning: combineWarnings(undefined) };
    } else if (allFiveEqual && tierNotSoft && scRaw?.seriesFlip !== true && ((avgProb != null && avgProb >= 0.58) || allFiveEqual !== alias(game.home.team))) {
      // 5/5 pero F5 discrepa: solo si tier no moderado, sin series_flip, y prob≥0.58 O pick es visita
      const isVisita = allFiveEqual !== alias(game.home.team);
      recommendation = { tier: 'playB', team: allFiveEqual, label: 'JUEGA-B', note: `5/5 unánime${isVisita ? ' + visita' : ` prob ${Math.round((avgProb ?? 0) * 1000) / 10}%`} (histórico 70%)`, boost: combineBoosts(), warning: combineWarnings('F5 discrepa') };
    } else if (fourFive && rotationSignal && rotationSignal.team === fourFive && rotationSignal.strong) {
      recommendation = { tier: 'play', team: fourFive, label: 'JUEGA', note: `4 StatsMLB + rotación estable (${rotationChangesLabel})`, boost: combineBoosts() };
    } else if (fourFive && seriesSignal && seriesSignal.team === fourFive && seriesSignal.strong) {
      recommendation = { tier: 'play', team: fourFive, label: 'JUEGA', note: `4 StatsMLB + serie: ${seriesSignal.note}`, boost: combineBoosts() };
    } else if (rotationSignal && rotationSignal.strong) {
      recommendation = { tier: 'rot', team: rotationSignal.team, label: 'ROTACIÓN', note: rotationSignal.note, boost: rotationChangesLabel ?? undefined };
    } else if (seriesSignal && seriesSignal.strong) {
      recommendation = { tier: 'rot', team: seriesSignal.team, label: 'SERIE', note: seriesSignal.note };
    }

    // Fallback cuando StrikeCast no responde: aceptar 4/4 StatsMLB unánime como FUERTE.
    // Antes esto caía a PASAR porque las reglas requerían el 5º voto (SC).
    const strikecastAvailable = strikecast.length > 0;
    if (recommendation.tier === 'skip' && !strikecastAvailable && smAllEqual) {
      recommendation = {
        tier: 'strong',
        team: smAllEqual,
        label: 'FUERTE',
        note: '4/4 StatsMLB unánime (StrikeCast no disponible)',
        boost: combineBoosts(),
        warning: combineWarnings(undefined),
      };
    }

    // Filtro de mercado (validado en backtest 2024-2026, n=150):
    // FUERTE + mercado ≥58% mismo lado → LOCK. Win-rate 67.3% vs 69.7% del LOCK base.
    if (marketUpgrade && recommendation.tier === 'strong') {
      recommendation = {
        ...recommendation,
        tier: 'lock',
        label: 'LOCK',
        note: `${recommendation.note} · elevado por mercado`,
        boost: combineBoosts(),
      };
    }

    // Calibrador empírico (walk-forward bucket, ver scripts/stacked_calibrator.ps1).
    // Reasigna el tier según el win-rate histórico OOS del bucket {side, prob, agreement}.
    // Preserva volumen: cada juego con datos completos recibe un tier explícito y el equipo
    // sigue siendo el pick del top1 - sólo la etiqueta y la nota cambian.
    if (calibratedWinRate != null && byName.topAccuracy) {
      const cwr = calibratedWinRate;
      const nStr = calibratedBucketN ? ` · n=${calibratedBucketN}` : '';
      const bucketNote = `Calibrado ${(cwr * 100).toFixed(1)}%${nStr} · bucket ${calibratedBucket}`;
      let newTier: typeof recommendation.tier;
      let newLabel: string;
      if (cwr >= 0.65) { newTier = 'lock'; newLabel = 'LOCK'; }
      else if (cwr >= 0.60) { newTier = 'strong'; newLabel = 'FUERTE'; }
      else if (cwr >= 0.55) { newTier = 'play'; newLabel = 'JUEGA'; }
      else { newTier = 'playB'; newLabel = 'JUEGA-B'; }
      // Mercado sigue promoviendo FUERTE→LOCK
      if (marketUpgrade && newTier === 'strong') { newTier = 'lock'; newLabel = 'LOCK'; }
      recommendation = {
        tier: newTier,
        team: byName.topAccuracy.code,
        label: newLabel,
        note: bucketNote,
        boost: combineBoosts(),
        warning: combineWarnings(recommendation.warning),
      };
    }

    // Promoción VALOR: cuando el tier final es JUEGA / JUEGA-B, el pick cae en
    // zona de valor de mercado (odds decimal 1.80-2.30) y el modelo tiene conf ≥ 0.58,
    // el acc histórico OOS sube de 54-55% a ~60% (walk-forward 2024-2026).
    // Sube el tier un escalón y agrega la etiqueta VALOR. No toca LOCK/FUERTE porque
    // ahí el filtro baja acc (les quita chalk favorable a ROI negativo).
    if ((recommendation.tier === 'play' || recommendation.tier === 'playB') && marketPick && byName.topAccuracy) {
      const proposedPick = recommendation.team ?? byName.topAccuracy.code;
      const mProbForPick = marketPick.code === proposedPick ? marketPick.prob : 1 - marketPick.prob;
      const oddsDec = mProbForPick > 0 && mProbForPick < 1 ? 1 / mProbForPick : null;
      const modelConf = byName.topAccuracy.prob;
      if (oddsDec != null && oddsDec >= 1.80 && oddsDec <= 2.30 && modelConf >= 0.58) {
        const promotedTier = recommendation.tier === 'playB' ? 'play' : 'strong';
        const promotedLabel = promotedTier === 'strong' ? 'FUERTE · VALOR' : 'JUEGA · VALOR';
        const valorNote = `Zona valor mercado ${oddsDec.toFixed(2)} + conf ${(modelConf * 100).toFixed(1)}% · OOS 60%+ (ROI +17-22%)`;
        recommendation = {
          ...recommendation,
          tier: promotedTier,
          label: promotedLabel,
          note: `${recommendation.note} · ${valorNote}`,
        };
      }
    }

    return { game, configs, strikecastPick, valuePick, marketPick, alignmentPick, consensus, winnerCode, recommendation };
  });

  return (
    <section className="section comparison-section" id="comparacion">
      <div className="section-heading split-heading">
        <div>
          <span className="eyebrow"><BarChart3 size={14} /> Comparación de modelos</span>
          <h2>StatsMLB vs StrikeCast</h2>
          <p>Compara los picks de StatsMLB, StrikeCast, Value Model, mercado y alineación. Alineación usa la misma señal que las tarjetas de la jornada, cuando hay datos suficientes.</p>
        </div>
        <div className="comparison-summary">
          <strong>{games.length}</strong>
          <span>juegos · {label}</span>
          {valuePicksUpdatedAt && (
            <span
              style={{ display: 'block', fontSize: 11, opacity: 0.7, marginTop: 4 }}
              title={`Modelo re-entrenado ${new Date(valuePicksUpdatedAt).toLocaleString()}. Auto-refresh cada 5 min.`}
            >
              Value model: {formatRelativeTime(valuePicksUpdatedAt)} · auto-refresh 5 min
            </span>
          )}
        </div>
      </div>
      <div className="comparison-controls">
        <div className="comparison-quicks" role="tablist" aria-label="Fecha rápida">
          {quickDates.map(quick => (
            <button key={quick.key} type="button" role="tab" aria-selected={comparisonDate === quick.date} className={comparisonDate === quick.date ? 'selected' : ''} onClick={() => onChangeDate(quick.date)}>
              <span>{quick.label}</span>
              <b>{quick.date}</b>
            </button>
          ))}
        </div>
        <label className="comparison-datepicker">
          <CalendarDays size={14} />
          <span>Elegir fecha</span>
          <input type="date" value={comparisonDate} max={tomorrowDate} onChange={event => onChangeDate(event.target.value)} />
        </label>
      </div>
      {strikecastError && <div className="refresh-feedback error"><Info size={15} /><span>StrikeCast: {strikecastError}. La tabla sigue mostrando los cálculos internos.</span></div>}
      {strikecastLoading && <div className="refresh-feedback"><RefreshCw size={15} className="spin" /><span>Consultando StrikeCast…</span></div>}
      {!!games.length && (() => {
        const counts = { lock: 0, strong: 0, play: 0, playB: 0, rot: 0, skip: 0 } as Record<string, number>;
        const wins = { lock: 0, strong: 0, play: 0, playB: 0, rot: 0 } as Record<string, number>;
        const played = { lock: 0, strong: 0, play: 0, playB: 0, rot: 0 } as Record<string, number>;
        rows.forEach(({ recommendation, winnerCode }) => {
          counts[recommendation.tier]++;
          if (recommendation.team && winnerCode && (recommendation.tier === 'lock' || recommendation.tier === 'strong' || recommendation.tier === 'play' || recommendation.tier === 'playB' || recommendation.tier === 'rot')) {
            played[recommendation.tier]++;
            if (recommendation.team === winnerCode) wins[recommendation.tier]++;
          }
        });
        const totalPicks = counts.lock + counts.strong + counts.play + counts.playB + counts.rot;
        const totalPlayed = played.lock + played.strong + played.play + played.playB + played.rot;
        const totalWins = wins.lock + wins.strong + wins.play + wins.playB + wins.rot;
        return (
          <div className="reco-summary">
            <article className="reco-summary-tier lock"><span className="reco-badge reco-badge-lock">LOCK</span><strong>{counts.lock}</strong><small>picks{isHistorical && played.lock ? ` · ${wins.lock}/${played.lock} = ${(wins.lock / played.lock * 100).toFixed(0)}%` : ''}</small></article>
            <article className="reco-summary-tier strong"><span className="reco-badge reco-badge-strong">FUERTE</span><strong>{counts.strong}</strong><small>picks{isHistorical && played.strong ? ` · ${wins.strong}/${played.strong} = ${(wins.strong / played.strong * 100).toFixed(0)}%` : ''}</small></article>
            <article className="reco-summary-tier play"><span className="reco-badge reco-badge-play">JUEGA</span><strong>{counts.play}</strong><small>picks{isHistorical && played.play ? ` · ${wins.play}/${played.play} = ${(wins.play / played.play * 100).toFixed(0)}%` : ''}</small></article>
            <article className="reco-summary-tier playB"><span className="reco-badge reco-badge-playB">JUEGA-B</span><strong>{counts.playB}</strong><small>picks{isHistorical && played.playB ? ` · ${wins.playB}/${played.playB} = ${(wins.playB / played.playB * 100).toFixed(0)}%` : ''}</small></article>
            <article className="reco-summary-tier rot"><span className="reco-badge reco-badge-rot">ROT/SERIE</span><strong>{counts.rot}</strong><small>picks contexto{isHistorical && played.rot ? ` · ${wins.rot}/${played.rot} = ${(wins.rot / played.rot * 100).toFixed(0)}%` : ''}</small></article>
            <article className="reco-summary-tier skip"><span className="reco-badge reco-badge-skip">PASAR</span><strong>{counts.skip}</strong><small>sin consenso</small></article>
            <article className="reco-summary-tier total"><span className="reco-label">TOTAL RECOMENDADO</span><strong>{totalPicks}</strong><small>{isHistorical && totalPlayed ? `${totalWins}/${totalPlayed} = ${(totalWins / totalPlayed * 100).toFixed(1)}%` : `de ${games.length} juegos`}</small></article>
          </div>
        );
      })()}
      {!games.length ? (
        <div className="empty-state"><CalendarDays size={28} /><h3>No hay juegos para {label}</h3><p>Cargá otra fecha desde los atajos o el selector.</p></div>
      ) : (
        <div className="comparison-table-wrap">
          <table className="comparison-table">
            <thead>
              <tr>
                <th>Hora</th>
                <th>Visitante</th>
                <th>Local</th>
                {COMPARISON_CONFIGS.map(config => (
                  <th key={config.key} title={config.label}><span className="col-full">{config.label}</span><span className="col-short">{config.short}</span></th>
                ))}
                <th>StrikeCast</th>
                <th title="Value Model: LGBM entrenado en 2023-2025 sobre features de pitcher/lineup/park/weather/elo. Muestra pick solo si edge (p_modelo - p_mercado) >= 5%. VALOR-FUERTE si edge >= 7% y el modelo de primer-anotador coincide en dirección."><span className="col-full">Value Model</span><span className="col-short">VAL</span></th>
                <th title="Mercado (probabilidad implícita desvigada, media entre casas)">Mercado</th>
                <th title="Mismo pick de Alineación de la jornada: requiere ambos lineups confirmados y al menos cinco bateadores con datos por equipo. No se calcula para fechas pasadas con datos actuales.">Alineación</th>
                <th>Consenso</th>
                <th>Recomendación</th>
                {isHistorical && <th>Resultado</th>}
              </tr>
            </thead>
            <tbody>
              {rows.map(({ game, configs, strikecastPick, valuePick, marketPick, alignmentPick, consensus, winnerCode, recommendation }) => (
                <tr key={game.gamePk}>
                  <td className="mono">{gameTime(game.gameDate)}</td>
                  <td>{alias(game.away.team)}</td>
                  <td>{alias(game.home.team)}</td>
                  {configs.map(config => (
                    <td key={config.key} className={config.pick && winnerCode ? (config.pick.code === winnerCode ? 'pick-hit' : 'pick-miss') : ''}>
                      {config.pick ? <><b>{config.pick.code}</b> <span className="prob">{Math.round(config.pick.prob * 1000) / 10}%</span></> : <span className="prob">—</span>}
                    </td>
                  ))}
                  <td className={strikecastPick && winnerCode ? (strikecastPick.code === winnerCode ? 'pick-hit' : 'pick-miss') : ''}>
                    {strikecastPick ? <><b>{strikecastPick.code}</b> <span className="prob">{Math.round(strikecastPick.prob * 1000) / 10}%</span></> : <span className="prob">—</span>}
                  </td>
                  <td
                    className={valuePick && winnerCode ? (valuePick.code === winnerCode ? 'pick-hit' : 'pick-miss') : ''}
                    title={valuePick ? `${valuePick.tier} · edge ${(valuePick.edge! * 100).toFixed(1)}pp` : 'Sin edge >= 5%'}
                    style={valuePick?.tier === 'VALOR-FUERTE' ? { fontWeight: 700 } : undefined}
                  >
                    {valuePick ? <><b>{valuePick.code}</b> <span className="prob">{Math.round(valuePick.prob * 1000) / 10}%</span>{valuePick.tier === 'VALOR-FUERTE' && <span className="prob" style={{ color: '#22c55e', marginLeft: 4 }}>★</span>}</> : <span className="prob">—</span>}
                  </td>
                  <td className={marketPick && winnerCode ? (marketPick.code === winnerCode ? 'pick-hit' : 'pick-miss') : ''}>
                    {marketPick ? <><b>{marketPick.code}</b> <span className="prob">{Math.round(marketPick.prob * 1000) / 10}%</span></> : <span className="prob">—</span>}
                  </td>
                  <td className={alignmentPick && winnerCode ? (alignmentPick.code === winnerCode ? 'pick-hit' : 'pick-miss') : ''} title={alignmentPick ? 'Pick de alineación de la jornada' : isHistorical ? 'Sin pick de alineación congelado para esta fecha' : 'Sin señal de alineación elegible o sin datos suficientes'}>
                    {alignmentPick ? <><b>{alignmentPick.code}</b> <span className="prob">{pct(alignmentPick.prob)}</span></> : <span className="prob">—</span>}
                  </td>
                  <td className={consensus && winnerCode ? (consensus.code === winnerCode ? 'pick-hit' : '') : ''}>
                    {consensus ? <><b>{consensus.code}</b> <span className="prob">{consensus.count}/{consensus.total}{consensus.count === consensus.total ? ' unánime' : ''}</span></> : <span className="prob">—</span>}
                  </td>
                  <td className={`reco-cell reco-${recommendation.tier} ${recommendation.team && winnerCode ? (recommendation.team === winnerCode ? 'pick-hit' : 'pick-miss') : ''}`} title={[recommendation.note, recommendation.warning && `⚠ ${recommendation.warning}`, recommendation.boost && `✓ ${recommendation.boost}`].filter(Boolean).join(' · ')}>
                    <span className={`reco-badge reco-badge-${recommendation.tier}`}>{recommendation.label}</span>
                    {recommendation.team && <b className="reco-team">{recommendation.team}</b>}
                    {recommendation.warning && <span className="reco-warn" title={recommendation.warning}>⚠ {recommendation.warning.length > 20 ? recommendation.warning.slice(0, 18) + '…' : recommendation.warning}</span>}
                    {recommendation.boost && <span className="reco-boost" title={recommendation.boost}>✓ {recommendation.boost.length > 20 ? recommendation.boost.slice(0, 18) + '…' : recommendation.boost}</span>}
                  </td>
                  {isHistorical && <td className="mono">{winnerCode ?? '—'}</td>}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

export default function HomePage() {
  const [stats, setStats] = useState<StatsData | null>(null);
  const [seasonL10Audit, setSeasonL10Audit] = useState<SeasonL10Audit | null>(null);
  const [schedule, setSchedule] = useState<ScheduleData | null>(null);
  const [tomorrowSchedule, setTomorrowSchedule] = useState<ScheduleData | null>(null);
  const [scheduleArchive, setScheduleArchive] = useState<Record<string, ScheduleData>>({});
  const [recentWalkforward, setRecentWalkforward] = useState<Record<string, FrozenWalkforwardDay>>({});
  const [historicalPredictionsLoading, setHistoricalPredictionsLoading] = useState(false);
  const [previousGames, setPreviousGames] = useState<ScheduleGame[]>([]);
  const [odds, setOdds] = useState<OddsData | null>(null);
  const [telegramConfigured, setTelegramConfigured] = useState(false);
  const [updatingDaily, setUpdatingDaily] = useState(false);
  const [refreshProgress, setRefreshProgress] = useState<(RefreshProgress & { kind: 'running' | 'success' | 'error' }) | null>(null);
  const [refreshFeedback, setRefreshFeedback] = useState<{ kind: 'success' | 'warning' | 'error'; text: string } | null>(null);
  const [selectedGameDate, setSelectedGameDate] = useState(mexicoIsoDate());
  const [loadError, setLoadError] = useState('');
  const [rankingView, setRankingView] = useState('sweep');
  const [ascending, setAscending] = useState(false);
  const [showAll, setShowAll] = useState(false);
  const [active, setActive] = useState<Record<FactorKey, boolean>>({ localia: true, strength: false, recent: true, runDiff: true, series: false, marketSchedule: true, bestPlayersTest: false, coachRotationTest: false, rotationQualityTest: false, lineupFatigueTest: false, opponentFormTest: false });
  const [copyFeedback, setCopyFeedback] = useState<{ kind: 'success' | 'error'; text: string } | null>(null);
  const [comparisonDate, setComparisonDate] = useState(mexicoIsoDate());
  const [strikecastByDate, setStrikecastByDate] = useState<Record<string, StrikecastPick[]>>({});
  const [strikecastState, setStrikecastState] = useState<{ loading: boolean; error: string | null }>({ loading: false, error: null });
  const [valuePicksByDate, setValuePicksByDate] = useState<Record<string, ValuePick[]>>({});
  const [valuePicksUpdatedAt, setValuePicksUpdatedAt] = useState<string | null>(null);
  const [comparisonMasks, setComparisonMasks] = useState<ComparisonMaskSet | null>(null);
  const [oddsEngine, setOddsEngine] = useState<OddsEnginePayload | null>(null);
  const [oddsPredictions, setOddsPredictions] = useState<OddsPredictionsPayload | null>(null);
  const [oddsHistory, setOddsHistory] = useState<OddsHistoryPayload | null>(null);
  const [calibratorBuckets, setCalibratorBuckets] = useState<{ key: string; n: number; winRate: number }[] | null>(null);
  const [cardEvidence, setCardEvidence] = useState<CardEvidencePayload | null>(null);
  const [playerPerformance, setPlayerPerformance] = useState<PlayerPerformancePayload | null>(null);
  const [historyIndex, setHistoryIndex] = useState<HistoryIndex | null>(null);
  const [historyCache, setHistoryCache] = useState<Record<string, HistorySnapshot>>({});
  const historyFetchDate = selectedGameDate < mexicoIsoDate() ? selectedGameDate : null;
  useEffect(() => {
    if (!historyFetchDate) return;
    if (historyCache[historyFetchDate]) return;
    if (historyIndex && !historyIndex.dates.includes(historyFetchDate)) return;
    const controller = new AbortController();
    fetch(`/data/history/${historyFetchDate}.json`, { signal: controller.signal, cache: 'no-store' })
      .then(r => r.ok ? r.json() as Promise<HistorySnapshot> : null)
      .then(snap => { if (snap) setHistoryCache(prev => ({ ...prev, [historyFetchDate]: snap })); })
      .catch(() => {});
    return () => controller.abort();
  }, [historyFetchDate, historyIndex, historyCache]);

  const refresh = async () => {
    setLoadError('');
    try {
      const todayDate = mexicoIsoDate();
      const tomorrowDate = mexicoIsoDate(1);
      const previousDates = [-1, -2, -3, -4, -5, -6, -7].map(offset => mexicoIsoDate(offset));
      const [statsResponse, seasonL10Response, storedOddsResponse, staticOddsResponse, telegramResponse, todayResponse, tomorrowResponse, previousResponses] = await Promise.all([
        fetchWithTimeout('/data/stats.json', { cache: 'no-store' }, 15_000),
        fetchWithTimeout('/data/season-l10-audit.json', { cache: 'no-store' }, 8_000).catch(() => null),
        fetchWithTimeout(`/api/odds?startDate=${todayDate}&endDate=${tomorrowDate}`, { cache: 'no-store' }, 12_000).catch(() => null),
        fetchWithTimeout('/data/current-odds.json', { cache: 'no-store' }, 8_000).catch(() => null),
        fetchWithTimeout('/api/telegram', { cache: 'no-store' }, 8_000).catch(() => null),
        fetchWithTimeout(`/api/today?date=${todayDate}`, { cache: 'no-store' }, 20_000).catch(() => null),
        fetchWithTimeout(`/api/today?date=${tomorrowDate}`, { cache: 'no-store' }, 15_000).catch(() => null),
        Promise.all(previousDates.map(date => fetchWithTimeout(`/api/today?date=${date}`, { cache: 'no-store' }, 15_000).catch(() => null))),
      ]);
      if (!statsResponse.ok) throw new Error(`stats.json respondió ${statsResponse.status}`);
      setStats(await statsResponse.json() as StatsData);
      setSeasonL10Audit(seasonL10Response?.ok ? await seasonL10Response.json() as SeasonL10Audit : null);
      const storedOdds = storedOddsResponse?.ok ? await storedOddsResponse.json() as OddsData : null;
      const staticOdds = staticOddsResponse?.ok ? await staticOddsResponse.json() as OddsData : null;
      setOdds(storedOdds?.games.length ? storedOdds : staticOdds);
      setTelegramConfigured(telegramResponse?.ok ? Boolean((await telegramResponse.json() as { configured?: boolean }).configured) : false);
      let todayPayload: ScheduleData;
      if (todayResponse?.ok) {
        todayPayload = await todayResponse.json() as ScheduleData;
      } else {
        const fallback = await fetchWithTimeout('/data/today-snapshot.json', { cache: 'no-store' }, 8_000);
        if (!fallback.ok) throw new Error(`respaldo del calendario respondió ${fallback.status}`);
        todayPayload = await fallback.json() as ScheduleData;
      }
      const tomorrowPayload = tomorrowResponse?.ok
        ? await tomorrowResponse.json() as ScheduleData
        : { sourceState: 'UNAVAILABLE', source: 'MLB StatsAPI', retrievedAt: new Date().toISOString(), selectedDate: tomorrowDate, games: [] };
      const previousPayloads = await Promise.all(previousResponses.filter((response): response is Response => Boolean(response?.ok)).map(response => response.json() as Promise<ScheduleData>));
      const archive = Object.fromEntries([...previousPayloads, todayPayload, tomorrowPayload].map(payload => [payload.selectedDate ?? payload.games[0]?.officialDate, payload]).filter(([date]) => Boolean(date))) as Record<string, ScheduleData>;
      setSchedule(todayPayload);
      setTomorrowSchedule(tomorrowPayload);
      setScheduleArchive(archive);
      setPreviousGames(previousPayloads.flatMap(payload => payload.games));
      setSelectedGameDate(current => archive[current] ? current : todayDate);
    } catch (error) {
      console.error('No se pudo iniciar StatsMLB', error);
      setLoadError(error instanceof Error ? error.message : 'Error inesperado al cargar los datos');
    }
  };

  const updateDailyData = async () => {
    setUpdatingDaily(true);
    setRefreshFeedback(null);
    setRefreshProgress({ kind: 'running', percent: 1, stage: 'Preparando', message: 'Conectando con las fuentes de datos' });
    try {
      const result = await requestDailyRefresh(progress => setRefreshProgress({ ...progress, kind: 'running' }));
      setRefreshProgress({ kind: 'running', percent: 100, stage: 'Recargando', message: 'Mostrando los datos recién guardados' });
      await refresh();
      const detail = `${result.yesterdayGames} de ayer reconciliados · ${result.todayGames} de hoy · ${result.tomorrowGames} de mañana · ${result.finalGames} finalizados guardados · momios en ${result.oddsEvents} juegos`;
      setRefreshFeedback({ kind: result.status === 'partial' ? 'warning' : 'success', text: result.status === 'partial' ? `${detail}. Pendiente: ${result.errors.join(' · ')}` : detail });
      setRefreshProgress({
        kind: result.status === 'partial' ? 'error' : 'success',
        percent: 100,
        stage: result.status === 'partial' ? 'Actualización parcial' : 'Actualización completa',
        message: result.status === 'partial' ? 'Se conservaron los datos anteriores de la fuente pendiente' : 'La pantalla ya muestra los resultados, juegos y momios guardados',
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : 'No se pudo actualizar';
      setRefreshFeedback({ kind: 'error', text: message });
      setRefreshProgress({ kind: 'error', percent: 100, stage: 'No se completó', message });
    } finally {
      setUpdatingDaily(false);
    }
  };

  useEffect(() => {
    const timer = window.setTimeout(() => { void refresh(); }, 0);
    return () => window.clearTimeout(timer);
  }, []);
  useEffect(() => {
    if (selectedGameDate >= mexicoIsoDate() || recentWalkforward[selectedGameDate]) return;
    const controller = new AbortController();
    setHistoricalPredictionsLoading(true);
    fetch(`/data/walkforward-${selectedGameDate.slice(0, 4)}.json`, { signal: controller.signal, cache: 'no-store' })
      .then(response => {
        if (!response.ok) throw new Error(`walkforward histórico respondió ${response.status}`);
        return response.json() as Promise<{ days: Record<string, FrozenWalkforwardDay> }>;
      })
      .then(payload => setRecentWalkforward(current => ({ ...current, ...payload.days })))
      .catch(error => {
        if (error instanceof DOMException && error.name === 'AbortError') return;
        console.error('No se pudieron cargar las predicciones históricas', error);
      })
      .finally(() => setHistoricalPredictionsLoading(false));
    return () => controller.abort();
  }, [recentWalkforward, selectedGameDate]);
  useEffect(() => {
    const controller = new AbortController();
    Promise.all([
      fetch('/data/odds-walkforward.json', { signal: controller.signal, cache: 'no-store' }).then(r => r.ok ? r.json() as Promise<OddsEnginePayload> : null).catch(() => null),
      fetch('/data/odds-predictions-today.json', { signal: controller.signal, cache: 'no-store' }).then(r => r.ok ? r.json() as Promise<OddsPredictionsPayload> : null).catch(() => null),
      fetch('/data/odds-history.json', { signal: controller.signal, cache: 'no-store' }).then(r => r.ok ? r.json() as Promise<OddsHistoryPayload> : null).catch(() => null),
      fetch('/data/card-evidence.json', { signal: controller.signal, cache: 'no-store' }).then(r => r.ok ? r.json() as Promise<CardEvidencePayload> : null).catch(() => null),
      fetch('/data/player-performance.json', { signal: controller.signal, cache: 'no-store' }).then(r => r.ok ? r.json() as Promise<PlayerPerformancePayload> : null).catch(() => null),
      fetch('/data/history-index.json', { signal: controller.signal, cache: 'no-store' }).then(r => r.ok ? r.json() as Promise<HistoryIndex> : null).catch(() => null),
    ]).then(([engine, preds, history, evidence, performance, historyIdx]) => { if (engine) setOddsEngine(engine); if (preds) setOddsPredictions(preds); if (history) setOddsHistory(history); if (evidence) setCardEvidence(evidence); if (performance) setPlayerPerformance(performance); if (historyIdx) setHistoryIndex(historyIdx); });
    return () => controller.abort();
  }, []);
  useEffect(() => {
    if (calibratorBuckets) return;
    const controller = new AbortController();
    fetch('/data/stacked-calibrator.json', { signal: controller.signal, cache: 'no-store' })
      .then(response => response.ok ? response.json() as Promise<{ buckets: { key: string; n: number; winRate: number }[] }> : null)
      .then(payload => { if (payload?.buckets) setCalibratorBuckets(payload.buckets); })
      .catch(error => { if (!(error instanceof DOMException && error.name === 'AbortError')) console.error('No se pudo cargar stacked-calibrator.json', error); });
    return () => controller.abort();
  }, [calibratorBuckets]);
  useEffect(() => {
    if (comparisonMasks) return;
    const controller = new AbortController();
    fetch('/data/walkforward.json', { signal: controller.signal, cache: 'no-store' })
      .then(response => response.ok ? response.json() as Promise<{ defaultMask?: number; bestFifteenGameAccuracyMask?: number; bestWinningDaysMask?: number; combinations?: { mask: number; accuracy: number; seasons?: Record<string, { accuracy: number }> }[] }> : null)
      .then(payload => {
        if (!payload) return;
        const topAccuracy = payload.combinations?.length ? [...payload.combinations].sort((a, b) => b.accuracy - a.accuracy)[0].mask : 0;
        const best2026 = payload.combinations?.length
          ? [...payload.combinations].sort((a, b) => (b.seasons?.['2026']?.accuracy ?? 0) - (a.seasons?.['2026']?.accuracy ?? 0))[0].mask
          : topAccuracy;
        setComparisonMasks({
          base: payload.defaultMask ?? 0,
          topAccuracy,
          fifteenGames: payload.bestFifteenGameAccuracyMask ?? topAccuracy,
          winningDays: payload.bestWinningDaysMask ?? topAccuracy,
          best2026,
        });
      })
      .catch(error => { if (!(error instanceof DOMException && error.name === 'AbortError')) console.error('No se pudo cargar walkforward.json', error); });
    return () => controller.abort();
  }, [comparisonMasks]);
  useEffect(() => {
    const controller = new AbortController();
    const load = (isFirst: boolean) => {
      if (isFirst) setStrikecastState({ loading: true, error: null });
      fetch(`/api/strikecast?date=${comparisonDate}`, { signal: controller.signal, cache: 'no-store' })
        .then(async response => {
          const payload = await response.json() as { games?: StrikecastPick[]; error?: string };
          if (!response.ok) throw new Error(payload.error ?? `Strikecast respondió ${response.status}`);
          setStrikecastByDate(current => ({ ...current, [comparisonDate]: payload.games ?? [] }));
          setStrikecastState({ loading: false, error: null });
        })
        .catch(error => {
          if (error instanceof DOMException && error.name === 'AbortError') return;
          setStrikecastState({ loading: false, error: error instanceof Error ? error.message : 'No se pudo consultar StrikeCast' });
        });
    };
    load(true);
    const interval = window.setInterval(() => load(false), 5 * 60 * 1000);
    return () => { controller.abort(); window.clearInterval(interval); };
  }, [comparisonDate]);
  useEffect(() => {
    const controller = new AbortController();
    const load = () => {
      fetch(`/api/value?date=${comparisonDate}`, { signal: controller.signal, cache: 'no-store' })
        .then(async response => {
          const payload = await response.json() as { picks?: ValuePick[]; updatedAt?: string | null; error?: string };
          if (!response.ok) return;
          setValuePicksByDate(current => ({ ...current, [comparisonDate]: payload.picks ?? [] }));
          if (payload.updatedAt) setValuePicksUpdatedAt(payload.updatedAt);
        })
        .catch(error => { if (!(error instanceof DOMException && error.name === 'AbortError')) console.error('No se pudo cargar Value Model', error); });
    };
    load();
    // Refresco automatico cada 5 min mientras la pestana este viva; captura
    // reentrenos del scheduled task sin obligar al usuario a apretar F5.
    const interval = window.setInterval(load, 5 * 60 * 1000);
    return () => { controller.abort(); window.clearInterval(interval); };
  }, [comparisonDate]);
  useEffect(() => {
    if (recentWalkforward[comparisonDate]) return;
    if (comparisonDate >= mexicoIsoDate()) return;
    const controller = new AbortController();
    fetch(`/data/walkforward-${comparisonDate.slice(0, 4)}.json`, { signal: controller.signal, cache: 'no-store' })
      .then(response => response.ok ? response.json() as Promise<{ days: Record<string, FrozenWalkforwardDay> }> : null)
      .then(payload => { if (payload) setRecentWalkforward(current => ({ ...current, ...payload.days })); })
      .catch(error => { if (!(error instanceof DOMException && error.name === 'AbortError')) console.error('No se pudo cargar walkforward histórico', error); });
    return () => controller.abort();
  }, [comparisonDate, recentWalkforward]);
  const rankedRows = useMemo(() => {
    if (!stats) return [];
    const key = rankingView === 'sweep' ? 'goodToBad' : rankingView === 'vulnerable' ? 'mostVulnerable' : rankingView;
    const rows = stats.rankings[key] ?? [];
    return rows;
  }, [stats, rankingView]);

  if (!stats || !schedule) return <main className="loading-screen"><div className="loading-mark"><Baseball size={28} /><span /></div><p>{loadError ? 'No se pudo preparar StatsMLB' : 'Preparando StatsMLB…'}</p>{loadError && <><small>{loadError}</small><button className="primary-button" onClick={() => void refresh()}>Reintentar</button></>}</main>;

  const scenario = stats.scenarios;
  const todayDate = mexicoIsoDate();
  const tomorrowDate = mexicoIsoDate(1);
  const dateChoices = Array.from({ length: 9 }, (_, index) => mexicoIsoDate(index - 7));
  const displaySchedule = scheduleArchive[selectedGameDate] ?? (selectedGameDate === tomorrowDate ? tomorrowSchedule : schedule) ?? schedule;
  const selectedDate = displaySchedule.selectedDate ?? selectedGameDate;
  const isHistoricalSlate = selectedDate < todayDate;
  const historicalSnapshot: HistorySnapshot | undefined = isHistoricalSlate ? historyCache[selectedDate] : undefined;
  const activeMask = activeFactorMask(active);
  const frozenGames = new Map((recentWalkforward[selectedDate]?.games ?? []).map(game => [game.gamePk, game]));
  const slateLabel = new Intl.DateTimeFormat('es-MX', { timeZone: 'UTC', weekday: 'long', day: 'numeric', month: 'long' }).format(new Date(`${selectedDate}T12:00:00Z`));
  const slateTitle = isHistoricalSlate ? 'Juegos anteriores' : selectedDate === tomorrowDate ? 'Juegos de mañana' : 'Juegos de hoy';
  const copyPicksList = async () => {
    if (!displaySchedule.games.length) return;
    const dateLabel = new Intl.DateTimeFormat('es-ES', { timeZone: 'UTC', day: 'numeric', month: 'long', year: 'numeric' }).format(new Date(`${selectedDate}T12:00:00Z`));
    const rows: string[] = ['⚾ STRIKECAST — PICKS DEL DÍA', '', `📅 ${dateLabel}`, '', '#\tHora\tPartido\tPick StrikeCast\tProb. modelo'];
    displaySchedule.games.forEach((game, index) => {
      let homeP = 0.5;
      if (isHistoricalSlate) {
        const frozen = frozenPredictionForMask(frozenGames.get(game.gamePk), activeMask);
        homeP = frozen?.homeProbability ?? 0.5;
      } else {
        homeP = estimateGame(game, stats, active, findCurrentOdds(game, odds)).homeProbability;
      }
      const favoriteHome = homeP >= 0.5;
      const favProb = favoriteHome ? homeP : 1 - homeP;
      const favTeam = favoriteHome ? game.home : game.away;
      const gap = Math.abs(homeP - 0.5);
      const icon = gap >= 0.10 ? '🟢' : gap >= 0.05 ? '🔵' : '🔴';
      const shorten = (name: string) => name.split(' ').pop() ?? name;
      const awayShort = `${game.away.team} ${shorten(game.away.name)}`;
      const homeShort = `${game.home.team} ${shorten(game.home.name)}`;
      rows.push(`${index + 1}\t${gameTime12(game.gameDate)}\t${awayShort} vs ${homeShort}\t${icon} ${favTeam.team}\t${Math.round(favProb * 100)}%`);
    });
    const text = rows.join('\n');
    try {
      await navigator.clipboard.writeText(text);
      setCopyFeedback({ kind: 'success', text: 'Lista copiada al portapapeles.' });
    } catch {
      setCopyFeedback({ kind: 'error', text: 'No se pudo copiar. Copia manualmente desde la consola.' });
      console.log(text);
    }
    window.setTimeout(() => setCopyFeedback(null), 3500);
  };
  return (
    <DashboardTabs
      cutoff={stats.cutoffDate}
      combination={FACTORS.filter(factor => active[factor.key]).map(factor => factor.label.replace(' (prueba)', '')).join(' · ')}
      telegramConfigured={telegramConfigured}
      updating={updatingDaily}
      onRefresh={() => void updateDailyData()}
      source={<SourceBadge schedule={schedule} />}
      feedback={refreshProgress ? <div className={`refresh-progress ${refreshProgress.kind}`} role="status"><div className="refresh-progress-head"><strong>{refreshProgress.stage}</strong><span>{refreshProgress.percent}%</span></div><div className="refresh-progress-track" role="progressbar" aria-label="Progreso de actualización" aria-valuemin={0} aria-valuemax={100} aria-valuenow={refreshProgress.percent}><i style={{ width: `${refreshProgress.percent}%` }} /></div><small>{refreshFeedback?.text ?? refreshProgress.message}</small></div> : null}
      metrics={[
        { label: 'Juegos analizados', value: stats.coverage.final_regular_games.toLocaleString('es-MX'), detail: 'Histórico de temporada regular' },
        { label: 'Series de tres juegos', value: stats.seriesSummary.eligible_three_game_series.toLocaleString('es-MX'), detail: `${stats.seriesSummary.swept_three_game_series.toLocaleString('es-MX')} barridas detectadas` },
        { label: 'Victoria tras barrer', value: pct(stats.afterSweepSummary.next_game_win_rate), detail: 'En el siguiente partido' },
        { label: 'Rebote tras ser barridos', value: pct(stats.afterSweptSummary.next_game_win_rate), detail: 'Victoria en el siguiente partido' },
      ]}
    >

      <section className="section" id="hoy">
        <div className="section-heading split-heading"><div><span className="eyebrow"><CalendarDays size={14} /> {slateLabel}</span><h2>{slateTitle}</h2><p>{displaySchedule.games.length} encuentros · horarios de Ciudad de México{isHistoricalSlate ? ' · predicciones congeladas antes de jugar' : ''}</p></div><div className="model-proof"><CircleGauge size={22} /><div><strong>{pct(stats.model.metrics.accuracy)}</strong><span>acierto en prueba 2026 · {stats.model.metrics.holdoutGames.toLocaleString('es-MX')} juegos fuera de muestra</span></div></div></div>
        <div className="daily-controls"><div className="date-slate-tabs" role="tablist" aria-label="Seleccionar fecha de juegos">{dateChoices.map(date => { const payload = scheduleArchive[date] ?? (date === todayDate ? schedule : date === tomorrowDate ? tomorrowSchedule : undefined); const label = date === todayDate ? 'Hoy' : date === tomorrowDate ? 'Mañana' : shortDate(date); return <button type="button" role="tab" aria-selected={selectedDate === date} className={selectedDate === date ? 'selected' : ''} key={date} onClick={() => setSelectedGameDate(date)}><span>{label}</span><b>{payload?.games.length ?? 0}</b></button>; })}</div><button className="daily-refresh-button" onClick={() => void updateDailyData()} disabled={updatingDaily}><RefreshCw size={15} className={updatingDaily ? 'spin' : ''} />{updatingDaily ? 'Consultando fuentes…' : 'Actualizar resultados, juegos y momios'}</button><button className="daily-copy-button" onClick={() => void copyPicksList()} disabled={!displaySchedule.games.length} title="Copiar la lista de picks del día"><ClipboardList size={15} />Copiar picks</button></div>
        {copyFeedback && <div className={`refresh-feedback ${copyFeedback.kind === 'success' ? 'success' : 'error'}`}><ClipboardList size={15} /><span>{copyFeedback.text}</span></div>}
        <div className="market-source-bar"><div><span className={odds?.sourceState === 'REAL DATA' ? 'real' : 'missing'}>{odds?.sourceState === 'REAL DATA' ? 'REAL DATA' : 'SIN DATOS'}</span><strong>Mejores momios disponibles</strong><small>{odds ? `${odds.games.length} juegos · ${odds.selectionRule}` : 'No se pudieron cargar momios'}</small></div><b className={telegramConfigured ? 'connected' : 'pending'}><Send size={13} /> {telegramConfigured ? 'Telegram conectado' : 'Telegram pendiente'}</b></div>
        <div className="factor-toolbar"><span>Factores activos</span>{FACTORS.map((factor) => <button key={factor.key} type="button" aria-pressed={active[factor.key]} className={active[factor.key] ? 'factor-active' : ''} title={factor.detail} onClick={() => setActive(current => ({ ...current, [factor.key]: !current[factor.key] }))}>{active[factor.key] && <CheckCircle2 size={14} />}{factor.label}</button>)}</div>
        {isHistoricalSlate && historicalPredictionsLoading && <div className="historical-predictions-loading" role="status"><RefreshCw size={15} className="spin" /><span>Cargando las predicciones congeladas de {slateLabel}…</span></div>}
        {displaySchedule.games.length ? <div className="games-grid">{displaySchedule.games.map((game, index) => <GameCard key={game.gamePk} game={game} stats={stats} active={active} odds={isHistoricalSlate ? undefined : findCurrentOdds(game, odds)} pickNumber={index + 1} telegramConfigured={telegramConfigured} historical={isHistoricalSlate} frozenPrediction={isHistoricalSlate ? frozenPredictionForMask(frozenGames.get(game.gamePk), activeMask) : undefined} comparisonMasks={comparisonMasks} strikecastPick={strikecastByDate[selectedDate]?.find(pick => pick.gamePk === game.gamePk)} valuePick={valuePicksByDate[selectedDate]?.find(pick => pick.gamePk === game.gamePk)} oddsHistory={oddsHistory} evidenceProfiles={cardEvidence?.profiles ?? []} evidenceSignals={cardEvidence?.signals ?? []} playerPerformance={playerPerformance} historicalSnapshot={historicalSnapshot} />)}</div> : <div className="empty-state"><CalendarDays size={28} /><h3>No hay juegos disponibles</h3><p>El calendario no devolvió encuentros para {slateLabel}.</p></div>}
      </section>

      <div data-section="comparacion"><ComparisonSection
        stats={stats}
        odds={odds}
        scheduleArchive={scheduleArchive}
        schedule={schedule}
        tomorrowSchedule={tomorrowSchedule}
        recentWalkforward={recentWalkforward}
        comparisonDate={comparisonDate}
        onChangeDate={setComparisonDate}
        strikecast={strikecastByDate[comparisonDate] ?? []}
        strikecastLoading={strikecastState.loading}
        strikecastError={strikecastState.error}
        valuePicks={valuePicksByDate[comparisonDate] ?? []}
        valuePicksUpdatedAt={valuePicksUpdatedAt}
        masks={comparisonMasks}
        oddsHistory={oddsHistory}
        calibratorBuckets={calibratorBuckets}
        playerPerformance={playerPerformance}
      />

      </div>

      <div data-section="rendimiento"><WalkforwardCalendar
        active={active}
        onToggle={(key) => setActive(current => ({ ...current, [key]: !current[key] }))}
        onSetActive={(keys) => setActive({ localia: keys.includes('localia'), strength: keys.includes('strength'), recent: keys.includes('recent'), runDiff: keys.includes('runDiff'), series: keys.includes('series'), marketSchedule: keys.includes('marketSchedule'), bestPlayersTest: keys.includes('bestPlayersTest'), coachRotationTest: keys.includes('coachRotationTest'), rotationQualityTest: keys.includes('rotationQualityTest'), lineupFatigueTest: keys.includes('lineupFatigueTest'), opponentFormTest: keys.includes('opponentFormTest') })}
      />

      </div>

      {seasonL10Audit && <div data-section="auditoria-temporada-l10"><SeasonL10AuditSection audit={seasonL10Audit} /></div>}

      <div data-section="barridas-walkforward"><SeriesSweepAuditSection stats={stats} schedule={schedule} previousGames={previousGames} active={active} odds={odds} /></div>

      <section className="section audit-section" id="auditoria-juego-anterior"><div className="section-heading split-heading"><div><span className="eyebrow"><FlaskConical size={14} /> Auditoría juego anterior</span><h2>LOB, hits y jonrones</h2><p>Se probaron señales del partido inmediatamente anterior, separando si el equipo ganó o perdió. Cada cifra es walk-forward mensual.</p></div><div className="audit-baseline"><strong>{pct(stats.previousGameAudit.baseline.accuracy, 2)}</strong><span>base Mercado + descanso</span><small>{stats.previousGameAudit.baseline.correct}/{stats.previousGameAudit.games} aciertos</small></div></div><div className="audit-candidate-grid">{stats.previousGameAudit.candidates.map(candidate => <article key={candidate.name}><div><span>{candidate.status}</span><b className={candidate.deltaPoints > 0 ? 'positive' : 'negative'}>{candidate.deltaPoints > 0 ? '+' : ''}{candidate.deltaPoints.toFixed(2)} pts</b></div><h3>{candidate.name}</h3><strong>{pct(candidate.accuracy, 2)}</strong><small>2026: {pct(candidate.accuracy2026, 2)} · {candidate.correct}/{stats.previousGameAudit.games}</small><p>{candidate.reason}</p></article>)}</div><div className="audit-descriptive"><div><span>ESCENARIO DEL JUEGO PREVIO</span><b>Siguiente partido</b><b>2026</b><b>Muestra</b></div>{stats.previousGameAudit.descriptive.map(row => <div key={row.label}><strong>{row.label}</strong><b>{pct(row.nextWinRate)}</b><b>{pct(row.nextWinRate2026)}</b><span>{row.games.toLocaleString('es-MX')}</span></div>)}</div><p className="audit-conclusion"><ShieldCheck size={16} /> Conclusión: ninguna señal de LOB/hits/HR se activó en el modelo principal. La diferencia cruda de hits fue prometedora, pero no mejoró 2026 y su intervalo de incertidumbre todavía incluye cero.</p></section>

      <section className="section audit-section stars-audit" id="auditoria-mejores-jugadores"><div className="section-heading split-heading"><div><span className="eyebrow"><Sparkles size={14} /> Auditoría “carne al asador”</span><h2>¿Qué pasa después de usar a los mejores?</h2><p>Las estrellas se identifican antes de cada juego mediante producción acumulada suavizada y frecuencia de titularidad. Se mide su uso en ese partido y únicamente el resultado del siguiente.</p></div><div className="audit-baseline"><strong>{pct(stats.bestPlayersAudit.baseline.accuracy, 2)}</strong><span>base Mercado + descanso</span><small>{stats.bestPlayersAudit.coverage.teamGames.toLocaleString('es-MX')} actuaciones de equipo</small></div></div><div className="audit-candidate-grid stars-candidate-grid">{stats.bestPlayersAudit.candidates.map(candidate => <article key={candidate.name}><div><span className={candidate.status === 'OBSERVACIÓN' ? 'analysis-watch' : ''}>{candidate.status}</span><b className={candidate.deltaPoints > 0 ? 'positive' : 'negative'}>{candidate.deltaPoints > 0 ? '+' : ''}{candidate.deltaPoints.toFixed(2)} pts</b></div><h3>{candidate.name}</h3><strong>{pct(candidate.accuracy, 2)}</strong><small>2026: {pct(candidate.accuracy2026, 2)} ({candidate.delta2026Points > 0 ? '+' : ''}{candidate.delta2026Points.toFixed(2)} pts)</small><p>{candidate.reason}</p></article>)}</div><div className="audit-descriptive"><div><span>USO EN EL JUEGO ANTERIOR</span><b>Siguiente partido</b><b>2026</b><b>Muestra</b></div>{stats.bestPlayersAudit.descriptive.map(row => <div key={row.label}><strong>{row.label}</strong><b>{pct(row.nextWinRate)}</b><b>{pct(row.nextWinRate2026)}</b><span>{row.games.toLocaleString('es-MX')}</span></div>)}</div><p className="audit-conclusion audit-watch"><Info size={16} /> {stats.bestPlayersAudit.recommendation}</p></section>

      <section className="section audit-section coach-audit" id="auditoria-rotacion-coach"><div className="section-heading split-heading"><div><span className="eyebrow"><RefreshCw size={14} /> Auditoría de decisiones</span><h2>Rotación del coach después del resultado</h2><p>Compara titulares, orden al bate y cambios en el top 4 respecto al juego anterior. La alineación del partido analizado se trata como información previa al primer lanzamiento.</p></div><div className="audit-baseline"><strong>{pct(stats.coachRotationAudit.candidate.accuracy, 2)}</strong><span>Rotación coach (prueba)</span><small>{stats.coachRotationAudit.candidate.correct.toLocaleString('es-MX')}/{stats.coachRotationAudit.candidate.games.toLocaleString('es-MX')} aciertos</small></div></div><div className="coach-proof-grid"><article><span>MEJORA WALK-FORWARD</span><strong>+{stats.coachRotationAudit.candidate.deltaPoints.toFixed(2)} pts</strong><small>Base {pct(stats.coachRotationAudit.candidate.baseline.accuracy, 2)}</small></article><article><span>RESULTADO 2026</span><strong>{pct(stats.coachRotationAudit.candidate.seasons['2026']?.accuracy, 2)}</strong><small>+{stats.coachRotationAudit.candidate.delta2026Points.toFixed(2)} pts</small></article><article><span>INTERVALO 95%</span><strong>{stats.coachRotationAudit.candidate.ci95Points[0].toFixed(2)} a +{stats.coachRotationAudit.candidate.ci95Points[1].toFixed(2)}</strong><small>puntos de accuracy</small></article><article><span>CORTES MENSUALES</span><strong>{stats.coachRotationAudit.candidate.foldsBetter}-{stats.coachRotationAudit.candidate.foldsWorse}</strong><small>mejoró vs empeoró</small></article></div><div className="audit-descriptive"><div><span>DESPUÉS DE PERDER</span><b>Siguiente juego</b><b>2026</b><b>Muestra</b></div>{stats.coachRotationAudit.descriptive.filter(row => row.previousResult === 'loss').map(row => <div key={row.label}><strong>{row.rotation}</strong><b>{pct(row.winRate)}</b><b>{pct(row.winRate2026)}</b><span>{row.games.toLocaleString('es-MX')}</span></div>)}</div><p className="audit-conclusion coach-watch"><RefreshCw size={16} /> {stats.coachRotationAudit.recommendation}</p></section>

      <section className="section audit-section coach-audit" id="auditoria-calidad-rotacion"><div className="section-heading split-heading"><div><span className="eyebrow"><Sparkles size={14} /> Auditoría de calidad del lineup</span><h2>No solo cuántos cambian: quién entra y quién sale</h2><p>Valora cada bateador con producción y frecuencia de titular acumuladas antes del juego. Después compara la calidad del lineup nuevo, los reemplazos, el núcleo retenido y el top 4.</p></div><div className="audit-baseline"><strong>{pct(stats.rotationQualityAudit.candidate.accuracy, 2)}</strong><span>Calidad rotación (prueba)</span><small>{stats.rotationQualityAudit.candidate.correct.toLocaleString('es-MX')}/{stats.rotationQualityAudit.candidate.games.toLocaleString('es-MX')} aciertos</small></div></div><div className="coach-proof-grid"><article><span>MEJORA WALK-FORWARD</span><strong>+{stats.rotationQualityAudit.candidate.deltaPoints.toFixed(2)} pts</strong><small>Base {pct(stats.rotationQualityAudit.candidate.baseline.accuracy, 2)}</small></article><article><span>RESULTADO 2026</span><strong>{pct(stats.rotationQualityAudit.candidate.seasons['2026']?.accuracy, 2)}</strong><small>+{stats.rotationQualityAudit.candidate.delta2026Points.toFixed(2)} pts</small></article><article><span>INTERVALO 95%</span><strong>+{stats.rotationQualityAudit.candidate.ci95Points[0].toFixed(2)} a +{stats.rotationQualityAudit.candidate.ci95Points[1].toFixed(2)}</strong><small>puntos de accuracy</small></article><article><span>JUNTO A ROTACIÓN COACH</span><strong>{pct(stats.rotationQualityAudit.combinedWithCoach.accuracy, 2)}</strong><small>+{stats.rotationQualityAudit.combinedWithCoach.deltaPoints.toFixed(2)} pts vs base</small></article></div><div className="audit-descriptive"><div><span>DESPUÉS DE PERDER</span><b>Siguiente juego</b><b>2026</b><b>Muestra</b></div>{stats.rotationQualityAudit.descriptive.map(row => <div key={row.label}><strong>{row.label}</strong><b>{pct(row.winRate)}</b><b>{pct(row.winRate2026)}</b><span>{row.games.toLocaleString('es-MX')}</span></div>)}</div><p className="audit-conclusion coach-watch"><Sparkles size={16} /> {stats.rotationQualityAudit.recommendation}</p></section>

      <section className="section audit-section coach-audit" id="auditoria-fatiga-lineup"><div className="section-heading split-heading"><div><span className="eyebrow"><Activity size={14} /> Auditoría de carga reciente</span><h2>Fatiga del lineup confirmado</h2><p>Mide, antes del primer lanzamiento, cuántas veces iniciaron los bateadores y cuántas apariciones de plato acumularon en las últimas 72 horas y siete días. También identifica descansos menores a 30 horas.</p></div><div className="audit-baseline"><strong>{pct(stats.lineupFatigueAudit.candidate.accuracy, 2)}</strong><span>Fatiga lineup (prueba)</span><small>{stats.lineupFatigueAudit.candidate.correct.toLocaleString('es-MX')}/{stats.lineupFatigueAudit.candidate.games.toLocaleString('es-MX')} aciertos</small></div></div><div className="coach-proof-grid"><article><span>MEJORA WALK-FORWARD</span><strong>+{stats.lineupFatigueAudit.candidate.deltaPoints.toFixed(2)} pts</strong><small>Base {pct(stats.lineupFatigueAudit.candidate.baseline.accuracy, 2)}</small></article><article><span>VERIFICACIÓN 2026</span><strong>{pct(stats.lineupFatigueAudit.candidate.seasons['2026']?.accuracy, 2)}</strong><small>+{stats.lineupFatigueAudit.candidate.delta2026Points.toFixed(2)} pts</small></article><article><span>MITADES 2026</span><strong>{stats.lineupFatigueAudit.candidate.h1_2026.deltaPoints >= 0 ? '+' : ''}{stats.lineupFatigueAudit.candidate.h1_2026.deltaPoints.toFixed(2)} / {stats.lineupFatigueAudit.candidate.h2_2026.deltaPoints >= 0 ? '+' : ''}{stats.lineupFatigueAudit.candidate.h2_2026.deltaPoints.toFixed(2)}</strong><small>H1 / H2 en puntos</small></article><article><span>INTERVALO 95%</span><strong>+{stats.lineupFatigueAudit.candidate.ci95Points[0].toFixed(2)} a +{stats.lineupFatigueAudit.candidate.ci95Points[1].toFixed(2)}</strong><small>{stats.lineupFatigueAudit.candidate.foldsBetter}-{stats.lineupFatigueAudit.candidate.foldsWorse} cortes mejoró/empeoró</small></article></div><div className="audit-descriptive"><div><span>CARGA DEL LINEUP</span><b>Ganó el juego</b><b>2026</b><b>Muestra</b></div>{stats.lineupFatigueAudit.descriptive.map(row => <div key={row.label}><strong>{row.label}</strong><b>{pct(row.winRate)}</b><b>{pct(row.winRate2026)}</b><span>{row.games.toLocaleString('es-MX')}</span></div>)}</div><p className="audit-conclusion coach-watch"><Activity size={16} /> {stats.lineupFatigueAudit.recommendation}</p></section>

      <section className="section pair-audit-section" id="auditoria-cruces-lineup"><div className="section-heading split-heading"><div><span className="eyebrow"><BarChart3 size={14} /> Auditoría de cada indicación</span><h2>Qué ocurre cuando se cruzan</h2><p>Separa los cruces exactos entre visitante y local: cambios, calidad y carga. “Histórico” dice quién ganó más; “accuracy WF” mide si esa indicación podía anticiparlo usando solo fechas anteriores.</p></div><div className="audit-baseline"><strong>{stats.indicationPairAudit.minimumPriorGames}</strong><span>antecedentes mínimos</span><small>Corte {stats.indicationPairAudit.cutoffDate}</small></div></div><div className="pair-audit-note"><ShieldCheck size={16} /><span>{stats.indicationPairAudit.method}</span></div><div className="pair-audit-grid"><PairIndicationTable signal={stats.indicationPairAudit.signals.coachRotation} /><PairIndicationTable signal={stats.indicationPairAudit.signals.rotationQuality} /><PairIndicationTable signal={stats.indicationPairAudit.signals.lineupFatigue} /></div></section>

      <section className="section audit-section coach-audit" id="auditoria-forma-rival"><div className="section-heading split-heading"><div><span className="eyebrow"><ShieldCheck size={14} /> Auditoría de dificultad reciente</span><h2>Forma ajustada por la calidad del rival</h2><p>Compara la fuerza previa de los oponentes enfrentados en los últimos 10 y 20 juegos. Cada rival se valora sólo con lo que se conocía antes de ese partido.</p></div><div className="audit-baseline"><strong>{pct(stats.opponentFormAudit.candidate.accuracy, 2)}</strong><span>Fatiga + Calidad + Forma vs rival</span><small>{stats.opponentFormAudit.candidate.correct.toLocaleString('es-MX')}/{stats.opponentFormAudit.candidate.games.toLocaleString('es-MX')} aciertos</small></div></div><div className="coach-proof-grid"><article><span>APORTE ADICIONAL</span><strong>+{stats.opponentFormAudit.candidate.deltaPoints.toFixed(2)} pts</strong><small>Referencia {pct(stats.opponentFormAudit.candidate.baseline.accuracy, 2)}</small></article><article><span>VERIFICACIÓN 2026</span><strong>{pct(stats.opponentFormAudit.candidate.seasons['2026']?.accuracy, 2)}</strong><small>+{stats.opponentFormAudit.candidate.delta2026Points.toFixed(2)} pts</small></article><article><span>MITADES 2026</span><strong>+{stats.opponentFormAudit.candidate.h1_2026.deltaPoints.toFixed(2)} / {stats.opponentFormAudit.candidate.h2_2026.deltaPoints >= 0 ? '+' : ''}{stats.opponentFormAudit.candidate.h2_2026.deltaPoints.toFixed(2)}</strong><small>H1 / H2 en puntos</small></article><article><span>CORTES MENSUALES</span><strong>{stats.opponentFormAudit.candidate.foldsBetter}-{stats.opponentFormAudit.candidate.foldsWorse}</strong><small>mejoró vs empeoró · IC95 {stats.opponentFormAudit.candidate.ci95Points[0].toFixed(2)} a +{stats.opponentFormAudit.candidate.ci95Points[1].toFixed(2)}</small></article></div><div className="audit-descriptive"><div><span>RENDIMIENTO AJUSTADO L10</span><b>Ganó el juego</b><b>2026</b><b>Muestra</b></div>{stats.opponentFormAudit.descriptive.map(row => <div key={row.label}><strong>{row.label}</strong><b>{pct(row.winRate)}</b><b>{pct(row.winRate2026)}</b><span>{row.games.toLocaleString('es-MX')}</span></div>)}</div><p className="audit-conclusion coach-watch"><ShieldCheck size={16} /> {stats.opponentFormAudit.recommendation}</p></section>

      <section className="section dark-section" id="rankings"><div className="section-heading"><span className="eyebrow"><Trophy size={14} /> 2023 a hoy</span><h2>Quién domina las series</h2><p>Ordena los equipos por capacidad de barrer, vulnerabilidad o respuesta en el siguiente partido.</p></div><div className="ranking-controls"><div className="segmented">{[['sweep', 'Mejor balance'], ['vulnerable', 'Más vulnerables'], ['afterSweep', 'Después de barrer'], ['afterSwept', 'Después de ser barridos']].map(([key, label]) => <button key={key} className={rankingView === key ? 'selected' : ''} onClick={() => setRankingView(key)}>{label}</button>)}</div><button className="sort-button" onClick={() => setAscending(value => !value)}><ArrowDownUp size={16} /> {ascending ? 'Menor a mayor' : 'Mayor a menor'}</button></div><RankingTable rows={rankedRows} view={rankingView} ascending={ascending} showAll={showAll} /><button className="show-all" onClick={() => setShowAll(value => !value)}>{showAll ? 'Mostrar solo 10' : 'Mostrar los 30 equipos'} <ChevronDown size={16} className={showAll ? 'rotate' : ''} /></button></section>

      <section className="section" id="escenarios"><div className="section-heading"><span className="eyebrow"><BarChart3 size={14} /> Cruces inmediatos</span><h2>Cuando el contexto se enfrenta</h2><p>Estos porcentajes describen lo ocurrido históricamente en el primer partido posterior; la muestra siempre se muestra junto al resultado.</p></div><div className="scenario-grid"><article className="scenario-card featured"><span className="scenario-number">01</span><div className="scenario-icon"><Sparkles size={20} /></div><h3>Ambos vienen de barrer</h3><p className="scenario-result">El local ganó <strong>{pct(scenario.bothSweep.homeWinRate)}</strong></p><div className="mini-bar"><span style={{ width: `${scenario.bothSweep.homeWinRate * 100}%` }} /></div><footer>{scenario.bothSweep.homeWins}–{scenario.bothSweep.awayWins} · {scenario.bothSweep.games} juegos</footer></article><article className="scenario-card"><span className="scenario-number">02</span><div className="scenario-icon"><Activity size={20} /></div><h3>Ambos fueron barridos</h3><p className="scenario-result">El visitante ganó <strong>{pct(1 - scenario.bothSwept.homeWinRate)}</strong></p><div className="mini-bar muted"><span style={{ width: `${(1 - scenario.bothSwept.homeWinRate) * 100}%` }} /></div><footer>{scenario.bothSwept.awayWins}–{scenario.bothSwept.homeWins} · {scenario.bothSwept.games} juegos</footer></article><article className="scenario-card wide"><span className="scenario-number">03</span><div className="scenario-icon"><Home size={20} /></div><h3>Barrido 0–3 vs perdedor 1–2</h3><p className="scenario-result">El equipo que perdió 1–2 ganó <strong>{pct(1 - scenario.sweptVsLost12.sweptWinRate)}</strong></p><div className="split-stats"><div><strong>{pct(scenario.sweptVsLost12.sweptAtHome.wins / scenario.sweptVsLost12.sweptAtHome.games)}</strong><span>Barrido jugando en casa</span></div><div><strong>{pct(scenario.sweptVsLost12.sweptAway.wins / scenario.sweptVsLost12.sweptAway.games)}</strong><span>Barrido jugando fuera</span></div></div><footer>{scenario.sweptVsLost12.games} juegos · la localía explica más que el 0–3 previo</footer></article></div></section>

      <section className="section lab-section" id="laboratorio"><div className="section-heading"><span className="eyebrow"><FlaskConical size={14} /> Hoja de ruta analítica</span><h2>Más señales para considerar</h2><p>El modelo actual solo usa datos que están completos y comparables. Estas capas pueden añadirse con etiqueta de procedencia y validación cronológica.</p></div><div className="analysis-grid">{stats.additionalAnalyses.map((item, index) => <article key={item.name} className="analysis-card"><span className="analysis-index">0{index + 1}</span><div><span className={`analysis-status ${item.status === 'Exploratorio' ? 'explore' : ''}`}>{item.status}</span><h3>{item.name}</h3><p>{item.detail}</p></div></article>)}</div><div className="method-card"><div className="method-icon"><Info size={21} /></div><div><h3>Cómo se forma el porcentaje</h3><p>La combinación activa recomendada usa localía, forma L10, diferencial de carreras, consenso de mercado disponible antes del juego y días de descanso. Las 2,047 combinaciones se reentrenan de forma independiente en cada corte mensual, sin resultados futuros; los cinco factores marcados como “prueba” permanecen apagados por defecto.</p></div><div className="method-metrics"><span><strong>{pct(stats.model.metrics.accuracy)}</strong> acierto 2026</span><span><strong>{stats.model.metrics.brier.toFixed(3)}</strong> Brier</span><span><strong>{pct(stats.model.metrics.homeBaselineAccuracy)}</strong> base local</span></div></div></section>

      {(oddsEngine || oddsPredictions) && (
        <section className="section" id="motor-momios">
          <div className="section-heading split-heading">
            <div>
              <span className="eyebrow"><CircleGauge size={14} /> Motor de momios · prueba</span>
              <h2>Walk-forward sobre las líneas de cierre</h2>
              <p>Cada renglón es un juego visto solo con la historia previa de ese equipo. El modelo mezcla el último momio con la media móvil de sus últimos 5 partidos y compara contra el precio actual del mercado. Delta grande = mayor desacuerdo entre modelo simple y books.</p>
            </div>
            {oddsEngine && (
              <div className="audit-baseline">
                <strong>{Math.round((oddsEngine.overall.models.blend?.direction_accuracy ?? 0) * 100)}%</strong>
                <span>acierto de dirección (blend)</span>
                <small>MAE {oddsEngine.overall.models.blend?.mae_american ?? '—'} ¢ · {oddsEngine.overall.games.toLocaleString('es-MX')} juegos</small>
              </div>
            )}
          </div>
          {oddsEngine && (
            <div className="coach-proof-grid" style={{ marginBottom: 18 }}>
              {['naive', 'roll5', 'roll10', 'blend'].map(model => {
                const stat = oddsEngine.overall.models[model];
                if (!stat) return null;
                const label = model === 'naive' ? 'Último momio' : model === 'roll5' ? 'Media 5' : model === 'roll10' ? 'Media 10' : 'Blend';
                return (
                  <article key={model}>
                    <span>{label.toUpperCase()}</span>
                    <strong>MAE {stat.mae_american ?? '—'}¢</strong>
                    <small>Dirección {stat.direction_accuracy != null ? `${Math.round(stat.direction_accuracy * 100)}%` : '—'} · n={stat.n.toLocaleString('es-MX')}</small>
                  </article>
                );
              })}
            </div>
          )}
          {oddsPredictions && oddsPredictions.games.length > 0 && (
            <div className="audit-descriptive">
              <div><span>SLATE DE HOY · MODELO {oddsPredictions.model.toUpperCase()}</span><b>Predicho</b><b>Mercado</b><b>Δ</b></div>
              {oddsPredictions.games
                .slice()
                .sort((a, b) => Math.max(Math.abs(b.awayDelta ?? 0), Math.abs(b.homeDelta ?? 0)) - Math.max(Math.abs(a.awayDelta ?? 0), Math.abs(a.homeDelta ?? 0)))
                .flatMap(g => [
                  { key: `${g.gamePk}-a`, team: g.awayTeam, opp: `@ ${g.homeTeam}`, pred: g.awayPredicted, mkt: g.awayMarket, delta: g.awayDelta },
                  { key: `${g.gamePk}-h`, team: g.homeTeam, opp: `vs ${g.awayTeam}`, pred: g.homePredicted, mkt: g.homeMarket, delta: g.homeDelta },
                ])
                .map(row => (
                  <div key={row.key}>
                    <strong>{row.team} <small style={{ opacity: 0.6, marginLeft: 6 }}>{row.opp}</small></strong>
                    <b>{row.pred != null ? (row.pred > 0 ? `+${Math.round(row.pred)}` : Math.round(row.pred)) : '—'}</b>
                    <b>{row.mkt != null ? (row.mkt > 0 ? `+${row.mkt}` : row.mkt) : '—'}</b>
                    <span>{row.delta != null ? `${row.delta > 0 ? '+' : ''}${Math.round(row.delta)}` : '—'}</span>
                  </div>
                ))}
            </div>
          )}
          <p className="audit-conclusion coach-watch"><Info size={16} /> Backtest sobre 6,040 juegos (2024-2026) usando <code>walkforward_preds.parquet</code> + <code>odds_close.parquet</code>. Única regla que aportó señal real: <strong>cuando el pick califica como FUERTE y el mercado le da ≥58% al mismo lado, se promueve a LOCK</strong> (win-rate histórico 67.3% vs 69.7% del LOCK base; +23% volumen de picks LOCK). El resto de variantes (boost simple, veto por dirección contraria, downgrade de FUERTE por mercado adverso) no discriminaron y fueron descartadas.</p>
        </section>
      )}

    </DashboardTabs>
  );
}

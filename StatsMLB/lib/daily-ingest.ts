import path from 'node:path';
import {
  finishRefreshRun,
  readStoredGames,
  saveGames,
  saveOdds,
  startRefreshRun,
  type ConfirmedLineup,
  type OddsObservation,
  type ScheduleGame,
} from '@/db/daily';

const MEXICO_TIME_ZONE = 'America/Mexico_City';
const ACTION_USER_AGENT = 'StatsMLB/1.0 (+daily-results-and-odds-refresh)';
const VERIFIED_BOOKS = new Set(['DraftKings', 'FanDuel', 'BetRivers', 'BetMGM']);

export type DailyRefreshProgress = {
  percent: number;
  stage: string;
  message: string;
};

type DailyRefreshReporter = (progress: DailyRefreshProgress) => void | Promise<void>;

async function reportProgress(reporter: DailyRefreshReporter | undefined, progress: DailyRefreshProgress) {
  if (!reporter) return;
  try {
    await reporter(progress);
  } catch {
    // A disconnected browser must not interrupt persistence once refresh started.
  }
}

type MlbGame = {
  gamePk?: number;
  gameDate?: string;
  officialDate?: string;
  seriesGameNumber?: number;
  gamesInSeries?: number;
  seriesDescription?: string;
  status?: { detailedState?: string; abstractGameState?: string };
  venue?: { name?: string };
  teams?: {
    away?: { team?: { abbreviation?: string; name?: string }; score?: number; probablePitcher?: { fullName?: string } };
    home?: { team?: { abbreviation?: string; name?: string }; score?: number; probablePitcher?: { fullName?: string } };
  };
};

type BoxscoreTeam = {
  battingOrder?: number[];
  players?: Record<string, { person?: { id?: number; fullName?: string } }>;
};

type ActionBook = { id?: number; display_name?: string; parent_name?: string };
type ActionTeam = { id?: number; abbr?: string; full_name?: string };
type ActionOdds = {
  book_id?: number;
  ml_away?: number;
  ml_home?: number;
  inserted?: string;
  [key: string]: unknown;
};
type ActionGame = {
  id?: number;
  start_time?: string;
  away_team_id?: number;
  home_team_id?: number;
  teams?: ActionTeam[];
  odds?: ActionOdds[];
};

function dateParts(date: Date) {
  const parts = new Intl.DateTimeFormat('en-CA', {
    timeZone: MEXICO_TIME_ZONE, year: 'numeric', month: '2-digit', day: '2-digit',
  }).formatToParts(date);
  const values = Object.fromEntries(parts.map(part => [part.type, part.value]));
  return `${values.year}-${values.month}-${values.day}`;
}

export function mexicoDate(offsetDays = 0) {
  const base = dateParts(new Date());
  const atMiddayUtc = new Date(`${base}T12:00:00Z`);
  atMiddayUtc.setUTCDate(atMiddayUtc.getUTCDate() + offsetDays);
  return atMiddayUtc.toISOString().slice(0, 10);
}

function compactDate(date: string) {
  return date.replaceAll('-', '');
}

function canonicalTeam(code?: string) {
  const value = (code ?? '').toUpperCase();
  return ({ ARI: 'AZ', OAK: 'ATH', CHW: 'CWS', SFG: 'SF', KCR: 'KC', SDP: 'SD', TBR: 'TB', WAS: 'WSH' } as Record<string, string>)[value] ?? value;
}

async function fetchJson<T>(url: string, headers: HeadersInit = {}, timeoutMs = 15_000): Promise<T> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { cache: 'no-store', headers: { Accept: 'application/json', ...headers }, signal: controller.signal });
    if (!response.ok) throw new Error(`${new URL(url).hostname} respondió ${response.status}`);
    return await response.json() as T;
  } finally {
    clearTimeout(timeout);
  }
}

async function fetchLineups(gamePk?: number) {
  const empty = { away: null, home: null } as { away: ConfirmedLineup | null; home: ConfirmedLineup | null };
  if (!gamePk) return empty;
  try {
    const payload = await fetchJson<{ teams?: { away?: BoxscoreTeam; home?: BoxscoreTeam } }>(
      `https://statsapi.mlb.com/api/v1/game/${gamePk}/boxscore`, {}, 10_000,
    );
    const lineup = (team?: BoxscoreTeam): ConfirmedLineup | null => {
      const order = team?.battingOrder ?? [];
      if (order.length < 8) return null;
      return {
        confirmed: true,
        players: order.map((playerId, index) => ({
          playerId,
          battingOrder: index + 1,
          name: team?.players?.[`ID${playerId}`]?.person?.fullName,
        })),
      };
    };
    return { away: lineup(payload.teams?.away), home: lineup(payload.teams?.home) };
  } catch {
    return empty;
  }
}

export async function fetchMlbSchedule(startDate: string, endDate = startDate, includeLineups = false) {
  const url = new URL('https://statsapi.mlb.com/api/v1/schedule');
  url.searchParams.set('sportId', '1');
  url.searchParams.set('startDate', startDate);
  url.searchParams.set('endDate', endDate);
  url.searchParams.set('hydrate', 'probablePitcher,team,linescore');
  const payload = await fetchJson<{ dates?: { games?: MlbGame[] }[] }>(url.toString());
  const rawGames = (payload.dates ?? []).flatMap(block => block.games ?? []).filter(game => game.gamePk && game.gameDate && game.officialDate);
  const lineups = includeLineups
    ? await Promise.all(rawGames.map(game => fetchLineups(game.gamePk)))
    : rawGames.map(() => ({ away: null, home: null }));
  return rawGames.map((game, index): ScheduleGame => {
    const away = game.teams?.away;
    const home = game.teams?.home;
    return {
      gamePk: Number(game.gamePk),
      gameDate: String(game.gameDate),
      officialDate: String(game.officialDate),
      seriesGameNumber: game.seriesGameNumber,
      gamesInSeries: game.gamesInSeries,
      seriesDescription: game.seriesDescription,
      status: game.status?.detailedState ?? 'Programado',
      abstractState: game.status?.abstractGameState ?? 'Preview',
      venue: game.venue?.name ?? '',
      away: {
        team: canonicalTeam(away?.team?.abbreviation), name: away?.team?.name ?? canonicalTeam(away?.team?.abbreviation),
        score: away?.score, pitcher: away?.probablePitcher?.fullName, lineup: lineups[index].away,
      },
      home: {
        team: canonicalTeam(home?.team?.abbreviation), name: home?.team?.name ?? canonicalTeam(home?.team?.abbreviation),
        score: home?.score, pitcher: home?.probablePitcher?.fullName, lineup: lineups[index].home,
      },
    };
  }).sort((a, b) => a.gameDate.localeCompare(b.gameDate) || a.gamePk - b.gamePk);
}

function preferSavedFinal(live: ScheduleGame[], saved: ScheduleGame[]) {
  const savedByGame = new Map(saved.map(game => [game.gamePk, game]));
  const merged = live.map(game => {
    const stored = savedByGame.get(game.gamePk);
    if (!stored) return game;
    savedByGame.delete(game.gamePk);
    if (stored.abstractState === 'Final') return {
      ...game,
      ...stored,
      seriesGameNumber: game.seriesGameNumber ?? stored.seriesGameNumber,
      gamesInSeries: game.gamesInSeries ?? stored.gamesInSeries,
      seriesDescription: game.seriesDescription ?? stored.seriesDescription,
      away: { ...game.away, ...stored.away, lineup: stored.away.lineup ?? game.away.lineup },
      home: { ...game.home, ...stored.home, lineup: stored.home.lineup ?? game.home.lineup },
    };
    return {
      ...game,
      away: { ...game.away, lineup: game.away.lineup ?? stored.away.lineup },
      home: { ...game.home, lineup: game.home.lineup ?? stored.home.lineup },
    };
  });
  return [...merged, ...savedByGame.values()].sort((a, b) => a.gameDate.localeCompare(b.gameDate) || a.gamePk - b.gamePk);
}

export async function scheduleForDate(selectedDate: string) {
  const saved = await readStoredGames(selectedDate);
  try {
    const live = await fetchMlbSchedule(selectedDate, selectedDate, selectedDate === mexicoDate());
    return {
      sourceState: saved.length ? 'LIVE + SAVED' : 'LIVE SCHEDULE',
      source: saved.length ? 'MLB StatsAPI + StatsMLB DB' : 'MLB StatsAPI',
      retrievedAt: new Date().toISOString(),
      selectedDate,
      games: preferSavedFinal(live, saved),
    };
  } catch (error) {
    if (saved.length) {
      return {
        sourceState: 'SAVED SNAPSHOT', source: 'StatsMLB DB', retrievedAt: new Date().toISOString(), selectedDate,
        games: saved, warning: error instanceof Error ? error.message : 'No se pudo consultar MLB StatsAPI',
      };
    }
    throw error;
  }
}

async function fetchActionNetwork(dates: string[], mlbGames: ScheduleGame[]) {
  const headers = { 'User-Agent': ACTION_USER_AGENT };
  const [booksPayload, ...scoreboards] = await Promise.all([
    fetchJson<{ books?: ActionBook[] }>('https://api.actionnetwork.com/web/v1/books', headers),
    ...dates.map(date => fetchJson<{ games?: ActionGame[] }>(
      `https://api.actionnetwork.com/web/v1/scoreboard/mlb?period=game&date=${compactDate(date)}`, headers,
    )),
  ]);
  const books = new Map((booksPayload.books ?? []).flatMap(book => book.id == null ? [] : [[Number(book.id), book] as const]));
  const mlbByMatchup = new Map(mlbGames.map(game => [`${game.officialDate}:${canonicalTeam(game.away.team)}:${canonicalTeam(game.home.team)}`, game]));
  const observations: OddsObservation[] = [];
  const eventIds = new Set<string>();
  const sportsbookNames = new Set<string>();

  for (const scoreboard of scoreboards) {
    for (const event of scoreboard.games ?? []) {
      if (!event.id || !event.start_time) continue;
      const away = (event.teams ?? []).find(team => team.id === event.away_team_id);
      const home = (event.teams ?? []).find(team => team.id === event.home_team_id);
      if (!away?.abbr || !home?.abbr) continue;
      const officialDate = dateParts(new Date(event.start_time));
      const matchup = `${officialDate}:${canonicalTeam(away.abbr)}:${canonicalTeam(home.abbr)}`;
      const mlbGame = mlbByMatchup.get(matchup);
      const latestByBook = new Map<string, ActionOdds>();
      for (const odds of event.odds ?? []) {
        if (odds.book_id == null || odds.ml_away == null || odds.ml_home == null) continue;
        const book = books.get(Number(odds.book_id));
        const canonical = String(book?.parent_name || book?.display_name || '');
        if (!VERIFIED_BOOKS.has(canonical)) continue;
        const prior = latestByBook.get(canonical);
        if (!prior || Date.parse(String(odds.inserted ?? '')) > Date.parse(String(prior.inserted ?? ''))) latestByBook.set(canonical, odds);
      }
      for (const [sportsbook, odds] of latestByBook) {
        observations.push({
          gamePk: mlbGame?.gamePk,
          officialDate: mlbGame?.officialDate ?? officialDate,
          sourceEventId: String(event.id),
          commenceTime: event.start_time,
          awayTeam: mlbGame?.away.name ?? away.full_name ?? away.abbr,
          homeTeam: mlbGame?.home.name ?? home.full_name ?? home.abbr,
          sportsbook,
          awayAmerican: Number(odds.ml_away),
          homeAmerican: Number(odds.ml_home),
          sourceUpdatedAt: odds.inserted ? new Date(odds.inserted).toISOString() : new Date().toISOString(),
          raw: odds,
        });
        sportsbookNames.add(sportsbook);
      }
      if (latestByBook.size) eventIds.add(String(event.id));
    }
  }
  return { observations, eventCount: eventIds.size, sportsbookCount: sportsbookNames.size };
}

async function runOddsEngineScript(report?: DailyRefreshReporter): Promise<string | null> {
  void report;
  let spawnFn: typeof import('node:child_process').spawn;
  let existsSync: typeof import('node:fs').existsSync;
  try {
    ({ spawn: spawnFn } = await import('node:child_process'));
    ({ existsSync } = await import('node:fs'));
  } catch {
    return 'child_process/fs no disponibles en este runtime';
  }
  const scriptCandidates = [
    process.env.STATSMLB_ODDS_SCRIPT,
    path.resolve(process.cwd(), 'scripts', 'odds_walkforward.py'),
    'C:\\Users\\andra\\Desktop\\STRIKECAST\\StatsMLB\\scripts\\odds_walkforward.py',
  ].filter(Boolean) as string[];
  const scriptPath = scriptCandidates.find(p => existsSync(p));
  if (!scriptPath) return `Script no encontrado. Probados: ${scriptCandidates.join(' | ')}`;

  const candidates = [
    process.env.STATSMLB_PYTHON,
    process.env.PYTHON,
    'C:\\Users\\andra\\miniconda3\\python.exe',
    'C:\\Users\\andra\\anaconda3\\python.exe',
    'python',
    'py',
  ].filter(Boolean) as string[];

  const attempts: string[] = [];

  return new Promise<string | null>(resolve => {
    let index = 0;
    const tryNext = () => {
      if (index >= candidates.length) {
        return resolve(`No se pudo localizar un intérprete de Python. Probados: ${attempts.join(' | ')}`);
      }
      const bin = candidates[index++];
      // Si el candidato luce como path absoluto, verificar que exista antes de spawn.
      const looksAbsolute = /[\\\/]/.test(bin);
      if (looksAbsolute && !existsSync(bin)) {
        attempts.push(`${bin} (no existe)`);
        return tryNext();
      }
      try {
        const child = spawnFn(bin, [scriptPath], { windowsHide: true, shell: !looksAbsolute });
        let stderr = '';
        child.stderr?.on('data', chunk => { stderr += chunk.toString(); });
        child.on('error', err => {
          attempts.push(`${bin} (spawn error: ${err.message})`);
          tryNext();
        });
        child.on('close', code => {
          if (code === 0) resolve(null);
          else resolve(`odds_walkforward.py (${bin}) salió con código ${code}${stderr ? `: ${stderr.slice(0, 200)}` : ''}`);
        });
      } catch (err) {
        attempts.push(`${bin} (throw: ${err instanceof Error ? err.message : String(err)})`);
        tryNext();
      }
    };
    tryNext();
  });
}

export async function refreshDailyData(report?: DailyRefreshReporter) {
  const startedAt = new Date().toISOString();
  const runId = crypto.randomUUID();
  const yesterdayDate = mexicoDate(-1);
  const todayDate = mexicoDate();
  const tomorrowDate = mexicoDate(1);
  const errors: string[] = [];
  await reportProgress(report, { percent: 3, stage: 'Preparando', message: 'Iniciando la actualización diaria' });
  await startRefreshRun(runId, startedAt, todayDate, tomorrowDate);
  await reportProgress(report, { percent: 10, stage: 'Preparando', message: 'Registro de actualización creado' });

  let games: ScheduleGame[] = [];
  let oddsEvents = 0;
  let oddsBooks = 0;
  try {
    await reportProgress(report, { percent: 15, stage: 'Resultados y juegos', message: 'Consultando MLB para ayer, hoy y mañana' });
    // Reconcile yesterday as well: a refresh shortly after midnight must still
    // persist the games that just became final on the previous Mexico City day.
    games = await fetchMlbSchedule(yesterdayDate, tomorrowDate, true);
    await reportProgress(report, { percent: 43, stage: 'Resultados y juegos', message: `${games.length} juegos recibidos; guardando resultados y alineaciones` });
    await saveGames(games, 'MLB StatsAPI', new Date().toISOString());
    await reportProgress(report, { percent: 55, stage: 'Resultados y juegos', message: 'Resultados, horarios y juegos guardados' });
  } catch (error) {
    errors.push(`Calendario/resultados: ${error instanceof Error ? error.message : 'error desconocido'}`);
    games = [
      ...await readStoredGames(yesterdayDate),
      ...await readStoredGames(todayDate),
      ...await readStoredGames(tomorrowDate),
    ];
    await reportProgress(report, { percent: 55, stage: 'Resultados y juegos', message: 'La fuente MLB falló; se conservaron los datos guardados' });
  }

  try {
    await reportProgress(report, { percent: 62, stage: 'Momios', message: 'Consultando casas verificadas para hoy y mañana' });
    const odds = await fetchActionNetwork([todayDate, tomorrowDate], games);
    await reportProgress(report, { percent: 80, stage: 'Momios', message: `${odds.eventCount} juegos con momios encontrados; guardando precios` });
    await saveOdds(odds.observations, new Date().toISOString());
    oddsEvents = odds.eventCount;
    oddsBooks = odds.sportsbookCount;
    await reportProgress(report, { percent: 90, stage: 'Momios', message: `Momios guardados desde ${oddsBooks} casas verificadas` });
  } catch (error) {
    errors.push(`Momios: ${error instanceof Error ? error.message : 'error desconocido'}`);
    await reportProgress(report, { percent: 90, stage: 'Momios', message: 'La fuente de momios falló; se conservaron los precios guardados' });
  }

  // Motor de momios: intento oportunista. Si el servidor Next corre en un sandbox
  // sin acceso al filesystem del usuario (contenedor, deploy remoto), simplemente
  // no aplica — la tarea programada local del usuario mantiene los JSONs frescos.
  try {
    await reportProgress(report, { percent: 92, stage: 'Motor de momios', message: 'Regenerando JSONs derivados' });
    const oddsErr = await runOddsEngineScript(report);
    if (oddsErr) {
      const skippable = oddsErr.startsWith('Script no encontrado')
        || oddsErr.startsWith('No se pudo localizar un intérprete')
        || oddsErr.startsWith('child_process');
      if (!skippable) {
        errors.push(`Motor de momios: ${oddsErr}`);
      }
      // Si es skippable, ni siquiera lo mencionamos — es un entorno donde no aplica.
    }
  } catch (error) {
    errors.push(`Motor de momios: ${error instanceof Error ? error.message : 'error desconocido'}`);
  }

  const finishedAt = new Date().toISOString();
  const status = errors.length === 0 ? 'success' : games.length ? 'partial' : 'error';
  const summary = {
    id: runId,
    status,
    startedAt,
    finishedAt,
    yesterdayDate,
    todayDate,
    tomorrowDate,
    gamesSeen: games.length,
    yesterdayGames: games.filter(game => game.officialDate === yesterdayDate).length,
    todayGames: games.filter(game => game.officialDate === todayDate).length,
    tomorrowGames: games.filter(game => game.officialDate === tomorrowDate).length,
    finalGames: games.filter(game => game.abstractState === 'Final').length,
    oddsEvents,
    oddsBooks,
    errors,
  };
  await reportProgress(report, { percent: 96, stage: 'Finalizando', message: 'Guardando el resumen de la actualización' });
  await finishRefreshRun(runId, summary);
  await reportProgress(report, {
    percent: 100,
    stage: status === 'success' ? 'Actualización completa' : status === 'partial' ? 'Actualización parcial' : 'Actualización fallida',
    message: status === 'success' ? 'Resultados, juegos y momios quedaron actualizados' : errors.join(' · '),
  });
  return summary;
}

import { env } from 'cloudflare:workers';

export type ConfirmedLineup = {
  confirmed: boolean;
  players: { playerId: number; battingOrder: number; name?: string }[];
};

export type ScheduleGame = {
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

export type CurrentOddsGame = {
  eventId: string;
  gamePk?: number;
  officialDate: string;
  commenceTime: string;
  awayTeam: string;
  homeTeam: string;
  lastUpdated: string;
  bestAway: { book: string; american: number; decimal: number } | null;
  bestHome: { book: string; american: number; decimal: number } | null;
  books: { name: string; awayAmerican: number; homeAmerican: number; awayDecimal: number; homeDecimal: number; lastUpdated: string }[];
  // Positive values mean the observed market moved toward the home team.
  // It is measured from the first stored quote for the same book, not claimed
  // to be the sportsbook's official opening line.
  marketMoveHomePp?: number | null;
  marketMoveBooks?: number;
  marketMoveFrom?: string | null;
};

type D1Env = { DB: D1Database };
type D1Row = Record<string, unknown>;

const database = (env as unknown as D1Env).DB;
let schemaReady: Promise<void> | null = null;

const GAME_TABLE_SQL = `CREATE TABLE IF NOT EXISTS daily_games (
  game_pk INTEGER PRIMARY KEY,
  official_date TEXT NOT NULL,
  game_date TEXT NOT NULL,
  status TEXT NOT NULL,
  abstract_state TEXT NOT NULL,
  venue TEXT,
  away_team TEXT NOT NULL,
  away_name TEXT NOT NULL,
  away_score INTEGER,
  away_pitcher TEXT,
  away_lineup_json TEXT,
  home_team TEXT NOT NULL,
  home_name TEXT NOT NULL,
  home_score INTEGER,
  home_pitcher TEXT,
  home_lineup_json TEXT,
  source TEXT NOT NULL,
  raw_json TEXT NOT NULL,
  first_seen_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  final_at TEXT
)`;

const ODDS_TABLE_SQL = `CREATE TABLE IF NOT EXISTS odds_snapshots (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  game_pk INTEGER,
  official_date TEXT NOT NULL,
  source_event_id TEXT NOT NULL,
  commence_time TEXT NOT NULL,
  away_team TEXT NOT NULL,
  home_team TEXT NOT NULL,
  sportsbook TEXT NOT NULL,
  away_american INTEGER NOT NULL,
  home_american INTEGER NOT NULL,
  source_updated_at TEXT NOT NULL,
  captured_at TEXT NOT NULL,
  source TEXT NOT NULL,
  raw_json TEXT NOT NULL
)`;

const RUN_TABLE_SQL = `CREATE TABLE IF NOT EXISTS refresh_runs (
  id TEXT PRIMARY KEY,
  started_at TEXT NOT NULL,
  finished_at TEXT,
  status TEXT NOT NULL,
  today_date TEXT NOT NULL,
  tomorrow_date TEXT NOT NULL,
  games_seen INTEGER NOT NULL DEFAULT 0,
  final_games INTEGER NOT NULL DEFAULT 0,
  odds_events INTEGER NOT NULL DEFAULT 0,
  odds_books INTEGER NOT NULL DEFAULT 0,
  errors_json TEXT NOT NULL DEFAULT '[]'
)`;

export async function ensureDatabase() {
  if (!schemaReady) {
    schemaReady = (async () => {
      await database.batch([
        database.prepare(GAME_TABLE_SQL),
        database.prepare(ODDS_TABLE_SQL),
        database.prepare(RUN_TABLE_SQL),
        database.prepare('CREATE INDEX IF NOT EXISTS idx_daily_games_official_date ON daily_games (official_date, game_date)'),
        database.prepare('CREATE INDEX IF NOT EXISTS idx_daily_games_final_at ON daily_games (final_at)'),
        database.prepare('CREATE UNIQUE INDEX IF NOT EXISTS uq_odds_snapshot_observation ON odds_snapshots (source_event_id, sportsbook, source_updated_at, away_american, home_american)'),
        database.prepare('CREATE INDEX IF NOT EXISTS idx_odds_snapshots_date_game ON odds_snapshots (official_date, game_pk, captured_at)'),
        database.prepare('CREATE INDEX IF NOT EXISTS idx_refresh_runs_started_at ON refresh_runs (started_at)'),
      ]);
      await database.prepare('PRAGMA optimize').run();
    })().catch(error => {
      schemaReady = null;
      throw error;
    });
  }
  await schemaReady;
}

function parseLineup(value: unknown): ConfirmedLineup | null {
  if (typeof value !== 'string' || !value) return null;
  try { return JSON.parse(value) as ConfirmedLineup; } catch { return null; }
}

function rowToGame(row: D1Row): ScheduleGame {
  let raw: Partial<ScheduleGame> = {};
  try { raw = typeof row.raw_json === 'string' ? JSON.parse(row.raw_json) as Partial<ScheduleGame> : {}; } catch { raw = {}; }
  return {
    gamePk: Number(row.game_pk),
    gameDate: String(row.game_date),
    officialDate: String(row.official_date),
    seriesGameNumber: raw.seriesGameNumber,
    gamesInSeries: raw.gamesInSeries,
    seriesDescription: raw.seriesDescription,
    status: String(row.status),
    abstractState: String(row.abstract_state),
    venue: String(row.venue ?? ''),
    away: {
      team: String(row.away_team), name: String(row.away_name),
      score: row.away_score == null ? undefined : Number(row.away_score),
      pitcher: row.away_pitcher == null ? undefined : String(row.away_pitcher),
      lineup: parseLineup(row.away_lineup_json),
    },
    home: {
      team: String(row.home_team), name: String(row.home_name),
      score: row.home_score == null ? undefined : Number(row.home_score),
      pitcher: row.home_pitcher == null ? undefined : String(row.home_pitcher),
      lineup: parseLineup(row.home_lineup_json),
    },
  };
}

export async function readStoredGames(officialDate: string) {
  await ensureDatabase();
  const result = await database.prepare('SELECT * FROM daily_games WHERE official_date = ? ORDER BY game_date, game_pk')
    .bind(officialDate).all<D1Row>();
  return (result.results ?? []).map(rowToGame);
}

export async function saveGames(games: ScheduleGame[], source: string, capturedAt: string) {
  await ensureDatabase();
  if (!games.length) return;
  const statements = games.map(game => database.prepare(`INSERT INTO daily_games (
      game_pk, official_date, game_date, status, abstract_state, venue,
      away_team, away_name, away_score, away_pitcher, away_lineup_json,
      home_team, home_name, home_score, home_pitcher, home_lineup_json,
      source, raw_json, first_seen_at, updated_at, final_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(game_pk) DO UPDATE SET
      official_date = CASE WHEN daily_games.abstract_state = 'Final' THEN daily_games.official_date ELSE excluded.official_date END,
      game_date = CASE WHEN daily_games.abstract_state = 'Final' THEN daily_games.game_date ELSE excluded.game_date END,
      status = CASE WHEN daily_games.abstract_state = 'Final' THEN daily_games.status ELSE excluded.status END,
      abstract_state = CASE WHEN daily_games.abstract_state = 'Final' THEN daily_games.abstract_state ELSE excluded.abstract_state END,
      venue = COALESCE(excluded.venue, daily_games.venue),
      away_team = excluded.away_team,
      away_name = excluded.away_name,
      away_score = COALESCE(excluded.away_score, daily_games.away_score),
      away_pitcher = COALESCE(excluded.away_pitcher, daily_games.away_pitcher),
      away_lineup_json = COALESCE(excluded.away_lineup_json, daily_games.away_lineup_json),
      home_team = excluded.home_team,
      home_name = excluded.home_name,
      home_score = COALESCE(excluded.home_score, daily_games.home_score),
      home_pitcher = COALESCE(excluded.home_pitcher, daily_games.home_pitcher),
      home_lineup_json = COALESCE(excluded.home_lineup_json, daily_games.home_lineup_json),
      source = excluded.source,
      raw_json = excluded.raw_json,
      updated_at = excluded.updated_at,
      final_at = COALESCE(excluded.final_at, daily_games.final_at)`)
    .bind(
      game.gamePk, game.officialDate, game.gameDate, game.status, game.abstractState, game.venue || null,
      game.away.team, game.away.name, game.away.score ?? null, game.away.pitcher ?? null, game.away.lineup ? JSON.stringify(game.away.lineup) : null,
      game.home.team, game.home.name, game.home.score ?? null, game.home.pitcher ?? null, game.home.lineup ? JSON.stringify(game.home.lineup) : null,
      source, JSON.stringify(game), capturedAt, capturedAt, game.abstractState === 'Final' ? capturedAt : null,
    ));
  await database.batch(statements);
}

type OddsObservation = {
  gamePk?: number;
  officialDate: string;
  sourceEventId: string;
  commenceTime: string;
  awayTeam: string;
  homeTeam: string;
  sportsbook: string;
  awayAmerican: number;
  homeAmerican: number;
  sourceUpdatedAt: string;
  raw: unknown;
};

export async function saveOdds(observations: OddsObservation[], capturedAt: string) {
  await ensureDatabase();
  if (!observations.length) return;
  const statements = observations.map(item => database.prepare(`INSERT OR IGNORE INTO odds_snapshots (
      game_pk, official_date, source_event_id, commence_time, away_team, home_team, sportsbook,
      away_american, home_american, source_updated_at, captured_at, source, raw_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`)
    .bind(
      item.gamePk ?? null, item.officialDate, item.sourceEventId, item.commenceTime, item.awayTeam, item.homeTeam,
      item.sportsbook, item.awayAmerican, item.homeAmerican, item.sourceUpdatedAt, capturedAt,
      'Action Network', JSON.stringify(item.raw),
    ));
  await database.batch(statements);
}

function americanToDecimal(value: number) {
  return value > 0 ? 1 + value / 100 : 1 + 100 / Math.abs(value);
}

function deVigHomeProbability(homeAmerican: number, awayAmerican: number) {
  const home = 1 / americanToDecimal(homeAmerican);
  const away = 1 / americanToDecimal(awayAmerican);
  return home + away > 0 ? home / (home + away) : null;
}

export async function readStoredOdds(startDate: string, endDate: string) {
  await ensureDatabase();
  const result = await database.prepare(`SELECT * FROM odds_snapshots
    WHERE official_date BETWEEN ? AND ?
    ORDER BY source_updated_at DESC, captured_at DESC, id DESC`).bind(startDate, endDate).all<D1Row>();
  const latest = new Map<string, D1Row>();
  const opening = new Map<string, D1Row>();
  for (const row of result.results ?? []) {
    const key = `${row.source_event_id}:${row.sportsbook}`;
    if (!latest.has(key)) latest.set(key, row);
    // Query is newest → oldest, so the final row seen is the first quote we
    // captured for this event/book combination.
    opening.set(key, row);
  }
  const grouped = new Map<string, D1Row[]>();
  for (const row of latest.values()) {
    const key = String(row.source_event_id);
    grouped.set(key, [...(grouped.get(key) ?? []), row]);
  }
  const games: CurrentOddsGame[] = [];
  for (const [eventId, rows] of grouped) {
    const books = rows.map(row => {
      const awayAmerican = Number(row.away_american);
      const homeAmerican = Number(row.home_american);
      return {
        name: String(row.sportsbook), awayAmerican, homeAmerican,
        awayDecimal: americanToDecimal(awayAmerican), homeDecimal: americanToDecimal(homeAmerican),
        lastUpdated: String(row.source_updated_at),
      };
    }).sort((a, b) => a.name.localeCompare(b.name));
    const bestAway = books.reduce<(typeof books)[number] | null>((best, book) => !best || book.awayDecimal > best.awayDecimal ? book : best, null);
    const bestHome = books.reduce<(typeof books)[number] | null>((best, book) => !best || book.homeDecimal > best.homeDecimal ? book : best, null);
    const first = rows[0];
    const moves = rows.flatMap(row => {
      const key = `${row.source_event_id}:${row.sportsbook}`;
      const initial = opening.get(key);
      if (!initial) return [];
      const now = deVigHomeProbability(Number(row.home_american), Number(row.away_american));
      const then = deVigHomeProbability(Number(initial.home_american), Number(initial.away_american));
      return now == null || then == null ? [] : [now - then];
    });
    const marketMoveHomePp = moves.length ? moves.reduce((sum, value) => sum + value, 0) / moves.length * 100 : null;
    const marketMoveFrom = rows.map(row => opening.get(`${row.source_event_id}:${row.sportsbook}`)?.captured_at).filter((value): value is string => typeof value === 'string').sort()[0] ?? null;
    games.push({
      eventId,
      gamePk: first.game_pk == null ? undefined : Number(first.game_pk),
      officialDate: String(first.official_date),
      commenceTime: String(first.commence_time),
      awayTeam: String(first.away_team),
      homeTeam: String(first.home_team),
      lastUpdated: books.reduce((latestAt, book) => book.lastUpdated > latestAt ? book.lastUpdated : latestAt, books[0]?.lastUpdated ?? ''),
      books,
      bestAway: bestAway ? { book: bestAway.name, american: bestAway.awayAmerican, decimal: bestAway.awayDecimal } : null,
      bestHome: bestHome ? { book: bestHome.name, american: bestHome.homeAmerican, decimal: bestHome.homeDecimal } : null,
      marketMoveHomePp,
      marketMoveBooks: moves.length,
      marketMoveFrom,
    });
  }
  return games.sort((a, b) => a.commenceTime.localeCompare(b.commenceTime));
}

export async function startRefreshRun(id: string, startedAt: string, todayDate: string, tomorrowDate: string) {
  await ensureDatabase();
  await database.prepare(`INSERT INTO refresh_runs (id, started_at, status, today_date, tomorrow_date)
    VALUES (?, ?, 'running', ?, ?)`).bind(id, startedAt, todayDate, tomorrowDate).run();
}

export async function finishRefreshRun(id: string, summary: {
  status: string; finishedAt: string; gamesSeen: number; finalGames: number; oddsEvents: number; oddsBooks: number; errors: string[];
}) {
  await database.prepare(`UPDATE refresh_runs SET finished_at = ?, status = ?, games_seen = ?, final_games = ?,
    odds_events = ?, odds_books = ?, errors_json = ? WHERE id = ?`)
    .bind(summary.finishedAt, summary.status, summary.gamesSeen, summary.finalGames, summary.oddsEvents, summary.oddsBooks, JSON.stringify(summary.errors), id).run();
}

export async function latestRefreshRun() {
  await ensureDatabase();
  const row = await database.prepare('SELECT * FROM refresh_runs ORDER BY started_at DESC LIMIT 1').first<D1Row>();
  if (!row) return null;
  return {
    id: String(row.id), startedAt: String(row.started_at), finishedAt: row.finished_at ? String(row.finished_at) : null,
    status: String(row.status), todayDate: String(row.today_date), tomorrowDate: String(row.tomorrow_date),
    gamesSeen: Number(row.games_seen), finalGames: Number(row.final_games), oddsEvents: Number(row.odds_events), oddsBooks: Number(row.odds_books),
    errors: (() => { try { return JSON.parse(String(row.errors_json)) as string[]; } catch { return []; } })(),
  };
}

export type { OddsObservation };

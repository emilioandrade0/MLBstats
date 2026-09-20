import { env } from 'cloudflare:workers';
import type { LiveGame } from '../lib/nfl-live';
const db = (env as unknown as {DB?:D1Database}).DB;
let ready: Promise<void> | undefined;
async function ensure() {
  if (!db) throw Error('D1 no disponible');
  if (!ready) ready=db.batch([
    db.prepare('CREATE TABLE IF NOT EXISTS nfl_live_games (source_id TEXT PRIMARY KEY, season INTEGER NOT NULL, status TEXT NOT NULL, captured_at TEXT NOT NULL, payload TEXT NOT NULL)'),
    db.prepare('CREATE INDEX IF NOT EXISTS nfl_live_season ON nfl_live_games(season)'),
    db.prepare('CREATE TABLE IF NOT EXISTS nfl_live_checks (date TEXT PRIMARY KEY, checked_at TEXT NOT NULL)'),
  ]).then(()=>{}).catch(e=>{ready=undefined;throw e;});
  await ready;
  return db;
}
export async function readNflLive(season: number, date: string) {
  const database=await ensure();
  const [rows,check]=await Promise.all([
    database.prepare('SELECT payload FROM nfl_live_games WHERE season=?').bind(season).all<{payload:string}>(),
    database.prepare('SELECT checked_at FROM nfl_live_checks WHERE date=?').bind(date).first<{checked_at:string}>(),
  ]);
  return {games:rows.results.map(r=>JSON.parse(r.payload) as LiveGame),checkedAt:check?.checked_at||null};
}
export async function saveNflLive(games: LiveGame[], date: string, checkedAt: string) {
  const database=await ensure();
  const statements=games.map(g=>database.prepare(`INSERT INTO nfl_live_games(source_id,season,status,captured_at,payload) VALUES(?,?,?,?,?)
    ON CONFLICT(source_id) DO UPDATE SET season=excluded.season,status=excluded.status,captured_at=excluded.captured_at,payload=excluded.payload
    WHERE excluded.captured_at>=nfl_live_games.captured_at AND (nfl_live_games.status!='final' OR excluded.status='final')`).bind(g.sourceId,g.season,g.status,g.capturedAt,JSON.stringify(g)));
  statements.push(database.prepare('INSERT INTO nfl_live_checks(date,checked_at) VALUES(?,?) ON CONFLICT(date) DO UPDATE SET checked_at=excluded.checked_at WHERE excluded.checked_at>=nfl_live_checks.checked_at').bind(date,checkedAt));
  await database.batch(statements);
}

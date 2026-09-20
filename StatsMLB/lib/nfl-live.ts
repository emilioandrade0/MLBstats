import type { Game } from './nfl/api';

export type LiveMarket = { home: number | null; away: number | null; homeSpread: number | null; total: number | null; bookId: number; updatedAt: string | null };
export type LiveGame = Game & { sourceId: string; sourceStatus: string; capturedAt: string; clock: string | null; period: number | null; liveMarket: LiveMarket | null };
export type LiveResponse = { games: LiveGame[]; state: 'LIVE' | 'SAVED' | 'UNAVAILABLE'; checkedAt: string | null; date: string; error?: string; persistence: boolean };
type RawOdds = { type?: string; book_id?: number; ml_home?: number; ml_away?: number; spread_home?: number; total?: number; inserted?: string };
type RawGame = { id?: number; league_name?: string; season?: number; week?: number; type?: string; start_time?: string; status?: string; status_display?: string; home_team_id?: number; away_team_id?: number; teams?: { id?: number; abbr?: string }[]; odds?: RawOdds[]; boxscore?: { total_home_points?: number; total_away_points?: number; clock?: string; period?: number } };
const teams = new Set('ARI ATL BAL BUF CAR CHI CIN CLE DAL DEN DET GB HOU IND JAX KC LV LAC LA MIA MIN NE NO NYG NYJ PHI PIT SF SEA TB TEN WAS OAK SD STL'.split(' '));
export function canonicalNfl(code = '') { return ({ JAC:'JAX', LAR:'LA', WSH:'WAS' } as Record<string,string>)[code] || code; }
export function nflPeriodForDate(games: Game[], date: string, season: number) {
  const target=Date.parse(`${date}T12:00:00Z`);
  const candidates=games.filter(g=>g.season===season && g.kickoff_utc && Number.isFinite(Date.parse(g.kickoff_utc)))
    .sort((a,b)=>Math.abs(Date.parse(a.kickoff_utc!)-target)-Math.abs(Date.parse(b.kickoff_utc!)-target));
  const game=candidates[0];
  if (!game || Math.abs(Date.parse(game.kickoff_utc!)-target)>7*86400000) return null;
  const regularWeeks=season>=2021?18:17;
  const phase=game.game_type==='REG'?'reg':['WC','DIV','CON','SB','POST'].includes(game.game_type||'')?'post':null;
  if (!phase) return null;
  return {phase,week:phase==='post'?game.week-regularWeeks:game.week};
}
const numeric = (v: unknown, min: number, max: number): number | null => typeof v === 'number' && Number.isFinite(v) && v >= min && v <= max ? v : null;
const odds = (v: unknown) => typeof v === 'number' && Math.abs(v) >= 100 ? numeric(v,-100000,100000) : null;
export function normalizeNflLive(payload: unknown, capturedAt: string): LiveGame[] {
  if (!payload || typeof payload !== 'object' || !Array.isArray((payload as {games?:unknown}).games)) throw Error('Respuesta NFL inválida.');
  const result: LiveGame[] = [];
  for (const g of (payload as {games:RawGame[]}).games) {
    if (!g || g.league_name !== 'nfl' || !Number.isInteger(g.id) || !Number.isInteger(g.season) || !Number.isInteger(g.week) || !g.start_time || !Number.isFinite(Date.parse(g.start_time))) continue;
    const home = canonicalNfl(g.teams?.find(t=>t.id===g.home_team_id)?.abbr), away = canonicalNfl(g.teams?.find(t=>t.id===g.away_team_id)?.abbr);
    if (!teams.has(home) || !teams.has(away) || home===away) continue;
    const state = g.status || '';
    const status = ['complete','closed','final'].includes(state) ? 'final' : ['inprogress','in_progress','live','halftime'].includes(state) ? 'live' : 'upcoming';
    const line = (g.odds || []).filter(o=>o.type==='game' && Number.isInteger(o.book_id)).sort((a,b)=>(Date.parse(b.inserted||'')||0)-(Date.parse(a.inserted||'')||0))[0];
    result.push({ game_id:`action_${g.id}`,sourceId:String(g.id),season:g.season!,week:g.type==='post'?g.week!+(g.season!>=2021?18:17):g.week!,game_type:g.type==='post'?['WC','DIV','CON','SB'][g.week!-1]||'POST':g.type?.toUpperCase()||null,home_team:home,away_team:away,kickoff_utc:g.start_time,status,
      home_score:status==='upcoming'?null:numeric(g.boxscore?.total_home_points,0,150),away_score:status==='upcoming'?null:numeric(g.boxscore?.total_away_points,0,150),
      sourceStatus:status==='final'?'Final':status==='live'?'En juego':({scheduled:'Programado',postponed:'Pospuesto',cancelled:'Cancelado',canceled:'Cancelado',suspended:'Suspendido'} as Record<string,string>)[state]||g.status_display||state,capturedAt,clock:typeof g.boxscore?.clock==='string'?g.boxscore.clock:null,period:numeric(g.boxscore?.period,0,20),
      liveMarket:line?{ home:odds(line.ml_home),away:odds(line.ml_away),homeSpread:numeric(line.spread_home,-100,100),total:numeric(line.total,1,150),bookId:line.book_id!,updatedAt:line.inserted && Number.isFinite(Date.parse(line.inserted))?line.inserted:null }:null });
  }
  if ((payload as {games:unknown[]}).games.length && !result.length) throw Error('No se pudieron interpretar los partidos NFL.');
  return result;
}

// Only scores/scheduling change. Model probabilities and their original betting
// lines remain frozen, so live markets cannot rewrite historical pick results.
export function mergeNflLive<T extends Game & { filters: string[] }>(snapshots: T[], updates: LiveGame[]): (T & Partial<LiveGame>)[] {
  const merged: (T & Partial<LiveGame>)[] = snapshots.map(g=>({...g}));
  for (const live of updates) {
    const candidates = merged.filter(g=>g.sourceId===live.sourceId || (g.season===live.season && canonicalNfl(g.home_team)===live.home_team && canonicalNfl(g.away_team)===live.away_team && g.kickoff_utc && Math.abs(Date.parse(g.kickoff_utc)-Date.parse(live.kickoff_utc!))<=3*86400000));
    const old = candidates.length===1 ? candidates[0] : null;
    if (old) {
      if (old.status==='final' && live.status!=='final') continue;
      Object.assign(old,{sourceId:live.sourceId,sourceStatus:live.sourceStatus,capturedAt:live.capturedAt,clock:live.clock,period:live.period,liveMarket:live.liveMarket,kickoff_utc:live.kickoff_utc,status:live.status,
        home_score:live.home_score??old.home_score,away_score:live.away_score??old.away_score});
    } else merged.push({...live,filters:[],model_home_win_prob:null,model_cover_home_prob:null,model_total_over_prob:null} as unknown as T & LiveGame);
  }
  return merged;
}

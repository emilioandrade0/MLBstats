import { NextResponse } from 'next/server';
import { readNflLive, saveNflLive } from '../../../../db/nfl-live';
import { normalizeNflLive, type LiveResponse } from '../../../../lib/nfl-live';
export const dynamic='force-dynamic';
export async function GET(request: Request) {
  const url=new URL(request.url), date=url.searchParams.get('date')||'', season=Number(url.searchParams.get('season'));
  const week=Number(url.searchParams.get('week')), phase=url.searchParams.get('phase')||'reg';
  if (!Number.isInteger(week) || week<1 || week>(phase==='post'?4:18) || !['reg','post'].includes(phase)) return NextResponse.json({error:'Semana o fase inválida.'},{status:400});
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date) || !Number.isFinite(Date.parse(date)) || new Date(date).toISOString().slice(0,10)!==date || !Number.isInteger(season) || season<2000 || season>new Date().getUTCFullYear()+1 || Number(date.slice(0,4))<season || Number(date.slice(0,4))>season+1) return NextResponse.json({error:'Fecha o temporada inválida.'},{status:400});
  let saved: Awaited<ReturnType<typeof readNflLive>>={games:[],checkedAt:null}; let persistence=true;
  const cacheKey=`${date}/${season}/${phase}/${week}`;
  try { saved=await readNflLive(season,cacheKey); } catch { persistence=false; }
  let result: LiveResponse={...saved,date,persistence,state:saved.games.length?'SAVED':'UNAVAILABLE'};
  if (saved.checkedAt && Date.now()-Date.parse(saved.checkedAt)<45000) result.state='LIVE';
  else {
    try {
      const response=await fetch(`https://api.actionnetwork.com/web/v1/scoreboard/nfl?period=game&season=${season}&week=${week}&seasonType=${phase}`,{headers:{Accept:'application/json','User-Agent':'StatsMLB/1.0 (+daily-results-and-odds-refresh)'},signal:AbortSignal.timeout(12000),cache:'no-store'});
      if (!response.ok) throw Error(`Fuente NFL respondió HTTP ${response.status}.`);
      const checkedAt=new Date().toISOString();
      const normalized=normalizeNflLive(await response.json(),checkedAt);
      const expectedWeek=phase==='post'?week+(season>=2021?18:17):week;
      if(normalized.some(g=>g.season!==season || g.week!==expectedWeek))throw Error('La fuente devolvió otra jornada.');
      const fetched=normalized.map(g=>{
        const previous=saved.games.find(old=>old.sourceId===g.sourceId);
        return previous && g.status==='final' ? {...g,home_score:g.home_score??previous.home_score,away_score:g.away_score??previous.away_score} : g;
      });
      const byId=new Map(saved.games.map(g=>[g.sourceId,g]));
      for (const g of fetched) if (!(byId.get(g.sourceId)?.status==='final' && g.status!=='final')) byId.set(g.sourceId,g);
      try { await saveNflLive(fetched,cacheKey,checkedAt); persistence=true; } catch { persistence=false; }
      result={games:[...byId.values()],state:'LIVE',checkedAt,date,persistence};
    } catch { result.error='No se pudo consultar la fuente NFL. Se conservan los últimos datos disponibles; no son marcadores en vivo.'; }
  }
  return NextResponse.json(result,{headers:{'Cache-Control':'no-store'}});
}

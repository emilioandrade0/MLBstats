'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { Activity, BarChart3, CalendarDays, CircleDot, FlaskConical, LineChart, RefreshCw, Send } from 'lucide-react';
import SportSwitch from '../components/SportSwitch';
import { moveFocus } from '../components/DashboardTabs';
import { moneylinePick, spreadPick, totalPick, pickResult, americanToDecimal, type PickMode } from '../../lib/nfl/picks';
import type { Game } from '../../lib/nfl/api';
import { TEAMS, teamLogo } from '../../lib/nfl/teams';
import type { NflTelegramPick } from '../../lib/nfl-telegram';
import TelegramDialog from './TelegramDialog';
import './nfl.css';
import { nflPeriodForDate, mergeNflLive, type LiveGame, type LiveResponse } from '../../lib/nfl-live';

type NflGame = Game & Partial<LiveGame> & { filters: string[]; lvd?: { pick?: string; confidence?: number; p_local?: number; p_visita?: number; p_diferencia?: number; spread_source?: string } | null; [key: string]: unknown };
type Season = { season: number; games: number; evaluated: number; accuracy: number | null; brier: number | null };
type Manifest = { generatedAt: string; source: string; seasons: Season[]; filters: Record<string, { label: string; filters: [string,string][] }>; reports: Record<string, Record<string, unknown>[]>; models: { id: string; label: string; target: string; field: string }[] };
const tabs = [{ id:'jornada', label:'Jornada', icon:CalendarDays }, { id:'historial', label:'Historial', icon:Activity }, { id:'comparador', label:'Comparador', icon:BarChart3 }, { id:'movimiento', label:'Movimiento', icon:LineChart }, { id:'rendimiento', label:'Rendimiento', icon:Activity }, { id:'laboratorio', label:'Laboratorio', icon:FlaskConical }];

type NflLineMoveSnap = [string, number | null, number | null, number | null, number | null, number | null];
type NflLineMoveGame = { away: string; home: string; commence: string; opened: string; latest: string; homeImpliedOpen: number; homeImpliedLatest: number; shiftPp: number; spreadOpen: number | null; spreadLatest: number | null; totalOpen: number | null; totalLatest: number | null; snapshots: number; books: Record<string, NflLineMoveSnap[]> };
type NflLineMovePayload = { date: string; generatedAt: string; bookLabels: Record<string, string>; games: Record<string, NflLineMoveGame> };
type NflLineMoveIndex = { generatedAt: string; dates: string[] };

function NflLineMovementPanel({ liveDate }: { liveDate: string }) {
  const [index, setIndex] = useState<NflLineMoveIndex | null>(null);
  const [payload, setPayload] = useState<NflLineMovePayload | null>(null);
  const [selectedDate, setSelectedDate] = useState<string>('');
  useEffect(() => { fetch('/data/nfl/line-movement-index.json', { cache: 'no-store' }).then(r => r.ok ? r.json() as Promise<NflLineMoveIndex> : null).then(v => setIndex(v)).catch(() => {}); }, []);
  useEffect(() => { if (!index?.dates?.length) return; const wanted = index.dates.includes(liveDate) ? liveDate : index.dates[index.dates.length - 1]; setSelectedDate(wanted); }, [index, liveDate]);
  useEffect(() => { if (!selectedDate) return; fetch(`/data/nfl/line-movement/${selectedDate}.json`, { cache: 'no-store' }).then(r => r.ok ? r.json() as Promise<NflLineMovePayload> : null).then(v => setPayload(v)).catch(() => {}); }, [selectedDate]);
  const toDecimal = (a: number | null): string => { if (a == null || !Number.isFinite(a) || a === 0) return '—'; const d = a > 0 ? a/100 + 1 : 100/(-a) + 1; return d.toFixed(2); };
  const sparkline = (series: NflLineMoveSnap[], w = 160, h = 34) => { const pts = series.map(s => s[3]).filter((v): v is number => v != null); if (pts.length < 2) return null; const min = Math.min(...pts), max = Math.max(...pts), range = max - min || .001; const path = pts.map((v, i) => `${i === 0 ? 'M' : 'L'}${((i/(pts.length-1))*w).toFixed(1)},${(h - ((v-min)/range)*h).toFixed(1)}`).join(' '); return <svg width={w} height={h} className="lm-spark"><path d={path} /></svg>; };
  const games = payload?.games ? Object.entries(payload.games) : [];
  return <div className="line-movement-panel">
    <header>
      <div>
        <span className="eyebrow">🏈 Movimiento de línea NFL</span>
        <h3>Cómo se movieron los momios · {selectedDate || '—'}</h3>
      </div>
      <div className="line-movement-meta"><strong>{games.length}</strong><small>juegos con datos</small></div>
    </header>
    {index?.dates.length ? <div className="nfl-controls"><label>Fecha del snapshot<select value={selectedDate} onChange={e => setSelectedDate(e.target.value)}>{index.dates.map(d => <option key={d} value={d}>{d}</option>)}</select></label></div> : null}
    {games.length === 0 ? <div className="line-movement-empty"><p>Aún no hay snapshots capturados para esta fecha.</p>{index && <small>Fechas disponibles: {index.dates.length} · última: {index.dates[index.dates.length - 1] ?? '—'}</small>}</div> : <div className="line-movement-grid">{games.map(([gid, g]) => {
      const shift = g.shiftPp;
      const shiftClass = Math.abs(shift) < .5 ? 'lm-shift-flat' : shift > 0 ? 'lm-shift-up' : 'lm-shift-down';
      const primary = Object.values(g.books)[0] ?? [];
      const bookRows = Object.entries(g.books).filter(([, s]) => s.length > 0).map(([bk, s]) => { const first = s[0], last = s[s.length - 1]; const fi = first[3], li = last[3]; const bookShift = fi != null && li != null ? (li - fi) * 100 : null; return { book: bk, first, last, bookShift }; });
      return <article key={gid} className="line-movement-card">
        <header>
          <div className="lm-teams"><span>{g.away}</span><i>@</i><span>{g.home}</span></div>
          <div className={`lm-shift ${shiftClass}`}><b>{shift > 0 ? '+' : ''}{shift.toFixed(1)} pp</b><small>{Math.abs(shift) < .5 ? 'sin movimiento' : shift > 0 ? `hacia ${g.home}` : `hacia ${g.away}`}</small></div>
        </header>
        <div className="lm-summary">
          <div><span>Apertura</span><strong>{(g.homeImpliedOpen * 100).toFixed(1)}%</strong><small>{g.opened.slice(5, 16).replace('T', ' ')}</small></div>
          <div className="lm-arrow">{shift > 0 ? '→' : shift < 0 ? '←' : '·'}</div>
          <div><span>Actual</span><strong>{(g.homeImpliedLatest * 100).toFixed(1)}%</strong><small>{g.latest.slice(5, 16).replace('T', ' ')}</small></div>
        </div>
        {sparkline(primary)}
        <div className="lm-nfl-marketrow"><span>Spread {g.home}</span><b>{g.spreadOpen ?? '—'}</b><i>→</i><b>{g.spreadLatest ?? '—'}</b></div>
        <div className="lm-nfl-marketrow"><span>Total</span><b>{g.totalOpen ?? '—'}</b><i>→</i><b>{g.totalLatest ?? '—'}</b></div>
        <div className="lm-books">{bookRows.map(({ book, first, last, bookShift }) => (
          <div key={book} className="lm-book-row">
            <b>{payload?.bookLabels[book] ?? `Casa #${book}`}</b>
            <span>{toDecimal(first[1])} / {toDecimal(first[2])}</span>
            <i>→</i>
            <span>{toDecimal(last[1])} / {toDecimal(last[2])}</span>
            {bookShift != null && <em className={bookShift > 0 ? 'lm-up' : bookShift < 0 ? 'lm-down' : ''}>{bookShift > 0 ? '+' : ''}{bookShift.toFixed(1)}pp</em>}
          </div>
        ))}</div>
        <footer><small>{g.snapshots} snapshots · abre {g.opened.slice(0, 10)}, kickoff {g.commence.slice(11, 16)} UTC</small></footer>
      </article>;
    })}</div>}
  </div>;
}
const pct = (p: number | null | undefined) => p == null || !Number.isFinite(p) ? '—' : `${(p*100).toFixed(1)} %`;
function localDate(s?: string | null) { return s && Number.isFinite(Date.parse(s)) ? new Intl.DateTimeFormat('en-CA',{timeZone:'America/Mexico_City',year:'numeric',month:'2-digit',day:'2-digit'}).format(new Date(s)) : ''; }
function when(s?: string | null) { return s && Number.isFinite(Date.parse(s)) ? new Intl.DateTimeFormat('es-MX',{timeZone:'America/Mexico_City',dateStyle:'medium',timeStyle:'short'}).format(new Date(s)) : 'Horario por confirmar'; }
function Logo({team}:{team:string}) { return <img src={teamLogo(team)} alt="" width="40" height="40" loading="lazy" onError={e => { e.currentTarget.style.visibility='hidden'; }} />; }
function modelPick(g:NflGame,field:string) { const p=g[field]; return typeof p==='number' && Number.isFinite(p) && p>=0 && p<=1 ? `${p>=.5?g.home_team:g.away_team} · ${pct(Math.max(p,1-p))}` : 'Sin predicción'; }

export default function NflPage() {
  const [liveDate,setLiveDate]=useState(()=>localDate(new Date().toISOString()));
  const [liveData,setLiveData]=useState<(LiveResponse & {season:string})|null>(null);
  const [liveBusy,setLiveBusy]=useState(false);
  const [manifest,setManifest]=useState<Manifest|null>(null);const [games,setGames]=useState<NflGame[]>([]);const [season,setSeason]=useState('');const [week,setWeek]=useState('');const [team,setTeam]=useState('');
  const [tab,setTab]=useState('jornada');const [period,setPeriod]=useState('hoy');const [mode,setMode]=useState<PickMode>('confidence');const [filters,setFilters]=useState<string[]>([]);
  const [loading,setLoading]=useState(true);const [error,setError]=useState('');const [version,setVersion]=useState(0);const [limit,setLimit]=useState(48);const [configured,setConfigured]=useState(false);const [sendingPick,setSendingPick]=useState<NflTelegramPick|null>(null);const [report,setReport]=useState('roi_summary');
  useEffect(()=>{const sync=()=>{const id=window.location.hash.slice(1);setTab(tabs.some(t=>t.id===id)?id:'jornada');};sync();window.addEventListener('hashchange',sync);window.addEventListener('popstate',sync);return()=>{window.removeEventListener('hashchange',sync);window.removeEventListener('popstate',sync);};},[]);
  useEffect(()=>{const controller=new AbortController();fetch('/data/nfl/index.json',{signal:controller.signal,cache:'no-store'}).then(r=>{if(!r.ok)throw Error('No hay exportación NFL. Ejecuta scripts/export_nfl_web.py.');return r.json() as Promise<Manifest>;}).then(m=>{setManifest(m);setSeason(s=>s||String(m.seasons[0]?.season||''));}).catch(e=>{if(e.name!=='AbortError'){setError(e.message);setLoading(false);}});fetch('/api/nfl/telegram',{signal:controller.signal}).then(r=>r.json() as Promise<{configured:boolean}>).then(d=>setConfigured(d.configured===true)).catch(()=>{});return()=>controller.abort();},[version]);
  useEffect(()=>{if(!season)return;const controller=new AbortController();fetch(`/data/nfl/${season}.json`,{signal:controller.signal,cache:'no-store'}).then(r=>{if(!r.ok)throw Error('No se pudo cargar esta temporada.');return r.json() as Promise<{games:NflGame[]}>;}).then(d=>{setGames(d.games);setError('');setLoading(false);}).catch(e=>{if(e.name!=='AbortError'){setError(e.message);setGames([]);setLoading(false);}});return()=>controller.abort();},[season,version]);
  const livePeriod=nflPeriodForDate(games,liveDate,Number(season));
  const requestedWeek=livePeriod?.week, requestedPhase=livePeriod?.phase;
  useEffect(()=>{
    if(!requestedWeek || !requestedPhase || !season || Number(liveDate.slice(0,4))<Number(season) || Number(liveDate.slice(0,4))>Number(season)+1)return;
    const controller=new AbortController();let busy=false;
    // También consultamos la semana anterior (si aplica) para que los finales
    // recientes se reflejen en historial aunque el pipeline local no los tenga.
    const previousWeek=requestedPhase==='reg' && requestedWeek>1 ? requestedWeek-1 : null;
    const refresh=async()=>{
      if(busy || document.hidden)return;busy=true;setLiveBusy(true);
      try{
        const urls=[`/api/nfl/live?date=${liveDate}&season=${season}&week=${requestedWeek}&phase=${requestedPhase}`];
        if(previousWeek)urls.push(`/api/nfl/live?date=${liveDate}&season=${season}&week=${previousWeek}&phase=reg`);
        const responses=await Promise.all(urls.map(u=>fetch(u,{signal:controller.signal,cache:'no-store'}).then(r=>r.ok?r.json() as Promise<LiveResponse>:null).catch(()=>null)));
        const primary=responses[0];
        if(!primary)throw Error('No se pudo actualizar NFL.');
        const combinedGames=[...primary.games,...(responses[1]?.games||[])];
        if(!controller.signal.aborted)setLiveData({...primary,games:combinedGames,season});
      }catch{
        if(!controller.signal.aborted)setLiveData(old=>({...((old?.season===season)?old:{games:[],checkedAt:null}),date:liveDate,season,persistence:old?.persistence??false,state:old?.games.length?'SAVED':'UNAVAILABLE',error:'Conexión no disponible. Se muestran datos guardados.'}));
      }finally{busy=false;if(!controller.signal.aborted)setLiveBusy(false);}
    };
    void refresh();const timer=window.setInterval(()=>void refresh(),60000);
    const resume=()=>{if(!document.hidden)void refresh();};document.addEventListener('visibilitychange',resume);
    return()=>{controller.abort();window.clearInterval(timer);document.removeEventListener('visibilitychange',resume);};
  },[season,liveDate,version,requestedWeek,requestedPhase]);
  const live=liveData?.season===season?liveData:null;
  const activeGames=mergeNflLive(games,live?.games||[]) as NflGame[];
  const today=localDate(new Date().toISOString());
  const selection=activeGames.filter(g=>(!week||g.week===Number(week))&&(!team||g.home_team===team||g.away_team===team)&&filters.every(f=>g.filters.includes(f)));
  const displayed=selection.filter(g=>tab==='historial'?g.status==='final':tab==='jornada'||tab==='comparador'?period==='todos'?true:period==='hoy'?localDate(g.kickoff_utc)===today:period==='fecha'?localDate(g.kickoff_utc)===liveDate:g.status!=='final':true).sort((a,b)=>tab==='historial'?(b.kickoff_utc||'').localeCompare(a.kickoff_utc||''):(a.kickoff_utc||'').localeCompare(b.kickoff_utc||''));
  const summary=manifest?.seasons.find(s=>s.season===Number(season));const evaluated=selection.filter(g=>g.status==='final'&&g.home_score!=null&&g.away_score!=null&&g.home_score!==g.away_score&&g.model_home_win_prob!=null);
  const correct=evaluated.filter(g=>(g.model_home_win_prob!>=.5)===(g.home_score!>g.away_score!)).length;
  function changeSeason(value:string){setSeason(value);setLiveDate(d=>Number(d.slice(0,4))===Number(value)||(Number(d.slice(0,4))===Number(value)+1&&Number(d.slice(5,7))<=2)?d:`${value}-09-10`);setWeek('');setGames([]);setLoading(true);setLimit(48);}
  function navigate(id:string){setTab(id);setLimit(48);window.history.pushState(null,'',`#${id}`);}
  function openPick(g:NflGame,market:NflTelegramPick['market'],side:string,line?:number){setSendingPick({gameId:g.game_id,home:g.home_team,away:g.away_team,kickoff:g.kickoff_utc||'',market,side,line,pickNumber:1,decimalOdds:market==='ML'?(americanToDecimal(side===g.home_team?g.home_moneyline:g.away_moneyline)||0):0});}
  const controls=<div className="nfl-controls"><label>Temporada<select value={season} onChange={e=>changeSeason(e.target.value)}>{manifest?.seasons.map(s=><option key={s.season}>{s.season}</option>)}</select></label><label>Semana<select value={week} onChange={e=>{setWeek(e.target.value);setLimit(48);}}><option value="">Todas</option>{[...new Set(activeGames.map(g=>g.week))].sort((a,b)=>a-b).map(w=><option key={w}>{w}</option>)}</select></label><label>Equipo<select value={team} onChange={e=>{setTeam(e.target.value);setLimit(48);}}><option value="">Todos</option>{[...new Set(activeGames.flatMap(g=>[g.home_team,g.away_team]))].sort().map(t=><option key={t}>{t}</option>)}</select></label>{(tab==='jornada'||tab==='comparador')&&<label>Partidos<select value={period} onChange={e=>{setPeriod(e.target.value);setLimit(48);}}><option value="proximos">No finalizados</option><option value="hoy">Hoy · CDMX</option><option value="fecha">Fecha consultada</option><option value="todos">Todos</option></select></label>}<label>Modo de pick<select value={mode} onChange={e=>setMode(e.target.value as PickMode)}><option value="confidence">Mayor probabilidad</option><option value="value">Valor vs momio</option></select></label></div>;
  return <div className="dashboard nfl-dashboard"><a className="skip-link" href="#nfl-content">Saltar al contenido</a><header className="dashboard-header"><SportSwitch sport="NFL"/><a className="dashboard-brand" href="/nfl"><span className="dashboard-brand-icon">🏈</span><span>STRIKE<span>CAST</span><small>NFL / CENTRO DE ANÁLISIS</small></span></a><div className="dashboard-header-right"><span className={`connection-status ${configured?'connected':''}`}><i/>{configured?'Canal NFL configurado':'Canal NFL pendiente'}</span><button className="dashboard-refresh" disabled={loading} onClick={()=>{setLoading(true);setError('');setVersion(v=>v+1);}}><RefreshCw size={15}/>Actualizar datos NFL</button></div></header>
    <main className="dashboard-body" id="nfl-content"><div className="dashboard-intro"><div><span className="dashboard-overline">NFL INTELLIGENCE / STRIKECAST</span><h1>Otra liga. El mismo análisis.</h1><p>Juegos, picks y evidencia NFL, en tu centro de análisis.</p></div><div className="dashboard-source">{live?.state==='LIVE'?'Action Network · consulta actual':live?.state==='SAVED'?'NFL · datos guardados':'NFL · histórico importado'}<small>Última consulta: {when(live?.checkedAt)} · actualización cada 60 s con la pestaña visible</small></div></div>
    <div className="dashboard-metrics">{[{label:'Temporada',value:season||'—',detail:`${summary?.games||0} partidos guardados`},{label:'Historial evaluado',value:String(evaluated.length),detail:'Finales con predicción ML; sin empates'},{label:'Acierto ML',value:evaluated.length?pct(correct/evaluated.length):'—',detail:'Con los filtros seleccionados'},{label:'Canal independiente',value:'NFL',detail:'Los envíos MLB no cambian'}].map(m=><article key={m.label}><div><span>{m.label}</span></div><strong>{m.value}</strong><small>{m.detail}</small></article>)}</div>
    <div role="tablist" aria-label="Secciones NFL" className="dashboard-tabs" onKeyDown={moveFocus}>{tabs.map(t=><button role="tab" id={`nfl-tab-${t.id}`} aria-controls={`nfl-panel-${t.id}`} aria-selected={tab===t.id} tabIndex={tab===t.id?0:-1} key={t.id} onClick={()=>navigate(t.id)}><t.icon size={17}/>{t.label}</button>)}</div>
    {controls}<div className="nfl-controls nfl-live-controls"><label>Consultar jornada por fecha<input type="date" value={liveDate} min={`${season||2000}-01-01`} max={`${Number(season||2000)+1}-02-28`} onChange={e=>{if(e.target.value)setLiveDate(e.target.value);}}/></label><p role="status">{liveBusy?'Consultando marcadores…':!livePeriod?'No hay una semana identificada cerca de esta fecha en el calendario guardado.':live?.state==='LIVE'?'Semana consultada actualizada. Los demás juegos conservan su última observación.':'Mostrando información guardada.'}</p></div>{live?.error&&<p className="nfl-warning" role="status">{live.error}</p>}{live&&!live.persistence&&<p className="nfl-warning">Sin guardado persistente: revisa la conexión D1 del servidor.</p>}<p className="nfl-data-note">Calendario, marcadores y momios se consultan en línea. Las predicciones siguen siendo las del modelo exportado: no se generan picks nuevos al actualizar. Las líneas de evaluación originales no se sustituyen por momios en vivo.</p><details className="nfl-filter-details"><summary>Filtros situacionales ({filters.length} activos)</summary><p>Se conservan los filtros de la app NFL. Se combinan con AND y no reentrenan el modelo. Las condiciones históricas de clima y mercado pueden no haber estado disponibles al publicar el pick.</p><div className="nfl-filter-grid">{Object.entries(manifest?.filters||{}).map(([id,group])=><fieldset key={id}><legend>{group.label}</legend>{group.filters.map(([key,label])=><label key={key}><input type="checkbox" checked={filters.includes(key)} onChange={e=>{setFilters(f=>e.target.checked?[...f,key]:f.filter(x=>x!==key));setLimit(48);}}/>{label}</label>)}</fieldset>)}</div><button onClick={()=>setFilters([])}>Limpiar filtros</button></details>
    {error&&<p className="nfl-warning" role="alert">{error}</p>}{loading&&<p role="status">Cargando temporada…</p>}
    <section role="tabpanel" id={`nfl-panel-${tab}`} aria-labelledby={`nfl-tab-${tab}`} className="section nfl-panel">
    {(tab==='jornada'||tab==='historial')&&<><div className="section-heading"><h2>{tab==='historial'?'Historial de partidos y picks':'Partidos NFL'}</h2><p>{displayed.length} juegos · horarios CDMX · {mode==='value'?'Valor: ML usa cuotas guardadas; spread/total conserva el supuesto 52.4% de la app original.':'Picks del modelo existente, sin nuevos pesos.'}</p></div><div className="nfl-game-grid">{!loading&&displayed.slice(0,limit).map(g=>{const markets=[{label:'Ganador · ML',market:'ML' as const,pick:moneylinePick(g,mode)},{label:'Hándicap',market:'SPREAD' as const,pick:spreadPick(g,mode)},{label:'Total',market:'TOTAL' as const,pick:totalPick(g,mode)}];const mlPick=markets[0].pick;const mlResult=mlPick&&g.status==='final'?pickResult(mlPick,'ml',g):null;const resultCls=mlResult==='win'?'nfl-game-won':mlResult==='loss'?'nfl-game-lost':mlResult==='push'?'nfl-game-push':'';return <article key={g.game_id} className={`nfl-game ${resultCls}`}><header><span>SEMANA {g.week} · {g.game_type}</span><span>{g.sourceStatus||(g.status==='final'?'Final · guardado':g.status==='live'?'En juego · guardado':'Programado · guardado')}</span></header><small>{when(g.kickoff_utc)}{g.capturedAt?` · observado ${when(g.capturedAt)}`:''}{g.status==='live'&&g.clock?` · Q${g.period??'—'} ${g.clock}`:''}</small><div className="nfl-matchup"><div><Logo team={g.away_team}/><strong>{g.away_team}</strong><small>{TEAMS[g.away_team]?.name}</small><b>{g.away_score??'—'}</b></div><span>@</span><div><Logo team={g.home_team}/><strong>{g.home_team}</strong><small>{TEAMS[g.home_team]?.name}</small><b>{g.home_score??'—'}</b></div></div><p>{g.stadium||'Estadio por confirmar'}</p>{g.liveMarket&&<div className="nfl-live-market"><strong>Mercado observado · casa #{g.liveMarket.bookId}</strong><small>ML {g.away_team}: {g.liveMarket.away??'—'} / {g.home_team}: {g.liveMarket.home??'—'} · Spread local {g.liveMarket.homeSpread??'—'} · Total {g.liveMarket.total??'—'}</small><small>Hora de la cuota: {when(g.liveMarket.updatedAt)}. Referencia, no predicción. Confirma el momio antes de enviar.</small></div>}<div className="nfl-pick-list">{markets.map(m=>{const p=m.pick;const result=p&&g.status==='final'?pickResult(p,m.market==='ML'?'ml':m.market==='SPREAD'?'spread':'total',g):null;return <div key={m.market}><span>{m.label}</span>{p?<><strong>{p.side} {m.market==='ML'?'ML':p.line}</strong><small>{pct(p.confidence)}{result?` · ${result==='win'?'Acierto':result==='loss'?'Fallo':'Push'}`:''}{mode==='value'?` · edge ${p.edge==null?'no disponible':p.edge.toFixed(1)+' pts'}${p.recommend?'':' · sin recomendación'}`:''}</small><button aria-label={`Enviar ${m.label} ${g.away_team} vs ${g.home_team} a Telegram NFL`} onClick={()=>openPick(g,m.market==='TOTAL'?(p.side==='OVER'?'OVER':'UNDER'):m.market,p.side,m.market==='ML'?undefined:Number(p.line))}><Send size={13}/>Enviar a Telegram</button></>:<small>Sin predicción{g.model_home_win_prob==null?' guardada para este partido':''}</small>}</div>;})}{g.lvd?.pick&&<div><span>Margen · L/V/D</span><strong>{g.lvd.pick==='L'?'Local gana por >6':g.lvd.pick==='V'?'Visitante gana por >6':'Diferencia ≤6'}</strong><small>{pct(g.lvd.confidence)} · fuente {g.lvd.spread_source||'—'}</small><button onClick={()=>openPick(g,'LVD',g.lvd!.pick!)}><Send size={13}/>Enviar a Telegram</button></div>}</div></article>;})}</div></>}
    {tab==='comparador'&&<><h2>Comparador NFL</h2><p>Un modelo de ganador disponible. Mercado es una referencia, no un modelo independiente. L/V/D predice margen y no vota por el ganador. El registro permite añadir modelos de ganador sin cambiar la tabla.</p><div className="nfl-table"><table><thead><tr><th>Juego / semana</th>{manifest?.models.filter(m=>m.target==='winner').map(m=><th key={m.id}>{m.label}</th>)}<th>Mercado sin margen · referencia</th><th>L/V/D</th><th>Resultado</th></tr></thead><tbody>{!loading&&displayed.slice(0,limit).map(g=>{const hd=americanToDecimal(g.liveMarket?g.liveMarket.home:g.home_moneyline),ad=americanToDecimal(g.liveMarket?g.liveMarket.away:g.away_moneyline);const p=hd&&ad&&Number.isFinite(hd)&&Number.isFinite(ad)?(1/hd)/(1/hd+1/ad):null;return <tr key={g.game_id}><td>{g.away_team} @ {g.home_team}<small>Semana {g.week} · {when(g.kickoff_utc)}</small></td>{manifest?.models.filter(m=>m.target==='winner').map(m=><td key={m.id}>{modelPick(g,m.field)}</td>)}<td>{p==null?'Sin cuotas':`${p>=.5?g.home_team:g.away_team} · ${pct(Math.max(p,1-p))}`}<small>{g.liveMarket?`Cuota observada: ${when(g.liveMarket.updatedAt)}`:'Cuota del archivo original'}</small></td><td>{g.lvd?.pick?`${g.lvd.pick} · ${pct(g.lvd.confidence)}`:'Sin predicción'}</td><td>{g.status==='final'?`${g.away_score}–${g.home_score}`:'Pendiente'}</td></tr>;})}</tbody></table></div></>}
    {tab==='movimiento'&&<NflLineMovementPanel liveDate={liveDate} />}
    {tab==='rendimiento'&&<><h2>Rendimiento histórico del modelo</h2><p>Resumen por temporada del snapshot completo (no aplica los filtros superiores). Los empates se excluyen de accuracy/Brier ML. Son métricas del archivo original; esta integración no certifica su entrenamiento.</p><div className="nfl-table"><table><thead><tr><th>Temporada</th><th>Juegos evaluados</th><th>Accuracy ML</th><th>Brier ↓</th></tr></thead><tbody>{manifest?.seasons.map(s=><tr key={s.season}><td>{s.season}</td><td>{s.evaluated}</td><td>{pct(s.accuracy)}</td><td>{s.brier?.toFixed(4)??'—'}</td></tr>)}</tbody></table></div></>}
    {tab==='laboratorio'&&<><h2>ROI y patrones de la app NFL</h2><p>Informes originales conservados. Son retrospectivos; ROI no garantiza beneficios futuros. Esta vista no aplica los filtros superiores.</p><label>Informe<select value={report} onChange={e=>setReport(e.target.value)}>{Object.keys(manifest?.reports||{}).map(r=><option key={r} value={r}>{r.replaceAll('_',' ')}</option>)}</select></label><div className="nfl-table"><table><thead><tr>{Object.keys(manifest?.reports[report]?.[0]||{}).map(k=><th key={k}>{k.replaceAll('_',' ')}</th>)}</tr></thead><tbody>{manifest?.reports[report]?.slice(0,100).map((row,i)=><tr key={i}>{Object.entries(row).map(([key,value])=><td key={key}>{typeof value==='number'?Number.isInteger(value)?value:value.toFixed(2):String(value??'—')}</td>)}</tr>)}</tbody></table></div><a href="/data/nfl/index.json" download>Descargar todos los informes, filtros y registro de modelos</a></>}
    {['jornada','historial','comparador'].includes(tab)&&!loading&&(!displayed.length?<div className="empty-state"><h3>No hay partidos con estos filtros</h3><p>Prueba otra temporada, semana o selección de fechas. La ausencia de predicción no se reemplaza por un pick inventado.</p></div>:displayed.length>limit&&<button className="dashboard-refresh" onClick={()=>setLimit(n=>n+48)}>Mostrar más ({Math.min(limit,displayed.length)}/{displayed.length})</button>)}
    </section><footer className="dashboard-footer"><span><CircleDot size={14}/>STRIKECAST NFL</span><p>Datos separados. Modelos originales. Estimaciones, no garantías.</p><Link href="/#hoy">Volver a MLB</Link></footer></main>{sendingPick&&<TelegramDialog pick={sendingPick} configured={configured} close={()=>setSendingPick(null)}/>}</div>;
}

const ui = { date: null, dailyRequest: 0 };
const TEAM_IDS = { ARI:109, AZ:109, ATL:144, BAL:110, BOS:111, CHC:112, CHW:145, CWS:145, CIN:113, CLE:114, COL:115, DET:116, HOU:117, KC:118, LAA:108, LAD:119, MIA:146, MIL:158, MIN:142, NYM:121, NYY:147, OAK:133, ATH:133, PHI:143, PIT:134, SD:135, SF:137, SEA:136, STL:138, TB:139, TEX:140, TOR:141, WSH:120 };
const $ = id => document.getElementById(id);
const pct = v => v == null ? '—' : `${(Number(v) * 100).toFixed(1)}%`;
const num = (v, d=1) => v == null ? '—' : Number(v).toFixed(d);
const logo = team => `https://www.mlbstatic.com/team-logos/${TEAM_IDS[team] || 0}.svg`;
const localToday = () => { const d=new Date(); return `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}`; };
function shift(iso, n) { const [y,m,d]=iso.split('-').map(Number); const x=new Date(y,m-1,d); x.setDate(x.getDate()+n); return `${x.getFullYear()}-${String(x.getMonth()+1).padStart(2,'0')}-${String(x.getDate()).padStart(2,'0')}`; }
function metric(g, side, name) { return g[side]?.[name] ?? g[`${side}_${name}`] ?? null; }
function qualityLabel(q) { return ({chronological_oos:'OOS cronológico',pregame:'Captura pregame',retrospective_backfill:'Backfill retrospectivo',late_capture:'Captura tardía',unknown_time:'Hora desconocida'})[q] || q || 'Sin clasificar'; }
function resultLabel(r) { return ({correct:'ACIERTO',incorrect:'FALLO',pending:'PENDIENTE'})[r] || 'PENDIENTE'; }
function outcomeBadge(value) {
  if (value == null) return '';
  return `<small class="mini-outcome ${value ? 'correct' : 'incorrect'}">${value ? 'ACIERTO' : 'FALLO'}</small>`;
}
function totalsRow(g) {
  const line = g.market_over_under, pOver = g.sim_p_over_cal;
  if (line == null || pOver == null) {
    return `<div class="totals-row unavailable"><span>TOTAL O/U</span> <em>línea no disponible pregame</em></div>`;
  }
  const pick = pOver >= 0.5 ? 'OVER' : 'UNDER';
  const edge = Math.abs(pOver - 0.5);
  const tier = edge >= 0.10 ? 'fuerte' : edge >= 0.05 ? 'ligero' : 'marginal';
  const actual = g.actual_totals_result;
  const actualStr = actual == null ? '' : (actual === 'push'
    ? ` · <em>push (${num(g.actual_total_runs,0)})</em>`
    : ` · <em>real ${actual.toUpperCase()} (${num(g.actual_total_runs,0)})</em>${outcomeBadge(g.totals_correct)}`);
  return `<div class="totals-row tier-${tier}"><span>TOTAL O/U ${num(line,1)}</span> <b>${pick}</b> · ${pct(pOver)} · <small>edge ${tier}</small>${actualStr}</div>`;
}
function componentCell(g, side, name) {
  const predicted=metric(g,side,name), actual=g[`actual_${side}_${name}`];
  const correct=g[`${side}_${name}_correct`];
  if (actual == null) return `<span class="component-value"><strong>${num(predicted)}</strong><small>real —</small></span>`;
  return `<span class="component-value"><strong>${num(predicted)} → ${num(actual,0)}</strong>${outcomeBadge(correct)}</span>`;
}

async function jsonFetch(url) {
  const res = await fetch(url, {cache:'no-store'});
  if (!res.ok) { let msg=`HTTP ${res.status}`; try { const x=await res.json(); msg=x.detail || msg; } catch(_){} throw new Error(msg); }
  return res.json();
}

function renderSummary(s={}) {
  const values = [s.games ?? 0, pct(s.agreement_rate), s.graded ?? 0, pct(s.accuracy)];
  [...$('daily-summary').querySelectorAll('strong')].forEach((node,i)=>node.textContent=values[i]);
}

function card(g) {
  const away=g.away_abbrev, home=g.home_abbrev;
  const ar=metric(g,'away','runs'), hr=metric(g,'home','runs');
  const fsHome=g.first_score_p_home;
  const fsPick=g.first_score_pick || (fsHome == null ? '—' : fsHome >= .5 ? home : away);
  const result=g.result || 'pending';
  return `<article class="sim-card ${result}">
    <div class="card-top"><span>${g.status || `GAME ${g.game_pk}`}</span><span class="quality ${g.capture_quality || ''}">${qualityLabel(g.capture_quality)}</span></div>
    <div class="matchup">
      <div class="team"><img src="${logo(away)}" alt="${away}"><b>${away}</b></div>
      <div class="score-projection"><strong>${num(ar)} – ${num(hr)}</strong><small>CARRERAS ESP.</small></div>
      <div class="team"><img src="${logo(home)}" alt="${home}"><b>${home}</b></div>
    </div>
    <div class="pick-row">
      <div class="pick-box sim"><span>SIMULACIÓN · ${pct(g.sim_p_home)} HOME</span><b>${g.simulation_pick || '—'}</b>${outcomeBadge(g.simulation_correct)}</div>
      <div class="pick-box"><span>PICK OFICIAL · ${pct(g.official_p_home)}</span><b>${g.official_pick || '—'}</b>${outcomeBadge(g.official_correct)}</div>
    </div>
    <table class="component-table"><thead><tr><th>COMPONENTE</th><th>${away}</th><th>${home}</th></tr></thead><tbody>
      <tr><th>Hits</th><td>${componentCell(g,'away','hits')}</td><td>${componentCell(g,'home','hits')}</td></tr>
      <tr><th>Corredores</th><td>${componentCell(g,'away','baserunners')}</td><td>${componentCell(g,'home','baserunners')}</td></tr>
      <tr><th>Bases totales</th><td>${componentCell(g,'away','total_bases')}</td><td>${componentCell(g,'home','total_bases')}</td></tr>
      <tr><th>Dejados en base</th><td>${componentCell(g,'away','left_on_base')}</td><td>${componentCell(g,'home','left_on_base')}</td></tr>
    </tbody></table>
    <div class="first-score"><span>PRIMERA ANOTACIÓN</span> <b>${fsPick}</b>${fsHome == null ? '' : ` · ${pct(fsHome)} local`}${g.actual_first_score_pick ? ` <em>real ${g.actual_first_score_pick}</em>${outcomeBadge(g.first_score_correct)}` : ''}</div>
    ${totalsRow(g)}
    ${result === 'pending' ? '' : `<div class="card-result"><span class="result ${result}">${resultLabel(result)}</span> · Ganó ${g.actual_winner}</div>`}
  </article>`;
}

async function loadDaily(date, silent=false) {
  const requestId = ++ui.dailyRequest;
  ui.date=date; $('date-input').value=date; $('slate-title').textContent=`Simulaciones · ${date}`;
  if (!silent) { $('daily-status').textContent='Entrenando / cargando motores ligados…'; $('simulation-grid').innerHTML=''; }
  try {
    const data=await jsonFetch(`/api/simulations?sim_date=${date}&t=${Date.now()}`);
    if (requestId !== ui.dailyRequest || date !== ui.date) return;
    renderSummary(data.summary);
    $('lineage').textContent=`${data.engine_id || 'histórico'} · ${data.simulator_version || data.mode} · modo sombra`;
    $('daily-status').textContent=data.games.length ? `${data.games.length} simulaciones. Los picks oficiales permanecen intactos.` : 'No existe evidencia OOS ni captura para esta fecha.';
    $('simulation-grid').innerHTML=data.games.map(card).join('');
  } catch(err) {
    if (requestId !== ui.dailyRequest || date !== ui.date) return;
    $('daily-status').textContent=`No se pudieron cargar las simulaciones: ${err.message}`;
    renderSummary({});
  }
}

function renderHistory(data) {
  const s=data.summary;
  $('history-kpis').innerHTML=`<span>OOS <b>${s.oos_games}</b> · <b>${pct(s.oos_accuracy)}</b></span><span>Prospectivas válidas <b>${s.prospective_games}</b> · <b>${pct(s.prospective_accuracy)}</b></span><span>No prospectivas excluidas <b>${s.late_captures_excluded}</b></span><span>Total visible <b>${s.games}</b></span>`;
  $('history-body').innerHTML=data.games.length ? data.games.map(g=>`<tr>
    <td>${g.date}</td><td><b>${g.away_abbrev} @ ${g.home_abbrev}</b><br><small>#${g.game_pk}</small></td>
    <td><b>${g.simulation_pick || '—'}</b><br><small>${pct(g.sim_p_home)} local</small></td>
    <td>${g.official_pick || '—'}</td>
    <td>${g.actual_away_score == null ? '—' : `${num(g.actual_away_score,0)} – ${num(g.actual_home_score,0)}`}</td>
    <td>${qualityLabel(g.capture_quality)}</td><td><span class="result ${g.result}">${resultLabel(g.result)}</span></td>
  </tr>`).join('') : '<tr><td colspan="7">No hay filas con estos filtros.</td></tr>';
}

async function loadHistory() {
  const params=new URLSearchParams({result:$('history-result').value,limit:'250'});
  if ($('history-start').value) params.set('start',$('history-start').value);
  if ($('history-end').value) params.set('end',$('history-end').value);
  $('history-body').innerHTML='<tr><td colspan="7">Cargando historial…</td></tr>';
  try { renderHistory(await jsonFetch(`/api/simulations/history?${params}`)); }
  catch(err) { $('history-body').innerHTML=`<tr><td colspan="7">Error: ${err.message}</td></tr>`; }
}

$('prev-day').onclick=()=>loadDaily(shift(ui.date,-1));
$('next-day').onclick=()=>loadDaily(shift(ui.date,1));
$('today').onclick=()=>loadDaily(localToday());
$('reload').onclick=()=>{ loadDaily(ui.date); loadHistory(); };
$('date-input').onchange=e=>loadDaily(e.target.value);
$('history-apply').onclick=loadHistory;

const today=localToday();
$('history-start').value=shift(today,-30); $('history-end').value=today;
loadDaily(today); loadHistory();

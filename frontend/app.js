// State + helpers
const state = { date: null, detailPk: null, liveTimer: null, listTimer: null };
const fmtPct = p => p == null ? '-' : (p * 100).toFixed(1) + '%';
const fmtML = ml => ml == null ? '-' : (ml > 0 ? '+' + ml : '' + ml);
const fmtNum = (n, d = 2) => n == null ? '-' : Number(n).toFixed(d);
const fmt3 = n => n == null ? '-' : Number(n).toFixed(3).slice(1);
// Local-date ISO (YYYY-MM-DD) — NOT UTC. `new Date().toISOString()` shifts to UTC,
// which in the Americas (UTC-5 to -8) jumps to tomorrow during evening hours.
function todayISO() {
  const d = new Date();
  const yyyy = d.getFullYear();
  const mm = String(d.getMonth() + 1).padStart(2, '0');
  const dd = String(d.getDate()).padStart(2, '0');
  return `${yyyy}-${mm}-${dd}`;
}

// Add/subtract days from a YYYY-MM-DD string without ever crossing into UTC math.
function shiftDate(iso, days) {
  const [y, m, d] = iso.split('-').map(Number);
  const dt = new Date(y, m - 1, d);  // local midnight, no TZ surprise
  dt.setDate(dt.getDate() + days);
  const yyyy = dt.getFullYear();
  const mm = String(dt.getMonth() + 1).padStart(2, '0');
  const dd = String(dt.getDate()).padStart(2, '0');
  return `${yyyy}-${mm}-${dd}`;
}

function pickSide(p) {
  if (p == null) return 'NONE';
  if (p >= 0.5) return 'HOME';
  return 'AWAY';
}

// Strong-pick threshold: only highlight cards when the model leans clearly to one side.
function pickConfident(p, minEdge = 0.03) {
  if (p == null) return 'NONE';
  if (p > 0.5 + minEdge) return 'HOME';
  if (p < 0.5 - minEdge) return 'AWAY';
  return 'NONE';
}

function mlVerdict(pick, homeScore, awayScore) {
  if (pick === 'NONE') return 'none';
  if (homeScore == null || awayScore == null) return 'tbd';
  if (homeScore === awayScore) return 'push';
  const homeWon = homeScore > awayScore;
  const correct = (pick === 'HOME' && homeWon) || (pick === 'AWAY' && !homeWon);
  return correct ? 'win' : 'loss';
}

function ouPick(predTotal, marketTotal, edgeMin = 0.3) {
  if (predTotal == null || marketTotal == null) return { pick: 'NONE', edge: null };
  const edge = predTotal - marketTotal;
  if (Math.abs(edge) < edgeMin) return { pick: 'NONE', edge };
  return { pick: edge > 0 ? 'OVER' : 'UNDER', edge };
}

function ouVerdict(pick, marketTotal, homeScore, awayScore) {
  if (pick === 'NONE') return 'none';
  if (homeScore == null || awayScore == null || marketTotal == null) return 'tbd';
  const actual = homeScore + awayScore;
  if (actual === marketTotal) return 'push';
  const wentOver = actual > marketTotal;
  const correct = (pick === 'OVER' && wentOver) || (pick === 'UNDER' && !wentOver);
  return correct ? 'win' : 'loss';
}

function settledClass(verdict) {
  return ['win', 'loss', 'push'].includes(verdict) ? verdict : '';
}

function verdictLabel(verdict, win = 'GANO', loss = 'PERDIO') {
  return { win, loss, push: 'PUSH', tbd: 'TBD', none: '' }[verdict] || '';
}

function flipSummaryText(flip) {
  if (!flip) return 'Sin datos previos';
  return flip.summary || 'Sin datos previos';
}

async function loadDate(d, silent = false) {
  state.date = d;
  document.getElementById('datepick').value = d;
  if (!silent) document.getElementById('status').textContent = 'Cargando partidos...';
  const response = await fetch(`/api/games?start=${d}&end=${d}&t=${Date.now()}`);
  const games = await response.json();
  renderList(games);
  scheduleListAutoRefresh(d, games);
}

// Auto-refresh the list view every 30s when there are live games on the visible date,
// or once a minute when looking at today (so finals + new lineups roll in).
function scheduleListAutoRefresh(d, games) {
  if (state.listTimer) { clearInterval(state.listTimer); state.listTimer = null; }
  const today = todayISO();
  const anyLive = games.some(g => g.is_live);
  if (!(anyLive || d === today)) return;
  const intervalMs = anyLive ? 30_000 : 60_000;
  state.listTimer = setInterval(() => {
    // Only refresh if still on the list view (not in detail) and same date selected
    const listVisible = !document.getElementById('list-view').classList.contains('hidden');
    if (!listVisible || state.date !== d) return;
    loadDate(d, true);
  }, intervalMs);
}

function renderList(games) {
  const grid = document.getElementById('grid');
  const status = document.getElementById('status');
  if (!games.length) {
    grid.innerHTML = '';
    status.innerHTML = '<div class="empty">Sin partidos en esta fecha.</div>';
    return;
  }

  status.textContent = `${games.length} partidos - ${state.date}`;
  grid.innerHTML = games.map(g => {
    const pH = g.p_home;
    const pA = g.p_away;
    const originalPick = g.pick_original_xgboost || {};
    const pickH = originalPick.side === 'HOME' ? 'HOME' : originalPick.side === 'AWAY' ? 'AWAY' : pickSide(pH);
    const homeLogo = `https://www.mlbstatic.com/team-logos/${teamId(g.home_abbrev)}.svg`;
    const awayLogo = `https://www.mlbstatic.com/team-logos/${teamId(g.away_abbrev)}.svg`;
    const played = g.is_played ? 'played' : '';
    const live = g.live_score || {};
    const liveAway = live.away || {};
    const liveHome = live.home || {};
    const isLive = Boolean(g.is_live);
    const displayAwayScore = isLive && liveAway.runs != null ? liveAway.runs : g.away_score;
    const displayHomeScore = isLive && liveHome.runs != null ? liveHome.runs : g.home_score;
    const mlResult = g.is_played ? mlVerdict(pickH, g.home_score, g.away_score) : 'none';
    const totalPick = ouPick(g.pred_total, g.market_total);
    const totalResult = g.is_played ? ouVerdict(totalPick.pick, g.market_total, g.home_score, g.away_score) : 'none';
    // Always show the model's pick (team abbrev from backend, falls back to local computation).
    const pickAbbr = g.pick_abbrev || (pickH === 'HOME' ? g.home_abbrev : pickH === 'AWAY' ? g.away_abbrev : null);
    const pickDec = pickH === 'HOME' ? g.market_home_decimal : g.market_away_decimal;
    const mlLabel = pickAbbr
      ? (pickDec != null ? `ML: ${pickAbbr} @ ${pickDec.toFixed(2)}` : `ML: ${pickAbbr}`)
      : 'ML: -';
    const mlResultLabel = verdictLabel(mlResult);
    // VALUE pick — only relevant when the value side is the UNDERDOG with meaningful edge.
    // When the value pick aligns with the market favorite, it adds no info beyond ML.
    const valueSide = g.value_side; // 'HOME' | 'AWAY' | null
    const valueAbbr = g.value_team_abbrev;
    const valueDec  = g.value_decimal;
    const valueEdge = g.value_edge_pp;
    const valueImpliedProb = valueSide === 'HOME' ? g.market_home_implied
                          : valueSide === 'AWAY' ? g.market_away_implied : null;
    const valueIsUnderdog = valueImpliedProb != null && valueImpliedProb < 0.50;
    const EDGE_MIN = 2.0; // pp — anything smaller is noise
    const showValuePick = valueIsUnderdog && valueAbbr && valueEdge != null && valueEdge >= EDGE_MIN;
    const valueResult = (g.is_played && valueSide)
      ? mlVerdict(valueSide, g.home_score, g.away_score) : 'none';
    const valueResultLabel = verdictLabel(valueResult);
    const valueLabel = showValuePick
      ? `🎯 ${valueAbbr} @ ${valueDec?.toFixed(2)}  +${valueEdge?.toFixed(1)}pp`
      : null;
    const flipSummary = flipSummaryText(g.pick_invertido) || g.flip_summary || 'Sin datos previos';
    const flipConflict = Boolean(g.conflict_flip || (g.pick_invertido && g.pick_invertido.conflict_flip));
    // Card border = verdict of the pick we're actually showing.
    const shownPickResult = showValuePick ? valueResult : mlResult;
    const cardResult = settledClass(shownPickResult) || settledClass(totalResult);
    // Center scoreboard block (between the two team logos when the game has started)
    const hasScore = (isLive || g.is_played) &&
                     displayAwayScore != null && displayHomeScore != null;
    const centerBlock = hasScore
      ? `<div class="score-center">
           <span class="sc-away">${displayAwayScore}</span>
           <span class="sc-sep">-</span>
           <span class="sc-home">${displayHomeScore}</span>
           ${isLive ? `<small class="sc-live-tag">EN VIVO</small>` : ''}
         </div>`
      : `<span class="vs">VS</span>`;
    const totalLabel = totalPick.pick === 'NONE'
      ? (g.pred_total != null && g.market_total != null
          ? `Total: ~${g.market_total.toFixed(1)}`
          : 'Total: -')
      : `Total: ${totalPick.pick} ${g.market_total != null ? Number(g.market_total).toFixed(1) : ''}`;
    const totalResultLabel = verdictLabel(totalResult);
    // Live inning info (e.g. "T6 · 2 outs")
    let liveInningInfo = '';
    if (isLive && live.inning != null) {
      const half = (live.inning_half || '').slice(0, 1).toUpperCase(); // T / B
      const outs = live.outs != null ? `, ${live.outs} outs` : '';
      liveInningInfo = ` <small>(${half}${live.inning}${outs})</small>`;
    }
    const scoreLine = isLive
      ? `<strong class="live-score">EN VIVO ${g.away_abbrev} ${displayAwayScore ?? 0} - ${displayHomeScore ?? 0} ${g.home_abbrev}${liveInningInfo}</strong>`
      : g.is_played
        ? `<strong>${g.away_abbrev} ${g.away_score} - ${g.home_score} ${g.home_abbrev}</strong>`
        : 'Prediccion modelo';

    return `
      <article class="game-card ${played} ${cardResult}" data-pk="${g.game_pk}">
        <div class="gc-header">
          <span>${g.venue || '-'}</span>
          <span>${isLive ? 'EN VIVO' : g.is_played ? 'FINAL' : 'PROXIMO'}</span>
        </div>
        <div class="gc-teams">
          <div class="team-side">
            <img src="${awayLogo}" alt="${g.away_abbrev}" onerror="this.style.opacity=0.2">
            <span class="abbr">${g.away_abbrev}</span>
            <span class="prob">${fmtPct(pA)}</span>
          </div>
          ${centerBlock}
          <div class="team-side">
            <img src="${homeLogo}" alt="${g.home_abbrev}" onerror="this.style.opacity=0.2">
            <span class="abbr">${g.home_abbrev}</span>
            <span class="prob">${fmtPct(pH)}</span>
          </div>
        </div>
        <div class="gc-bars">
          <div class="prob-bar">
            <div class="away" style="width:${(pA || 0.5) * 100}%"></div>
            <div class="home" style="width:${(pH || 0.5) * 100}%"></div>
          </div>
        </div>
        <div class="gc-footer">
          <span>${scoreLine}</span>
          <span class="pick-stack">
            ${showValuePick
              ? `<span class="pick VALUE" title="Underdog con edge +${valueEdge?.toFixed(1)}pp vs mercado">${valueLabel}</span>
                 ${valueResultLabel ? `<span class="result-badge ${valueResult}">${valueResultLabel}</span>` : ''}`
              : `<span class="pick ${pickH}">${mlLabel}</span>
                 ${mlResultLabel ? `<span class="result-badge ${mlResult}">${mlResultLabel}</span>` : ''}`}
            <span class="pick FLIP" title="Resultado histórico del mismo día hace un año">Flip: ${flipSummary}</span>
            ${flipConflict ? `<span class="result-badge loss">CONFLICTO FLIP</span>` : ''}
            <span class="pick TOTAL ${totalPick.pick}">${totalLabel}</span>
            ${totalResultLabel ? `<span class="result-badge ${totalResult}">${totalResultLabel}</span>` : ''}
          </span>
        </div>
      </article>`;
  }).join('');

  for (const card of grid.querySelectorAll('.game-card')) {
    card.addEventListener('click', () => openDetail(card.dataset.pk));
  }
}

function teamId(abbr) {
  const teams = {
    ARI: 109, ATL: 144, BAL: 110, BOS: 111, CHC: 112, CHW: 145, CWS: 145,
    CIN: 113, CLE: 114, COL: 115, DET: 116, HOU: 117, KC: 118, LAA: 108,
    LAD: 119, MIA: 146, MIL: 158, MIN: 142, NYM: 121, NYY: 147,
    OAK: 133, ATH: 133, PHI: 143, PIT: 134, SD: 135, SF: 137,
    SEA: 136, STL: 138, TB: 139, TEX: 140, TOR: 141, WSH: 120,
  };
  return teams[abbr] || 0;
}

function showListView() {
  stopLiveRefresh();
  state.detailPk = null;
  document.getElementById('list-view').classList.remove('hidden');
  document.getElementById('detail-view').classList.add('hidden');
  document.getElementById('app-main').classList.remove('detail-page');
  document.getElementById('topbar').classList.remove('hidden');
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function stopLiveRefresh() {
  if (state.liveTimer) {
    clearInterval(state.liveTimer);
    state.liveTimer = null;
  }
}

function startLiveRefresh(pk) {
  stopLiveRefresh();
  state.liveTimer = setInterval(() => {
    if (!document.getElementById('detail-view').classList.contains('hidden')) {
      openDetail(pk, { silent: true });
    }
  }, 30000);
}

function showDetailView() {
  document.getElementById('list-view').classList.add('hidden');
  document.getElementById('detail-view').classList.remove('hidden');
  document.getElementById('app-main').classList.add('detail-page');
  document.getElementById('topbar').classList.add('hidden');
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

async function openDetail(pk, options = {}) {
  const body = document.getElementById('detail-body');
  state.detailPk = pk;
  if (!options.silent) {
    body.innerHTML = '<div class="empty">Cargando matchup...</div>';
    showDetailView();
  }
  const response = await fetch(`/api/games/${pk}?t=${Date.now()}`);
  const detail = await response.json();
  renderDetail(detail);
  if ((detail.live_score?.state || {}).is_live) {
    startLiveRefresh(pk);
  } else {
    stopLiveRefresh();
  }
}

function renderDetail(d) {
  const a = d.away_team;
  const h = d.home_team;
  const ap = d.away_pitcher;
  const hp = d.home_pitcher;
  const mdl = d.model;
  const mkt = d.market;
  const originalPick = d.pick_original_xgboost || {};
  const flip = d.pick_invertido || {};
  const flipSummary = d.flip_summary || flip.summary || 'Sin datos previos';
  const flipConflict = Boolean(d.conflict_flip || flip.conflict_flip);
  const pickSideOriginal = originalPick.side || (mdl.p_home == null ? 'NONE' : (mdl.p_home >= 0.5 ? 'HOME' : 'AWAY'));

  let edge = null;
  let edgeSign = '';
  if (mkt.p_home != null && mdl.p_home != null) {
    edge = mdl.p_home - mkt.p_home;
    edgeSign = edge >= 0 ? '+' : '';
  }

  const pickHome = pickSideOriginal === 'HOME';
  const pick = pickSideOriginal === 'NONE' ? 'NONE' : pickHome ? 'HOME' : 'AWAY';
  const pickAbbr = originalPick.team_abbrev || (pickHome ? h.abbrev : a.abbrev);
  const pickName = pickHome ? h.name : a.name;
  const mlResult = mlVerdict(pick, d.result.home_score, d.result.away_score);
  const totalPick = ouPick(mdl.pred_total_runs, mkt.total);
  const totalResult = ouVerdict(totalPick.pick, mkt.total, d.result.home_score, d.result.away_score);
  const awayProb = mdl.p_away ?? (mdl.p_home == null ? null : 1 - mdl.p_home);
  const modelLeader = pickHome ? h.abbrev : a.abbrev;
  const modelDog = pickHome ? a.abbrev : h.abbrev;
  const modelLeaderPct = pickHome ? mdl.p_home : awayProb;
  const modelDogPct = pickHome ? awayProb : mdl.p_home;
  const awayRecord = `${a.wins ?? '-'}-${a.losses ?? '-'}`;
  const homeRecord = `${h.wins ?? '-'}-${h.losses ?? '-'}`;
  const fairPick = pickHome ? mdl.fair_home_ml : mdl.fair_away_ml;
  const edgeText = edge == null ? '-' : `${edgeSign}${(edge * 100).toFixed(1)} pp`;
  const live = d.live_score || {};
  const liveState = live.state || {};
  const liveAway = live.away || {};
  const liveHome = live.home || {};
  const liveInnings = live.innings || [];
  const isLive = Boolean(liveState.is_live);
  const awayLiveRuns = liveAway.runs ?? d.result.away_score;
  const homeLiveRuns = liveHome.runs ?? d.result.home_score;
  const actualScore = (d.result.played || isLive)
    ? `${a.abbrev} ${awayLiveRuns ?? 0} - ${homeLiveRuns ?? 0} ${h.abbrev}`
    : 'Pendiente';

  const confidence = modelLeaderPct ?? 0.5;
  const confidenceLevel = confidence >= 0.58 ? 'Alta' : confidence >= 0.53 ? 'Media' : 'Baja';
  const resultText = d.result.played || isLive ? actualScore : 'Pendiente';
  const totalLine = mkt.total != null ? mkt.total.toFixed(1) : '-';
  const pickLine = totalPick.pick === 'NONE' ? 'Sin edge' : `${totalPick.pick} ${totalLine}`;
  const pickStatus = d.result.played
    ? (settledClass(mlResult) === 'win' ? 'PICK GANADO' : settledClass(mlResult) === 'loss' ? 'PICK PERDIDO' : 'PICK PENDIENTE')
    : 'PICK PENDIENTE';
  const dashStat = (side, metric, awayValue, homeValue) => `
    <div class="dash-table-row">
      <b class="${side === 'away' ? 'gold' : ''}">${awayValue}</b>
      <span>${metric}</span>
      <b class="blue">${homeValue}</b>
    </div>`;
  const keyItem = (icon, title, text) => `
    <div class="dash-key">
      <span>${icon}</span>
      <div><b>${title}</b><small>${text}</small></div>
    </div>`;
  const pickRow = (star, name, line, conf, risk, reason, active = false) => `
    <div class="dash-pick-row ${active ? 'active' : ''}">
      <span>${star}</span>
      <b>${name}</b>
      <em>${line}</em>
      <strong>${conf}</strong>
      <strong>${risk}</strong>
      <small>${reason}</small>
    </div>`;
  const inningHeaders = liveInnings.map(inn => `<b>${inn.num ?? '-'}</b>`).join('');
  const inningCells = (side) => liveInnings.map(inn => {
    const value = ((inn[side] || {}).runs);
    return `<span>${value == null ? '-' : value}</span>`;
  }).join('');
  const liveTable = (isLive || liveInnings.length) ? `
    <section class="dash-live-box ${isLive ? 'is-live' : ''}">
      <div class="live-meta">
        <span>${isLive ? 'EN VIVO' : liveState.detailed || 'LINE SCORE'}</span>
        <strong>${live.inning_state || ''} ${live.inning_ordinal || ''}</strong>
        <small>${live.outs ?? 0} outs · ${live.balls ?? 0}-${live.strikes ?? 0} cuenta</small>
      </div>
      <div class="line-score-table" style="--innings:${Math.max(liveInnings.length, 1)}">
        <div class="line-head"><span>Equipo</span>${inningHeaders}<b>R</b><b>H</b><b>E</b></div>
        <div class="line-row away">
          <strong>${a.abbrev}</strong>${inningCells('away')}
          <b>${liveAway.runs ?? '-'}</b><b>${liveAway.hits ?? '-'}</b><b>${liveAway.errors ?? '-'}</b>
        </div>
        <div class="line-row home">
          <strong>${h.abbrev}</strong>${inningCells('home')}
          <b>${liveHome.runs ?? '-'}</b><b>${liveHome.hits ?? '-'}</b><b>${liveHome.errors ?? '-'}</b>
        </div>
      </div>
    </section>` : '';

  document.getElementById('detail-body').innerHTML = `
    <article class="dash-report">
      <header class="dash-nav">
        <div class="dash-brand">
          <strong>STRIKECAST</strong>
          <span>Analisis Pro - MLB</span>
        </div>
        <button class="dash-back" type="button" onclick="showListView()">Eventos</button>
        <nav>
          <span class="active">Dashboard</span>
          <span>Analisis</span>
          <span>Modelos</span>
          <span>Picks</span>
          <span>Historial</span>
          <span>Mercados</span>
        </nav>
        <div class="dash-date">📅 ${fmtDateES(d.date)}</div>
        <div class="dash-league">MLB</div>
      </header>

      <section class="dash-scoreboard">
        <div class="dash-team away">
          <div class="team-accent"></div>
          <img src="${a.logo}" alt="${a.abbrev}" onerror="this.style.opacity=0.18">
          <div>
            <small>${(a.name || a.abbrev).split(' ')[0]}</small>
            <h1>${(a.name || a.abbrev).split(' ').slice(1).join(' ') || a.abbrev}</h1>
            <b>${awayRecord}</b>
            <span>${fmtPct(awayProb)} Win Prob.</span>
          </div>
        </div>
        <div class="dash-game-meta">
          <span>Juego de hoy</span>
          <b>VS</b>
          <small>${d.date}</small>
          <small>${d.venue || '-'}</small>
        </div>
        <div class="dash-team home">
          <div>
            <small>${(h.name || h.abbrev).split(' ')[0]}</small>
            <h1>${(h.name || h.abbrev).split(' ').slice(1).join(' ') || h.abbrev}</h1>
            <b>${homeRecord}</b>
            <span>${fmtPct(mdl.p_home)} Win Prob.</span>
          </div>
          <img src="${h.logo}" alt="${h.abbrev}" onerror="this.style.opacity=0.18">
        </div>
        <aside class="dash-recommend">
          <small>RECOMENDACION PRINCIPAL</small>
          <strong>${pickAbbr} -1.5</strong>
          <span>CONFIANZA ${confidenceLevel.toUpperCase()}</span>
          <div class="confidence-dots">${confidenceDots(confidenceLevel)}</div>
          <b>IMPLIED WIN PROB.</b>
          <em>${fmtPct(modelLeaderPct)}</em>
        </aside>
      </section>

      ${liveTable}

      <section class="dash-metric-grid">
        <div class="dash-metric"><span>◎</span><small>PICK PRINCIPAL</small><strong>${pickAbbr} -1.5</strong><em>Linea: ${fmtML(fairPick)}</em></div>
        <div class="dash-metric green"><span>↗</span><small>EDGE DETECTADO</small><strong>${edgeText}</strong><em>vs. Linea del mercado</em></div>
        <div class="dash-metric"><span>▥</span><small>TOTAL DE CARRERAS</small><strong>${fmtNum(mdl.pred_total_runs, 1)}</strong><em>Linea: ${totalLine}</em></div>
        <div class="dash-metric"><span>▦</span><small>MARCADOR PROYECTADO</small><strong>${probableScore(mdl.p_home, mdl.pred_total_runs, a.abbrev, h.abbrev)}</strong><em>Confianza: ${fmtPct(modelLeaderPct)}</em></div>
        <div class="dash-metric"><span>⟲</span><small>FLIP HISTORICO</small><strong>${flipSummary}</strong><em>${flipConflict ? 'Conflicto' : 'Sin conflicto'}</em></div>
      </section>

      <section class="dash-main-grid">
        <div class="dash-card dash-pitcher-card">
          <h3>MATCHUP DE ABRIDORES</h3>
          <div class="dash-pitchers">
            ${pitcherBlock(ap, a, 'away')}
            <div class="dash-vs">VS</div>
            ${pitcherBlock(hp, h, 'home')}
          </div>
        </div>

        <div class="dash-card dash-compare-card">
          <h3>COMPARATIVA DE EQUIPOS 2026</h3>
          <div class="dash-table-head"><b>${a.abbrev}</b><span>METRICA</span><b>${h.abbrev}</b></div>
          ${dashStat('away', 'Carreras por juego', fmtNum(a.runs_scored_l10, 1), fmtNum(h.runs_scored_l10, 1))}
          ${dashStat('away', 'wOBA ofensiva', fmt3(a.off_xwoba_l30), fmt3(h.off_xwoba_l30))}
          ${dashStat('away', 'wOBA recibida', fmt3(a.def_xwoba_l30), fmt3(h.def_xwoba_l30))}
          ${dashStat('away', 'K% ofensivo', a.off_k_pct_l30 == null ? '-' : (a.off_k_pct_l30 * 100).toFixed(1) + '%', h.off_k_pct_l30 == null ? '-' : (h.off_k_pct_l30 * 100).toFixed(1) + '%')}
          ${dashStat('away', 'K% recibido', a.def_k_pct_l30 == null ? '-' : (a.def_k_pct_l30 * 100).toFixed(1) + '%', h.def_k_pct_l30 == null ? '-' : (h.def_k_pct_l30 * 100).toFixed(1) + '%')}
          ${dashStat('away', 'WHIP', fmtNum(ap.starter_bb_pct_l15, 2), fmtNum(hp.starter_bb_pct_l15, 2))}
        </div>

        <div class="dash-card dash-model-card">
          <h3>MODELO PREDICTIVO (PROBABILIDAD DE VICTORIA)</h3>
          <div class="dash-donut-row">
            <div><b class="gold">${a.abbrev}</b><strong>${fmtPct(awayProb)}</strong></div>
            <div class="dash-donut" style="--p:${Math.max(0, Math.min(100, (mdl.p_home ?? 0.5) * 100))}"></div>
            <div><b class="blue">${h.abbrev}</b><strong>${fmtPct(mdl.p_home)}</strong></div>
          </div>
          <small>Basado en 10,000 simulaciones del modelo</small>
        </div>

        <div class="dash-card dash-keys-card">
          <h3>CLAVES DEL JUEGO</h3>
          ${keyItem('♕', 'Ventaja en abridores', `${pickAbbr} tiene mejor probabilidad ajustada.`)}
          ${keyItem('◉', 'Bateo oportuno', `${h.abbrev} wOBA: ${fmt3(h.off_xwoba_l30)} | ${a.abbrev}: ${fmt3(a.off_xwoba_l30)}`)}
          ${keyItem('◆', 'Bullpen solido', `Carreras permitidas L10: ${fmtNum(h.runs_allowed_l10, 1)}`)}
          ${keyItem('▣', 'Parque neutral', `${d.venue || 'Estadio'} con factor ${fmtNum(d.park_runs_factor, 2)}`)}
        </div>

        <div class="dash-card dash-picks-card">
          <h3>PICKS RECOMENDADOS</h3>
          <div class="dash-picks-head"><span></span><b>PICK</b><b>LINEA</b><b>CONFIANZA</b><b>RIESGO</b><b>RATIONALE</b></div>
          ${pickRow('⟲', 'Flip histórico', flipSummary, flipConflict ? 'Conflicto' : 'OK', flipConflict ? 'Alerta' : 'Bajo', 'Mismo día del año anterior')}
          ${pickRow('★', `${pickAbbr} -1.5`, fmtML(fairPick), confidenceLevel, edge != null && edge > 0 ? 'Bajo' : 'Medio', `Edge del modelo ${edgeText}`, true)}
          ${pickRow('•', `Total ${totalPick.pick === 'NONE' ? 'Menos' : totalPick.pick}`, totalLine, 'Media', 'Medio', 'Ambos abridores con buen control')}
          ${pickRow('•', `${pickAbbr} ML`, fmtML(fairPick), 'Media', 'Medio', 'Valor moderado vs probabilidad implicita')}
          <small>Las lineas pueden cambiar. Verifica antes de apostar.</small>
        </div>
      </section>

      <footer class="dash-final ${settledClass(mlResult) || 'tbd'}">
        <div><span>RESULTADO FINAL</span><strong>${scoreboardLineHTML(a, h, awayLiveRuns, homeLiveRuns)}</strong></div>
        <div><span>PICK:</span><strong>${pickAbbr} -1.5</strong></div>
        <div class="verdict-cell">
          <strong>${pickStatus}</strong>
        </div>
      </footer>
    </article>
  `;
}

// Pitcher card block matching the Picksteando layout:
//   hand badge (LHP/RHP) — photo — NAME — W-L | ERA | IP | K/9 — stat grid 4
function scoreboardLineHTML(a, h, awayRuns, homeRuns) {
  if (awayRuns == null || homeRuns == null) return 'Pendiente';
  return `<span class="sb-away">${a.abbrev} <b>${awayRuns}</b></span>
          <span class="sb-sep">-</span>
          <span class="sb-home"><b>${homeRuns}</b> ${h.abbrev}</span>`;
}

function pitcherBlock(p, team, side) {
  if (!p || !p.name) {
    return `<div class="dash-pitcher ${side}"><b>Probable TBD</b></div>`;
  }
  const hand = p.throws ? `${p.throws}HP` : '';
  const s = p.season || {};
  const wl  = (s.wins != null && s.losses != null) ? `${s.wins}-${s.losses}` : '—';
  const era = s.era != null ? Number(s.era).toFixed(2) : '—';
  const ip  = s.ip != null ? Number(s.ip).toFixed(1) : '—';
  const k9  = s.k9 != null ? Number(s.k9).toFixed(1)
            : (p.starter_k_pct_l15 != null ? (p.starter_k_pct_l15 * 9 / 0.27).toFixed(1) : '—');
  const bb9 = s.bb9 != null ? Number(s.bb9).toFixed(2)
            : (p.starter_bb_pct_l15 != null ? (p.starter_bb_pct_l15 * 9 / 0.27).toFixed(2) : '—');
  const whip = s.whip != null ? Number(s.whip).toFixed(2) : '—';
  const kPct = s.k_pct != null ? s.k_pct : p.starter_k_pct_l15;
  return `
    <div class="dash-pitcher ${side}">
      ${hand ? `<span class="hand-badge">${hand}</span>` : ''}
      <div class="pitcher-top">
        <img class="pitcher-photo" src="${p.headshot || ''}" alt="${p.name}" onerror="this.style.opacity=0.15">
        <div class="pitcher-info">
          <b>${p.name}</b>
          <span class="stat-line">${wl} · ${era} ERA</span>
          <span class="stat-line sub">${ip} IP · ${k9} K/9</span>
        </div>
      </div>
      <div class="dash-pitcher-stats">
        <span><b>${kPct == null ? '—' : (kPct*100).toFixed(1)+'%'}</b><small>K%</small></span>
        <span><b>${bb9}</b><small>BB/9</small></span>
        <span><b>${whip}</b><small>WHIP</small></span>
        <span><b>${fmt3(s.xwoba_vs_lhb)}</b><small>wOBA vs L</small></span>
        <span><b>${fmt3(s.xwoba_vs_rhb)}</b><small>wOBA vs R</small></span>
      </div>
    </div>`;
}

// Spanish-formatted date: '2026-06-21' -> '21 de junio, 2026'
const MONTHS_ES = ['enero','febrero','marzo','abril','mayo','junio','julio',
                   'agosto','septiembre','octubre','noviembre','diciembre'];
function fmtDateES(iso) {
  if (!iso) return '';
  const [y, m, d] = iso.split('-').map(Number);
  return `${d} de ${MONTHS_ES[m - 1]}, ${y}`;
}

// Render confidence dots: high=5/5, media=3/5, baja=2/5
function confidenceDots(level) {
  const filled = { Alta: 5, Media: 3, Baja: 2 }[level] || 0;
  let html = '';
  for (let i = 0; i < 5; i++) {
    html += `<i class="${i < filled ? '' : 'off'}"></i>`;
  }
  return html;
}

function probableScore(pHome, total, awayA, homeA) {
  if (pHome == null || total == null) return '-';
  const fav = pHome > 0.5 ? 'home' : 'away';
  const margin = Math.abs(pHome - 0.5) * 4;
  const favScore = Math.round((total + margin) / 2);
  const dogScore = Math.round((total - margin) / 2);
  return fav === 'home'
    ? `${awayA} ${dogScore}-${favScore} ${homeA}`
    : `${awayA} ${favScore}-${dogScore} ${homeA}`;
}

document.getElementById('back-to-events').addEventListener('click', showListView);
document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && !document.getElementById('detail-view').classList.contains('hidden')) {
    showListView();
  }
});

document.getElementById('datepick').addEventListener('change', e => {
  showListView();
  loadDate(e.target.value);
});
document.getElementById('prev').addEventListener('click', () => {
  showListView();
  loadDate(shiftDate(state.date, -1));
});
document.getElementById('next').addEventListener('click', () => {
  showListView();
  loadDate(shiftDate(state.date, 1));
});
document.getElementById('today').addEventListener('click', () => {
  showListView();
  loadDate(todayISO());
});

loadDate(todayISO());

(function () {
  'use strict';

  const TEAM_ID = {ARI:109,ATL:144,BAL:110,BOS:111,CHC:112,CHW:145,CWS:145,CIN:113,CLE:114,COL:115,DET:116,HOU:117,KC:118,LAA:108,LAD:119,MIA:146,MIL:158,MIN:142,NYM:121,NYY:147,OAK:133,ATH:133,PHI:143,PIT:134,SD:135,SF:137,SEA:136,STL:138,TB:139,TEX:140,TOR:141,WSH:120};
  const NICKNAMES = {ARI:'DIAMONDBACKS',ATL:'BRAVES',BAL:'ORIOLES',BOS:'RED SOX',CHC:'CUBS',CHW:'WHITE SOX',CWS:'WHITE SOX',CIN:'REDS',CLE:'GUARDIANS',COL:'ROCKIES',DET:'TIGERS',HOU:'ASTROS',KC:'ROYALS',LAA:'ANGELS',LAD:'DODGERS',MIA:'MARLINS',MIL:'BREWERS',MIN:'TWINS',NYM:'METS',NYY:'YANKEES',OAK:'ATHLETICS',ATH:'ATHLETICS',PHI:'PHILLIES',PIT:'PIRATES',SD:'PADRES',SF:'GIANTS',SEA:'MARINERS',STL:'CARDINALS',TB:'RAYS',TEX:'RANGERS',TOR:'BLUE JAYS',WSH:'NATIONALS'};
  const DISPLAY_FONT = '"Anton", Impact, "Arial Black", sans-serif';
  const DATA_FONT = '"Inter", Arial, sans-serif';
  const BLUE = '#30a8ff';
  const RED = '#ff443d';

  function loadImage(src) {
    return new Promise((resolve) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => resolve(img);
      img.onerror = () => resolve(null);
      img.src = src;
    });
  }

  function setFont(ctx, size, options) {
    const o = options || {};
    ctx.font = `${o.italic ? 'italic ' : ''}${o.weight || 900} ${size}px ${o.family || DATA_FONT}`;
  }

  function fittedSize(ctx, text, maxWidth, start, min, options) {
    let size = start;
    while (size > min) {
      setFont(ctx, size, options);
      if (ctx.measureText(String(text)).width <= maxWidth) break;
      size -= 2;
    }
    return size;
  }

  function drawText(ctx, text, x, y, size, color, options) {
    const o = Object.assign({
      align: 'center',
      baseline: 'middle',
      family: DATA_FONT,
      weight: 900,
      italic: false,
      maxWidth: null,
      minSize: 16,
      stroke: 'rgba(0,0,0,.92)',
      strokeWidth: 0,
      letterSpacing: 0,
    }, options || {});
    ctx.save();
    ctx.textAlign = o.align;
    ctx.textBaseline = o.baseline;
    ctx.lineJoin = 'round';
    ctx.miterLimit = 2;
    const actual = o.maxWidth
      ? fittedSize(ctx, text, o.maxWidth, size, o.minSize, o)
      : size;
    setFont(ctx, actual, o);
    ctx.fontKerning = 'normal';
    ctx.fillStyle = color;
    if (o.strokeWidth) {
      ctx.strokeStyle = o.stroke;
      ctx.lineWidth = o.strokeWidth;
      ctx.strokeText(String(text), x, y);
    }
    ctx.fillText(String(text), x, y);
    ctx.restore();
    return actual;
  }

  function drawMetalTitle(ctx, text, x, y, maxWidth, side) {
    const opts = {family: DISPLAY_FONT, weight: 400};
    const size = fittedSize(ctx, text, maxWidth, 154, 78, opts);
    ctx.save();
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.lineJoin = 'round';
    ctx.miterLimit = 2;
    setFont(ctx, size, opts);

    const edge = side === 'away' ? '#092c54' : '#48110f';
    for (let offset = 13; offset >= 3; offset -= 2) {
      ctx.strokeStyle = '#01050b';
      ctx.lineWidth = 8;
      ctx.strokeText(text, x + offset * .35, y + offset);
      ctx.fillStyle = edge;
      ctx.fillText(text, x + offset * .35, y + offset);
    }

    const metal = ctx.createLinearGradient(0, y - size * .62, 0, y + size * .62);
    metal.addColorStop(0, '#ffffff');
    metal.addColorStop(.18, '#e7ebef');
    metal.addColorStop(.48, '#ffffff');
    metal.addColorStop(.52, '#a9b1bc');
    metal.addColorStop(.82, '#d8dde3');
    metal.addColorStop(1, '#747d89');
    ctx.strokeStyle = '#07111e';
    ctx.lineWidth = 9;
    ctx.strokeText(text, x, y);
    ctx.strokeStyle = side === 'away' ? '#7fcaff' : '#ff928e';
    ctx.lineWidth = 2;
    ctx.strokeText(text, x, y);
    ctx.fillStyle = metal;
    ctx.fillText(text, x, y);

    ctx.globalAlpha = .55;
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1;
    ctx.strokeText(text, x, y - 2);
    ctx.restore();
  }

  function polygon(ctx, points) {
    ctx.beginPath();
    points.forEach((point, index) => {
      if (index === 0) ctx.moveTo(point[0], point[1]);
      else ctx.lineTo(point[0], point[1]);
    });
    ctx.closePath();
  }

  function drawTopBar(ctx) {
    const points = [[86,54],[830,54],[856,80],[856,139],[830,165],[111,165],[85,139],[85,80]];
    const border = ctx.createLinearGradient(85, 0, 856, 0);
    border.addColorStop(0, '#36b6ff');
    border.addColorStop(.48, '#a7b8cc');
    border.addColorStop(.52, '#b9aeb5');
    border.addColorStop(1, '#ff5752');
    ctx.save();
    polygon(ctx, points);
    ctx.fillStyle = 'rgba(3,12,25,.88)';
    ctx.fill();
    ctx.shadowColor = 'rgba(48,168,255,.32)';
    ctx.shadowBlur = 18;
    ctx.strokeStyle = border;
    ctx.lineWidth = 3;
    ctx.stroke();
    ctx.shadowBlur = 0;

    ctx.strokeStyle = 'rgba(225,234,244,.6)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(455, 77);
    ctx.lineTo(455, 142);
    ctx.stroke();

    ctx.strokeStyle = '#dce5ef';
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.arc(164, 109, 24, 0, Math.PI * 2);
    ctx.stroke();
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(164, 109);
    ctx.lineTo(164, 93);
    ctx.moveTo(164, 109);
    ctx.lineTo(176, 116);
    ctx.stroke();

    ctx.strokeStyle = '#dce5ef';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.ellipse(519, 104, 27, 8, 0, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(493, 104);
    ctx.lineTo(498, 125);
    ctx.quadraticCurveTo(519, 137, 540, 125);
    ctx.lineTo(545, 104);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(500, 88);
    ctx.lineTo(500, 99);
    ctx.moveTo(538, 88);
    ctx.lineTo(538, 99);
    ctx.stroke();
    ctx.restore();
  }

  function drawTeamPanel(ctx, side) {
    const left = side === 'away';
    const points = left
      ? [[39,579],[335,579],[417,650],[417,889],[381,928],[39,928]]
      : [[524,650],[606,579],[902,579],[902,928],[560,928],[524,889]];
    const panel = ctx.createLinearGradient(left ? 40 : 902, 579, left ? 417 : 524, 928);
    if (left) {
      panel.addColorStop(0, 'rgba(9,49,98,.84)');
      panel.addColorStop(1, 'rgba(2,10,22,.93)');
    } else {
      panel.addColorStop(0, 'rgba(79,14,17,.84)');
      panel.addColorStop(1, 'rgba(2,10,22,.93)');
    }
    ctx.save();
    polygon(ctx, points);
    ctx.fillStyle = panel;
    ctx.fill();
    ctx.strokeStyle = left ? '#33adff' : '#ff514b';
    ctx.lineWidth = 3;
    ctx.shadowColor = left ? 'rgba(48,168,255,.7)' : 'rgba(255,68,61,.7)';
    ctx.shadowBlur = 16;
    ctx.stroke();
    ctx.shadowBlur = 0;
    ctx.globalAlpha = .45;
    ctx.strokeStyle = '#eef7ff';
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.restore();
  }

  function contain(ctx, img, cx, cy, maxW, maxH) {
    if (!img) return false;
    const scale = Math.min(maxW / img.width, maxH / img.height);
    const w = img.width * scale;
    const h = img.height * scale;
    ctx.save();
    ctx.shadowColor = 'rgba(0,0,0,.78)';
    ctx.shadowBlur = 12;
    ctx.drawImage(img, cx - w / 2, cy - h / 2, w, h);
    ctx.restore();
    return true;
  }

  function drawProbabilityRing(ctx, cx, cy, awayProbability) {
    const radius = 114;
    ctx.save();
    ctx.lineWidth = 28;
    ctx.lineCap = 'butt';
    ctx.shadowBlur = 12;
    ctx.shadowColor = 'rgba(0,0,0,.8)';
    ctx.strokeStyle = RED;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, -.5 * Math.PI, 1.5 * Math.PI);
    ctx.stroke();
    ctx.strokeStyle = BLUE;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, -.5 * Math.PI, (-.5 - 2 * awayProbability) * Math.PI, true);
    ctx.stroke();
    ctx.shadowBlur = 0;
    ctx.lineWidth = 2;
    ctx.strokeStyle = 'rgba(255,255,255,.72)';
    ctx.beginPath();
    ctx.arc(cx, cy, radius - 14, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }

  function pitcherLabel(pitcher) {
    if (!pitcher || !pitcher.name) return 'POR DEFINIR';
    const parts = String(pitcher.name).trim().split(/\s+/);
    const lastName = parts[parts.length - 1].toUpperCase();
    const rawHand = pitcher.hand || pitcher.throws || pitcher.p_throws || '';
    const hand = String(rawHand).trim().toUpperCase().slice(0, 1);
    return hand === 'R' || hand === 'L' ? `${lastName} (${hand})` : lastName;
  }

  function safeProbability(value, fallback) {
    const number = Number(value);
    if (!Number.isFinite(number)) return fallback;
    return Math.max(0, Math.min(1, number));
  }

  function percentage(value) {
    return `${(safeProbability(value, .5) * 100).toFixed(1)}%`;
  }

  async function exportDailyPickPoster(p, gameDate) {
    const W = 941;
    const H = 1672;
    const canvas = document.createElement('canvas');
    canvas.width = W;
    canvas.height = H;
    const ctx = canvas.getContext('2d');

    if (document.fonts) {
      await Promise.allSettled([
        document.fonts.load('400 150px "Anton"'),
        document.fonts.load('900 40px "Inter"'),
        document.fonts.ready,
      ]);
    }

    const logoUrl = (abbr) => `https://www.mlbstatic.com/team-logos/${TEAM_ID[abbr] || 0}.svg`;
    const [background, awayLogo, homeLogo] = await Promise.all([
      loadImage('/static/daily-pick-poster-template-v2.png?v=20260724'),
      loadImage(logoUrl(p.away_abbrev)),
      loadImage(logoUrl(p.home_abbrev)),
    ]);

    if (background) {
      ctx.drawImage(background, 0, 0, W, H);
    } else {
      const fallback = ctx.createLinearGradient(0, 0, 0, H);
      fallback.addColorStop(0, '#03101f');
      fallback.addColorStop(1, '#010409');
      ctx.fillStyle = fallback;
      ctx.fillRect(0, 0, W, H);
    }

    const readability = ctx.createLinearGradient(0, 0, 0, 1310);
    readability.addColorStop(0, 'rgba(1,5,12,.08)');
    readability.addColorStop(.48, 'rgba(1,5,12,.18)');
    readability.addColorStop(1, 'rgba(1,5,12,.35)');
    ctx.fillStyle = readability;
    ctx.fillRect(0, 0, W, 1310);

    const awayProbability = safeProbability(p.p_away, .5);
    const homeProbability = safeProbability(p.p_home, 1 - awayProbability);
    const time = p.first_pitch_utc
      ? new Date(p.first_pitch_utc).toLocaleTimeString('en-US', {
          timeZone: 'America/New_York',
          hour: 'numeric',
          minute: '2-digit',
          hour12: true,
        }).toUpperCase()
      : 'HORA PENDIENTE';
    const venue = String(p.venue || 'SEDE POR DEFINIR').toUpperCase();
    const awayNickname = NICKNAMES[p.away_abbrev] || p.away_abbrev;
    const homeNickname = NICKNAMES[p.home_abbrev] || p.home_abbrev;

    drawTopBar(ctx);
    drawText(ctx, time, 309, 110, 41, '#f0f1ef', {
      family: DISPLAY_FONT,
      weight: 400,
      maxWidth: 235,
      minSize: 28,
      strokeWidth: 2,
    });
    drawText(ctx, venue, 695, 110, 31, '#e5e7e8', {
      family: DISPLAY_FONT,
      weight: 400,
      maxWidth: 280,
      minSize: 18,
      strokeWidth: 2,
    });

    drawMetalTitle(ctx, awayNickname, W / 2, 256, 735, 'away');
    drawText(ctx, 'VS', W / 2, 365, 62, '#f4f6f7', {
      family: DISPLAY_FONT,
      weight: 400,
      stroke: '#07101d',
      strokeWidth: 8,
    });
    drawMetalTitle(ctx, homeNickname, W / 2, 472, 735, 'home');

    drawTeamPanel(ctx, 'away');
    drawTeamPanel(ctx, 'home');
    if (!contain(ctx, awayLogo, 229, 744, 285, 255)) {
      drawText(ctx, p.away_abbrev, 229, 744, 104, '#f2f4f6', {
        family: DISPLAY_FONT,
        weight: 400,
        strokeWidth: 5,
      });
    }
    if (!contain(ctx, homeLogo, 712, 744, 285, 255)) {
      drawText(ctx, p.home_abbrev, 712, 744, 104, '#f2f4f6', {
        family: DISPLAY_FONT,
        weight: 400,
        strokeWidth: 5,
      });
    }

    const awayBand = ctx.createLinearGradient(54, 0, 404, 0);
    awayBand.addColorStop(0, 'rgba(5,55,111,.92)');
    awayBand.addColorStop(1, 'rgba(5,22,44,.72)');
    ctx.fillStyle = awayBand;
    ctx.fillRect(54, 852, 350, 76);
    const homeBand = ctx.createLinearGradient(537, 0, 887, 0);
    homeBand.addColorStop(0, 'rgba(62,15,20,.72)');
    homeBand.addColorStop(1, 'rgba(135,27,30,.92)');
    ctx.fillStyle = homeBand;
    ctx.fillRect(537, 852, 350, 76);
    drawText(ctx, p.away_abbrev, 229, 890, 58, '#f1f3f4', {
      family: DISPLAY_FONT,
      weight: 400,
      strokeWidth: 3,
    });
    drawText(ctx, p.home_abbrev, 712, 890, 58, '#f1f3f4', {
      family: DISPLAY_FONT,
      weight: 400,
      strokeWidth: 3,
    });

    drawProbabilityRing(ctx, W / 2, 1044, awayProbability);
    drawText(ctx, p.away_abbrev, 151, 1000, 42, '#f0f2f4', {
      family: DISPLAY_FONT,
      weight: 400,
      strokeWidth: 2,
    });
    drawText(ctx, percentage(awayProbability), 151, 1072, 57, '#f7f7f4', {
      family: DISPLAY_FONT,
      weight: 400,
      maxWidth: 255,
      minSize: 40,
      strokeWidth: 3,
    });
    drawText(ctx, 'PROBABILIDAD', 151, 1121, 19, '#c8cdd4', {
      maxWidth: 245,
      minSize: 14,
      strokeWidth: 1,
    });
    drawText(ctx, p.home_abbrev, 790, 1000, 42, '#f0f2f4', {
      family: DISPLAY_FONT,
      weight: 400,
      strokeWidth: 2,
    });
    drawText(ctx, percentage(homeProbability), 790, 1072, 57, '#f7f7f4', {
      family: DISPLAY_FONT,
      weight: 400,
      maxWidth: 255,
      minSize: 40,
      strokeWidth: 3,
    });
    drawText(ctx, 'PROBABILIDAD', 790, 1121, 19, '#c8cdd4', {
      maxWidth: 245,
      minSize: 14,
      strokeWidth: 1,
    });

    drawText(ctx, 'PROB.', W / 2, 1007, 21, '#ced4dc', {
      maxWidth: 150,
      minSize: 16,
      strokeWidth: 1,
    });
    drawText(ctx, 'MODELO', W / 2, 1036, 21, '#ced4dc', {
      maxWidth: 150,
      minSize: 16,
      strokeWidth: 1,
    });
    drawText(ctx, `${Math.round(awayProbability * 100)}%`, 428, 1081, 31, BLUE, {
      family: DISPLAY_FONT,
      weight: 400,
      align: 'right',
      strokeWidth: 2,
    });
    drawText(ctx, '|', W / 2, 1081, 29, '#aeb5bf', {strokeWidth: 0});
    drawText(ctx, `${Math.round(homeProbability * 100)}%`, 513, 1081, 31, RED, {
      family: DISPLAY_FONT,
      weight: 400,
      align: 'left',
      strokeWidth: 2,
    });

    ctx.save();
    const pitcherLine = ctx.createLinearGradient(100, 0, 841, 0);
    pitcherLine.addColorStop(0, 'rgba(48,168,255,0)');
    pitcherLine.addColorStop(.35, 'rgba(48,168,255,.85)');
    pitcherLine.addColorStop(.5, 'rgba(235,239,244,.85)');
    pitcherLine.addColorStop(.65, 'rgba(255,68,61,.85)');
    pitcherLine.addColorStop(1, 'rgba(255,68,61,0)');
    ctx.strokeStyle = pitcherLine;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(76, 1235);
    ctx.lineTo(865, 1235);
    ctx.stroke();
    ctx.restore();
    drawText(ctx, pitcherLabel(p.pitcher_away), 267, 1238, 34, '#d9dde2', {
      family: DISPLAY_FONT,
      weight: 400,
      maxWidth: 330,
      minSize: 22,
      strokeWidth: 2,
    });
    drawText(ctx, 'VS', W / 2, 1238, 39, '#f1f2f3', {
      family: DISPLAY_FONT,
      weight: 400,
      strokeWidth: 4,
    });
    drawText(ctx, pitcherLabel(p.pitcher_home), 674, 1238, 34, '#d9dde2', {
      family: DISPLAY_FONT,
      weight: 400,
      maxWidth: 330,
      minSize: 22,
      strokeWidth: 2,
    });

    ctx.save();
    const footerLine = ctx.createLinearGradient(20, 0, 921, 0);
    footerLine.addColorStop(0, BLUE);
    footerLine.addColorStop(.48, 'rgba(75,120,165,.35)');
    footerLine.addColorStop(.52, 'rgba(160,75,75,.35)');
    footerLine.addColorStop(1, RED);
    ctx.strokeStyle = footerLine;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(21, 1597);
    ctx.lineTo(920, 1597);
    ctx.stroke();
    ctx.restore();
    drawText(ctx, '—  ·  ·  ·  ★   PROYECCIÓN BASADA EN DATOS   ★  ·  ·  ·  —', W / 2, 1630, 19, '#aeb6c2', {
      family: DATA_FONT,
      weight: 800,
      maxWidth: 850,
      minSize: 14,
      strokeWidth: 1,
    });

    await new Promise((resolve, reject) => canvas.toBlob((blob) => {
      if (!blob) {
        reject(new Error('No se pudo crear la imagen'));
        return;
      }
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `strikecast-${gameDate || 'analisis'}-analisis-${p.away_abbrev}-${p.home_abbrev}.png`;
      window.__lastDailyPickPoster = {blob, filename: link.download};
      document.body.appendChild(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1500);
      resolve();
    }, 'image/png'));
  }

  window.exportDailyPickPoster = exportDailyPickPoster;
})();

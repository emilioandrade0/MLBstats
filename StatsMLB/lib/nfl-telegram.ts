export type NflTelegramPick = { gameId: string; home: string; away: string; kickoff: string; market: 'ML' | 'SPREAD' | 'OVER' | 'UNDER' | 'LVD'; side: string; line?: number; decimalOdds: number; pickNumber: number };
export function nflTelegramSettings(bindings: Record<string, string | undefined>, local: Record<string, string | undefined>) {
  return { token: bindings.NFL_TELEGRAM_BOT_TOKEN || local.NFL_TELEGRAM_BOT_TOKEN || bindings.TELEGRAM_BOT_TOKEN || local.TELEGRAM_BOT_TOKEN,
    chat: bindings.NFL_TELEGRAM_CHAT_ID || local.NFL_TELEGRAM_CHAT_ID };
}
const teams = new Set('ARI ATL BAL BUF CAR CHI CIN CLE DAL DEN DET GB HOU IND JAX JAC KC LV OAK LAC SD LA LAR STL MIA MIN NE NO NYG NYJ PHI PIT SF SEA TB TEN WAS WSH'.split(' '));
const TEAM_FULL: Record<string, string> = {
  ARI:'Arizona Cardinals', ATL:'Atlanta Falcons', BAL:'Baltimore Ravens', BUF:'Buffalo Bills',
  CAR:'Carolina Panthers', CHI:'Chicago Bears', CIN:'Cincinnati Bengals', CLE:'Cleveland Browns',
  DAL:'Dallas Cowboys', DEN:'Denver Broncos', DET:'Detroit Lions', GB:'Green Bay Packers',
  HOU:'Houston Texans', IND:'Indianapolis Colts', JAX:'Jacksonville Jaguars', JAC:'Jacksonville Jaguars',
  KC:'Kansas City Chiefs', LV:'Las Vegas Raiders', OAK:'Las Vegas Raiders',
  LAC:'Los Angeles Chargers', SD:'Los Angeles Chargers',
  LA:'Los Angeles Rams', LAR:'Los Angeles Rams', STL:'Los Angeles Rams',
  MIA:'Miami Dolphins', MIN:'Minnesota Vikings', NE:'New England Patriots',
  NO:'New Orleans Saints', NYG:'New York Giants', NYJ:'New York Jets',
  PHI:'Philadelphia Eagles', PIT:'Pittsburgh Steelers', SF:'San Francisco 49ers',
  SEA:'Seattle Seahawks', TB:'Tampa Bay Buccaneers', TEN:'Tennessee Titans',
  WAS:'Washington Commanders', WSH:'Washington Commanders',
};
const fullName = (abbr: string) => TEAM_FULL[abbr] || abbr;
export function nflMessage(value: unknown): string {
  if (!value || typeof value !== 'object') throw Error('Pick inválido.');
  const p = value as NflTelegramPick;
  if (!teams.has(p.home) || !teams.has(p.away) || p.home === p.away) throw Error('Equipos NFL inválidos.');
  if (typeof p.gameId !== 'string' || !/^[A-Za-z0-9_-]{3,60}$/.test(p.gameId)) throw Error('Partido inválido.');
  if (typeof p.kickoff !== 'string' || !Number.isFinite(Date.parse(p.kickoff))) throw Error('Falta fecha válida del partido.');
  if (!Number.isInteger(p.pickNumber) || p.pickNumber < 1 || p.pickNumber > 999) throw Error('Número de pick inválido.');
  if (!Number.isFinite(p.decimalOdds) || p.decimalOdds < 1.01 || p.decimalOdds > 100) throw Error('Introduce un momio decimal entre 1.01 y 100.');
  if (!['ML','SPREAD','OVER','UNDER','LVD'].includes(p.market)) throw Error('Mercado inválido.');
  let pick = '';
  if (p.market === 'ML' || p.market === 'SPREAD') {
    if (p.side !== p.home && p.side !== p.away) throw Error('El equipo no pertenece al partido.');
    pick = `${fullName(p.side)} ML`;
    if (p.market === 'SPREAD') {
      if (typeof p.line !== 'number' || !Number.isFinite(p.line) || Math.abs(p.line) > 100) throw Error('Hándicap inválido.');
      pick = `${fullName(p.side)} ${p.line > 0 ? '+' : ''}${p.line} (spread)`;
    }
  } else if (p.market === 'LVD') {
    if (!['L','V','D'].includes(p.side)) throw Error('Selección L/V/D inválida.');
    pick = p.side === 'L' ? `${fullName(p.home)} gana por más de 6` : p.side === 'V' ? `${fullName(p.away)} gana por más de 6` : 'Diferencia de 6 puntos o menos';
  } else {
    if (typeof p.line !== 'number' || !Number.isFinite(p.line) || p.line <= 0 || p.line > 120) throw Error('Total NFL inválido.');
    pick = `${p.market} ${p.line}`;
  }
  const when = new Intl.DateTimeFormat('es-MX', { timeZone: 'America/Mexico_City', dateStyle: 'medium', timeStyle: 'short' }).format(new Date(p.kickoff));
  return `🏈 PICK #${p.pickNumber} | NFL\n\n${fullName(p.away)} vs ${fullName(p.home)}\nFecha: ${when} (CDMX)\n\n🎯 ${pick}\n💰 Momio: ${p.decimalOdds.toFixed(2)}`;
}

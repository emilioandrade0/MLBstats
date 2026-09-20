export type TeamInfo = { name: string; city: string; color: string };

// ESPN CDN logo URL — public, no auth
export function teamLogo(abbrev: string): string {
  const lower = (abbrev || "").toLowerCase();
  // Some historical codes need mapping to ESPN's current naming
  const map: Record<string, string> = {
    lar: "lar", la: "lar", oak: "lv", sd: "lac", stl: "lar",
    was: "wsh", jac: "jax",
  };
  return `https://a.espncdn.com/i/teamlogos/nfl/500/${map[lower] || lower}.png`;
}

export const TEAMS: Record<string, TeamInfo> = {
  ARI: { name: "Cardinals",   city: "Arizona",       color: "#97233F" },
  ATL: { name: "Falcons",     city: "Atlanta",       color: "#A71930" },
  BAL: { name: "Ravens",      city: "Baltimore",     color: "#241773" },
  BUF: { name: "Bills",       city: "Buffalo",       color: "#00338D" },
  CAR: { name: "Panthers",    city: "Carolina",      color: "#0085CA" },
  CHI: { name: "Bears",       city: "Chicago",       color: "#0B162A" },
  CIN: { name: "Bengals",     city: "Cincinnati",    color: "#FB4F14" },
  CLE: { name: "Browns",      city: "Cleveland",     color: "#311D00" },
  DAL: { name: "Cowboys",     city: "Dallas",        color: "#003594" },
  DEN: { name: "Broncos",     city: "Denver",        color: "#FB4F14" },
  DET: { name: "Lions",       city: "Detroit",       color: "#0076B6" },
  GB:  { name: "Packers",     city: "Green Bay",     color: "#203731" },
  HOU: { name: "Texans",      city: "Houston",       color: "#03202F" },
  IND: { name: "Colts",       city: "Indianapolis",  color: "#002C5F" },
  JAX: { name: "Jaguars",     city: "Jacksonville",  color: "#101820" },
  KC:  { name: "Chiefs",      city: "Kansas City",   color: "#E31837" },
  LV:  { name: "Raiders",     city: "Las Vegas",     color: "#000000" },
  LAC: { name: "Chargers",    city: "Los Angeles",   color: "#0080C6" },
  LA:  { name: "Rams",        city: "Los Angeles",   color: "#003594" },
  LAR: { name: "Rams",        city: "Los Angeles",   color: "#003594" },
  MIA: { name: "Dolphins",    city: "Miami",         color: "#008E97" },
  MIN: { name: "Vikings",     city: "Minnesota",     color: "#4F2683" },
  NE:  { name: "Patriots",    city: "New England",   color: "#002244" },
  NO:  { name: "Saints",      city: "New Orleans",   color: "#D3BC8D" },
  NYG: { name: "Giants",      city: "New York",      color: "#0B2265" },
  NYJ: { name: "Jets",        city: "New York",      color: "#125740" },
  PHI: { name: "Eagles",      city: "Philadelphia",  color: "#004C54" },
  PIT: { name: "Steelers",    city: "Pittsburgh",    color: "#FFB612" },
  SF:  { name: "49ers",       city: "San Francisco", color: "#AA0000" },
  SEA: { name: "Seahawks",    city: "Seattle",       color: "#002244" },
  TB:  { name: "Buccaneers",  city: "Tampa Bay",     color: "#D50A0A" },
  TEN: { name: "Titans",      city: "Tennessee",     color: "#0C2340" },
  WAS: { name: "Commanders",  city: "Washington",    color: "#5A1414" },
  WSH: { name: "Commanders",  city: "Washington",    color: "#5A1414" },
};

export function teamInfo(abbrev: string): TeamInfo {
  return TEAMS[abbrev?.toUpperCase()] || { name: abbrev, city: "", color: "#6b7280" };
}

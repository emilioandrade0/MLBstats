/* eslint-disable @next/next/no-img-element */

const TEAM_IDS: Record<string, number> = {
  ARI: 109,
  AZ: 109,
  ATL: 144,
  BAL: 110,
  BOS: 111,
  CHC: 112,
  CHW: 145,
  CWS: 145,
  CIN: 113,
  CLE: 114,
  COL: 115,
  DET: 116,
  HOU: 117,
  KC: 118,
  LAA: 108,
  LAD: 119,
  MIA: 146,
  MIL: 158,
  MIN: 142,
  NYM: 121,
  NYY: 147,
  OAK: 133,
  ATH: 133,
  PHI: 143,
  PIT: 134,
  SD: 135,
  SF: 137,
  SEA: 136,
  STL: 138,
  TB: 139,
  TEX: 140,
  TOR: 141,
  WSH: 120,
};

export function teamLogoUrl(team: string) {
  const id = TEAM_IDS[team.trim().toUpperCase()];
  return id ? `https://www.mlbstatic.com/team-logos/${id}.svg` : '';
}

export default function TeamLogo({ team, name, className = '' }: { team: string; name?: string; className?: string }) {
  const code = team.trim().toUpperCase();
  const src = teamLogoUrl(code);

  return <span className={`team-logo ${className}`} aria-hidden="true">
    <b>{code.slice(0, 3)}</b>
    {src && <img src={src} alt={`Logo de ${name ?? code}`} loading="lazy" decoding="async" onError={event => { event.currentTarget.style.display = 'none'; }} />}
  </span>;
}

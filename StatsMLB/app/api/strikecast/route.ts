import { NextRequest, NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

const STRIKECAST_BASE = process.env.STRIKECAST_BASE_URL ?? 'http://127.0.0.1:8000';

function validDate(value: string) {
  return /^\d{4}-\d{2}-\d{2}$/.test(value) && !Number.isNaN(Date.parse(`${value}T12:00:00Z`));
}

type StrikecastGame = {
  game_pk: number;
  date: string;
  first_pitch_utc: string;
  away_abbrev: string;
  home_abbrev: string;
  p_home: number | null;
  p_away: number | null;
  pick_abbrev: string | null;
  confidence_tier?: string | null;
  confidence_pp?: number | null;
  f5_p_home?: number | null;
  series_flip?: boolean | null;
  team_flip?: boolean | null;
  is_played?: boolean | null;
  home_score?: number | null;
  away_score?: number | null;
};

export async function GET(request: NextRequest) {
  const date = request.nextUrl.searchParams.get('date');
  if (!date || !validDate(date)) {
    return NextResponse.json({ error: 'Fecha inválida' }, { status: 400 });
  }
  const url = `${STRIKECAST_BASE.replace(/\/+$/, '')}/api/games?start=${date}&end=${date}`;
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 10_000);
    const response = await fetch(url, { cache: 'no-store', signal: controller.signal });
    clearTimeout(timeout);
    if (!response.ok) {
      return NextResponse.json({ date, source: STRIKECAST_BASE, games: [], error: `Strikecast respondió ${response.status}` }, { status: 502, headers: { 'Cache-Control': 'no-store' } });
    }
    const raw = await response.json() as StrikecastGame[];
    const games = raw.map(item => ({
      gamePk: item.game_pk,
      date: item.date,
      firstPitchUtc: item.first_pitch_utc,
      away: item.away_abbrev,
      home: item.home_abbrev,
      pHome: item.p_home,
      pAway: item.p_away,
      pick: item.pick_abbrev,
      confidenceTier: item.confidence_tier ?? null,
      confidencePp: item.confidence_pp ?? null,
      f5PHome: item.f5_p_home ?? null,
      seriesFlip: item.series_flip ?? null,
      teamFlip: item.team_flip ?? null,
      isPlayed: item.is_played ?? null,
      homeScore: item.home_score ?? null,
      awayScore: item.away_score ?? null,
    }));
    return NextResponse.json({ date, source: STRIKECAST_BASE, games }, { headers: { 'Cache-Control': 'no-store' } });
  } catch (error) {
    return NextResponse.json({ date, source: STRIKECAST_BASE, games: [], error: error instanceof Error ? error.message : 'Error desconocido' }, { status: 502, headers: { 'Cache-Control': 'no-store' } });
  }
}

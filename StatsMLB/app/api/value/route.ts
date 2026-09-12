import { NextRequest, NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

const STRIKECAST_BASE = process.env.STRIKECAST_BASE_URL ?? 'http://127.0.0.1:8000';

function validDate(value: string) {
  return /^\d{4}-\d{2}-\d{2}$/.test(value) && !Number.isNaN(Date.parse(`${value}T12:00:00Z`));
}

type ValuePickRow = {
  game_pk: number;
  date: string;
  home_abbrev: string;
  away_abbrev: string;
  home_ml: number | null;
  away_ml: number | null;
  p_home_win: number | null;
  p_home_first: number | null;
  imp_home: number | null;
  edge_win: number | null;
  edge_first: number | null;
  pick_code: string | null;
  pick_prob: number | null;
  tier: 'VALOR-FUERTE' | 'VALOR' | 'NEUTRO';
};

export async function GET(request: NextRequest) {
  const date = request.nextUrl.searchParams.get('date');
  if (!date || !validDate(date)) {
    return NextResponse.json({ error: 'Fecha inválida' }, { status: 400 });
  }
  const url = `${STRIKECAST_BASE.replace(/\/+$/, '')}/api/value_picks?start=${date}&end=${date}`;
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 10_000);
    const response = await fetch(url, { cache: 'no-store', signal: controller.signal });
    clearTimeout(timeout);
    if (!response.ok) {
      return NextResponse.json({ date, source: STRIKECAST_BASE, picks: [], error: `Value API respondió ${response.status}` }, { status: 502, headers: { 'Cache-Control': 'no-store' } });
    }
    const payload = await response.json() as { updated_at?: string; picks?: ValuePickRow[] };
    const updatedAt = payload.updated_at ?? null;
    const picks = (payload.picks ?? []).map(item => ({
      gamePk: item.game_pk,
      date: item.date,
      home: item.home_abbrev,
      away: item.away_abbrev,
      homeMl: item.home_ml,
      awayMl: item.away_ml,
      pHomeWin: item.p_home_win,
      pHomeFirst: item.p_home_first,
      impHome: item.imp_home,
      edgeWin: item.edge_win,
      edgeFirst: item.edge_first,
      pickCode: item.pick_code,
      pickProb: item.pick_prob,
      tier: item.tier,
    }));
    return NextResponse.json({ date, source: STRIKECAST_BASE, updatedAt, picks }, { headers: { 'Cache-Control': 'no-store' } });
  } catch (error) {
    return NextResponse.json({ date, source: STRIKECAST_BASE, picks: [], error: error instanceof Error ? error.message : 'Error desconocido' }, { status: 502, headers: { 'Cache-Control': 'no-store' } });
  }
}

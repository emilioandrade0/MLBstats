import { NextRequest, NextResponse } from 'next/server';
import { latestRefreshRun, readStoredOdds } from '@/db/daily';
import { mexicoDate } from '@/lib/daily-ingest';

export const dynamic = 'force-dynamic';

export async function GET(request: NextRequest) {
  const startDate = request.nextUrl.searchParams.get('startDate') ?? mexicoDate();
  const endDate = request.nextUrl.searchParams.get('endDate') ?? mexicoDate(1);
  try {
    const [games, lastRun] = await Promise.all([readStoredOdds(startDate, endDate), latestRefreshRun()]);
    return NextResponse.json({
      sourceState: games.length ? 'REAL DATA' : 'UNAVAILABLE',
      source: 'Action Network · casas verificadas · guardado en StatsMLB DB',
      generatedAt: new Date().toISOString(),
      sourceUpdatedAt: games.reduce((latest, game) => game.lastUpdated > latest ? game.lastUpdated : latest, ''),
      selectionRule: 'Mejor moneyline disponible entre DraftKings, FanDuel, BetRivers y BetMGM.',
      games,
      lastRefresh: lastRun,
    }, { headers: { 'Cache-Control': 'no-store' } });
  } catch (error) {
    return NextResponse.json({ sourceState: 'UNAVAILABLE', source: 'StatsMLB DB', generatedAt: new Date().toISOString(), games: [], error: error instanceof Error ? error.message : 'Error desconocido' }, {
      status: 500, headers: { 'Cache-Control': 'no-store' },
    });
  }
}

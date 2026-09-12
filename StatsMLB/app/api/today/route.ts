import { NextRequest, NextResponse } from 'next/server';
import { mexicoDate, scheduleForDate } from '@/lib/daily-ingest';

export const dynamic = 'force-dynamic';

function validDate(value: string) {
  return /^\d{4}-\d{2}-\d{2}$/.test(value) && !Number.isNaN(Date.parse(`${value}T12:00:00Z`));
}

export async function GET(request: NextRequest) {
  const selectedDate = request.nextUrl.searchParams.get('date') ?? mexicoDate();
  if (!validDate(selectedDate)) return NextResponse.json({ error: 'Fecha inválida' }, { status: 400 });
  try {
    return NextResponse.json(await scheduleForDate(selectedDate), { headers: { 'Cache-Control': 'no-store' } });
  } catch (error) {
    return NextResponse.json({
      sourceState: 'UNAVAILABLE', source: 'MLB StatsAPI + StatsMLB DB', retrievedAt: new Date().toISOString(), selectedDate,
      games: [], error: error instanceof Error ? error.message : 'Error desconocido',
    }, { status: 502, headers: { 'Cache-Control': 'no-store' } });
  }
}

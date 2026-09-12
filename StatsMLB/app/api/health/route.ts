import { NextResponse } from 'next/server';
import { ensureDatabase, latestRefreshRun } from '@/db/daily';

export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    await ensureDatabase();
    const lastRun = await latestRefreshRun();
    return NextResponse.json({
      status: 'ok',
      database: 'ready',
      lastRefreshAt: lastRun?.finishedAt ?? null,
    }, { headers: { 'Cache-Control': 'no-store' } });
  } catch (error) {
    return NextResponse.json({
      status: 'error',
      database: 'unavailable',
      error: error instanceof Error ? error.message : 'Error desconocido',
    }, { status: 503, headers: { 'Cache-Control': 'no-store' } });
  }
}

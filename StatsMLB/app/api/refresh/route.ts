import { NextResponse } from 'next/server';
import { latestRefreshRun } from '@/db/daily';
import { refreshDailyData, type DailyRefreshProgress } from '@/lib/daily-ingest';

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

export async function GET() {
  return NextResponse.json({ lastRun: await latestRefreshRun() }, { headers: { 'Cache-Control': 'no-store' } });
}

export async function POST(request: Request) {
  const wantsStream = new URL(request.url).searchParams.get('stream') === '1'
    || request.headers.get('accept')?.includes('application/x-ndjson');

  if (wantsStream) {
    const encoder = new TextEncoder();
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        let connected = true;
        const send = (payload: { type: 'progress'; progress: DailyRefreshProgress } | { type: 'complete'; result: Awaited<ReturnType<typeof refreshDailyData>> } | { type: 'error'; error: string }) => {
          if (!connected) return;
          try {
            controller.enqueue(encoder.encode(`${JSON.stringify(payload)}\n`));
          } catch {
            connected = false;
          }
        };

        void (async () => {
          try {
            const result = await refreshDailyData(progress => send({ type: 'progress', progress }));
            send({ type: 'complete', result });
          } catch (error) {
            send({ type: 'error', error: error instanceof Error ? error.message : 'Error desconocido' });
          } finally {
            if (connected) {
              try {
                controller.close();
              } catch {
                connected = false;
              }
            }
          }
        })();
      },
    });

    return new Response(stream, {
      headers: {
        'Cache-Control': 'no-store, no-transform',
        'Content-Type': 'application/x-ndjson; charset=utf-8',
        'X-Content-Type-Options': 'nosniff',
      },
    });
  }

  try {
    const result = await refreshDailyData();
    return NextResponse.json(result, {
      status: result.status === 'error' ? 502 : 200,
      headers: { 'Cache-Control': 'no-store' },
    });
  } catch (error) {
    return NextResponse.json({ status: 'error', error: error instanceof Error ? error.message : 'Error desconocido' }, {
      status: 500, headers: { 'Cache-Control': 'no-store' },
    });
  }
}

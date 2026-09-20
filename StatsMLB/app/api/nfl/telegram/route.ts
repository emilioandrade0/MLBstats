import { NextResponse } from 'next/server';
import { env } from 'cloudflare:workers';
import { nflMessage, nflTelegramSettings } from '../../../../lib/nfl-telegram';

export const dynamic = 'force-dynamic';
function settings() {
  const bindings = env as unknown as Record<string, string | undefined>;
  return nflTelegramSettings(bindings, process.env);
}
export async function GET() {
  const { token, chat } = settings();
  return NextResponse.json({ configured: Boolean(token && chat), channel: 'NFL' });
}
export async function POST(request: Request) {
  const origin = request.headers.get('origin');
  if (!origin || origin !== new URL(request.url).origin) return NextResponse.json({ error: 'Origen no permitido.' }, { status: 403 });
  let body: { pick?: unknown; preview?: boolean }; let text;
  try { body = await request.json() as typeof body; text = nflMessage(body.pick); }
  catch (e) { return NextResponse.json({ error: e instanceof Error ? e.message : 'Solicitud inválida.' }, { status: 400 }); }
  if (body.preview === true) return NextResponse.json({ text, preview: true });
  const { token, chat } = settings();
  if (!token || !chat) return NextResponse.json({ error: 'Falta configurar el canal NFL. No se usará el canal MLB.' }, { status: 503 });
  try {
    const response = await fetch(`https://api.telegram.org/bot${token}/sendMessage`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ chat_id: chat, text }), signal: AbortSignal.timeout(12000) });
    const result = await response.json() as { ok?: boolean; result?: { message_id: number } };
    if (!response.ok || !result.ok) return NextResponse.json({ error: 'Telegram rechazó el envío. Verifica el canal NFL y los permisos del bot.' }, { status: 502 });
    return NextResponse.json({ sent: true, messageId: result.result?.message_id });
  } catch { return NextResponse.json({ error: 'No se pudo confirmar el envío. Revisa el canal antes de volver a intentar.' }, { status: 502 }); }
}

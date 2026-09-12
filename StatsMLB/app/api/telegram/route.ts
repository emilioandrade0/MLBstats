import { NextResponse } from 'next/server';
import { env } from 'cloudflare:workers';

export const dynamic = 'force-dynamic';

type PickType = 'ML' | 'F5_ML' | 'RL_MINUS_1_5' | 'RL_PLUS_1_5' | 'NRFI' | 'YRFI' | 'OVER' | 'UNDER';
const PICK_TYPES: PickType[] = ['ML', 'F5_ML', 'RL_MINUS_1_5', 'RL_PLUS_1_5', 'NRFI', 'YRFI', 'OVER', 'UNDER'];
const TEAM_BASED: PickType[] = ['ML', 'F5_ML', 'RL_MINUS_1_5', 'RL_PLUS_1_5'];
const TOTAL_BASED: PickType[] = ['OVER', 'UNDER'];

type TelegramPick = {
  pickNumber?: number;
  awayTeam?: string;
  homeTeam?: string;
  gameDate?: string;
  time?: string;
  pickTeam?: string;
  decimalOdds?: number;
  pickType?: PickType;
  live?: boolean;
  totalPoint?: number;
};

type TelegramEnvironment = {
  TELEGRAM_BOT_TOKEN?: string;
  TELEGRAM_CHAT_ID?: string;
};

function telegramEnvironment() {
  const bindings = env as unknown as TelegramEnvironment;
  return {
    token: bindings.TELEGRAM_BOT_TOKEN || process.env.TELEGRAM_BOT_TOKEN,
    chatId: bindings.TELEGRAM_CHAT_ID || process.env.TELEGRAM_CHAT_ID,
  };
}

function configured() {
  const telegram = telegramEnvironment();
  return Boolean(telegram.token && telegram.chatId);
}

function isShortText(value: unknown, maximum = 80): value is string {
  return typeof value === 'string' && value.trim().length > 0 && value.trim().length <= maximum;
}

function sameOrigin(request: Request) {
  const origin = request.headers.get('origin');
  const host = request.headers.get('x-forwarded-host') ?? request.headers.get('host');
  if (!origin || !host) return true;
  try {
    return new URL(origin).host === host;
  } catch {
    return false;
  }
}

export async function GET() {
  return NextResponse.json({ configured: configured() });
}

export async function POST(request: Request) {
  if (!sameOrigin(request)) return NextResponse.json({ error: 'Origen no permitido.' }, { status: 403 });
  const { token, chatId } = telegramEnvironment();
  if (!token || !chatId) return NextResponse.json({ error: 'Telegram todavía no está configurado.' }, { status: 503 });
  if (!/^\d+:[A-Za-z0-9_-]{20,}$/.test(token)) return NextResponse.json({ error: 'El token configurado no tiene un formato válido.' }, { status: 503 });

  let payload: TelegramPick;
  try {
    payload = await request.json() as TelegramPick;
  } catch {
    return NextResponse.json({ error: 'Solicitud inválida.' }, { status: 400 });
  }

  const pickNumber = Number(payload.pickNumber);
  const decimalOdds = Number(payload.decimalOdds);
  const pickType: PickType = PICK_TYPES.includes(payload.pickType as PickType) ? (payload.pickType as PickType) : 'ML';
  const live = payload.live === true;
  if (!Number.isInteger(pickNumber) || pickNumber < 1 || pickNumber > 999) return NextResponse.json({ error: 'El número de pick no es válido.' }, { status: 400 });
  if (!Number.isFinite(decimalOdds) || decimalOdds < 1.01 || decimalOdds > 100) return NextResponse.json({ error: 'El momio decimal no es válido.' }, { status: 400 });
  if (!isShortText(payload.awayTeam) || !isShortText(payload.homeTeam) || !isShortText(payload.time, 10)) return NextResponse.json({ error: 'Faltan datos del partido.' }, { status: 400 });
  if (!/^\d{2}:\d{2}$/.test(payload.time)) return NextResponse.json({ error: 'La hora debe usar el formato HH:MM.' }, { status: 400 });
  if (TEAM_BASED.includes(pickType)) {
    if (!isShortText(payload.pickTeam)) return NextResponse.json({ error: 'Falta el equipo del pick.' }, { status: 400 });
    if (payload.pickTeam !== payload.awayTeam && payload.pickTeam !== payload.homeTeam) return NextResponse.json({ error: 'El pick debe corresponder a uno de los equipos.' }, { status: 400 });
  }
  let totalPoint: number | null = null;
  if (TOTAL_BASED.includes(pickType)) {
    totalPoint = Number(payload.totalPoint);
    if (!Number.isFinite(totalPoint) || totalPoint <= 0 || totalPoint > 30) return NextResponse.json({ error: 'El total (line) no es válido.' }, { status: 400 });
  }
  const gameTime = new Date(payload.gameDate ?? '').getTime();
  if (!Number.isFinite(gameTime)) return NextResponse.json({ error: 'La fecha del partido no es válida.' }, { status: 400 });

  const pickLine = (() => {
    const team = (payload.pickTeam ?? '').trim();
    switch (pickType) {
      case 'ML': return `${team} ML`;
      case 'F5_ML': return `${team} ML (F5)`;
      case 'RL_MINUS_1_5': return `${team} -1.5 (Run Line)`;
      case 'RL_PLUS_1_5': return `${team} +1.5 (Run Line)`;
      case 'NRFI': return `NRFI (sin carrera 1er inning)`;
      case 'YRFI': return `YRFI (carrera en 1er inning)`;
      case 'OVER': return `OVER ${totalPoint}`;
      case 'UNDER': return `UNDER ${totalPoint}`;
    }
  })();

  const liveBadge = live ? `\n🔴 EN VIVO` : '';
  const text = `🚨 PICK #${pickNumber} | MLB${liveBadge}\n\n⚾️ ${payload.awayTeam.trim()} vs ${payload.homeTeam.trim()}\n🕣 Hora: ${payload.time}\n\n🎯 Pick: ${pickLine}\n💰 Momio: ${decimalOdds.toFixed(2)}`;
  try {
    const response = await fetch(`https://api.telegram.org/bot${token}/sendMessage`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ chat_id: chatId, text }),
      signal: AbortSignal.timeout(12_000),
    });
    const result = await response.json() as { ok?: boolean; description?: string; result?: { message_id?: number } };
    if (!response.ok || !result.ok) return NextResponse.json({ error: result.description ?? 'Telegram rechazó el mensaje.' }, { status: 502 });
    return NextResponse.json({ sent: true, messageId: result.result?.message_id, text });
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : 'No se pudo conectar con Telegram.' }, { status: 502 });
  }
}

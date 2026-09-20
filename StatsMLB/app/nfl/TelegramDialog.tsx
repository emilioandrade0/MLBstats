'use client';
import { useEffect, useRef, useState } from 'react';
import { nflMessage, type NflTelegramPick } from '../../lib/nfl-telegram';

const MARKETS: { value: NflTelegramPick['market']; label: string; needsTeam: boolean; needsLine: boolean; lineLabel?: string }[] = [
  { value: 'ML',     label: 'Money Line (ganador)',        needsTeam: true,  needsLine: false },
  { value: 'SPREAD', label: 'Hándicap (spread)',           needsTeam: true,  needsLine: true, lineLabel: 'Puntos de hándicap (ej. -3.5)' },
  { value: 'OVER',   label: 'OVER (más de X puntos)',      needsTeam: false, needsLine: true, lineLabel: 'Total (ej. 45.5)' },
  { value: 'UNDER',  label: 'UNDER (menos de X puntos)',   needsTeam: false, needsLine: true, lineLabel: 'Total (ej. 45.5)' },
  { value: 'LVD',    label: 'Margen · L/V/D',              needsTeam: false, needsLine: false },
];

const LVD_OPTIONS: { value: string; label: (home: string, away: string) => string }[] = [
  { value: 'L', label: h => `${h} gana por más de 6` },
  { value: 'V', label: (_h, a) => `${a} gana por más de 6` },
  { value: 'D', label: () => 'Diferencia de 6 puntos o menos' },
];

export default function TelegramDialog({ pick, configured, close }: { pick: NflTelegramPick; configured: boolean; close: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null);
  const busy = useRef(false);
  const [market, setMarket] = useState<NflTelegramPick['market']>(pick.market);
  const [side, setSide] = useState<string>(pick.side);
  const [line, setLine] = useState<string>(pick.line != null ? String(pick.line) : '');
  const [odds, setOdds] = useState(pick.decimalOdds ? pick.decimalOdds.toFixed(2) : '');
  const [number, setNumber] = useState(String(pick.pickNumber || 1));
  const [isLive, setIsLive] = useState(false);
  const [sending, setSending] = useState(false);
  const [sent, setSent] = useState(false);
  const [feedback, setFeedback] = useState('');
  useEffect(() => { dialog.current?.showModal(); }, []);

  const meta = MARKETS.find(m => m.value === market)!;
  // Cuando cambias de mercado, reset side a un valor válido
  useEffect(() => {
    if (meta.needsTeam) {
      if (side !== pick.home && side !== pick.away) setSide(pick.away);
    } else if (market === 'LVD') {
      if (!['L','V','D'].includes(side)) setSide('L');
    } else {
      setSide(market);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [market]);

  const numericLine = line.trim() ? Number(line.replace(',', '.')) : NaN;
  const numericOdds = Number(odds.replace(',', '.'));
  const numericPickNo = Number(number);

  const payload: NflTelegramPick = {
    ...pick,
    market,
    side: market === 'OVER' || market === 'UNDER' ? market : side,
    line: meta.needsLine && Number.isFinite(numericLine) ? numericLine : undefined,
    decimalOdds: numericOdds,
    pickNumber: numericPickNo,
  };
  let text = ''; let error = '';
  try { text = nflMessage(payload); } catch (e) { error = e instanceof Error ? e.message : 'Pick inválido'; }
  const preview = isLive && text ? text.replace('🏈 PICK', '🔴 EN VIVO · PICK') : text;

  async function send() {
    if (busy.current || sent || error) return;
    busy.current = true; setSending(true); setFeedback('');
    try {
      const r = await fetch('/api/nfl/telegram', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pick: payload, live: isLive }),
      });
      const data = await r.json() as { error?: string };
      if (!r.ok) throw Error(data.error || 'No se pudo enviar.');
      setSent(true); setFeedback('Pick enviado al canal NFL.');
    } catch (e) {
      setFeedback(e instanceof Error ? e.message : 'No se pudo confirmar. Revisa el canal antes de reintentar.');
    } finally { busy.current = false; setSending(false); }
  }

  return <dialog className="nfl-dialog" ref={dialog} aria-labelledby="nfl-send-title" onCancel={e => { e.preventDefault(); if (!sending) close(); }}>
    <h2 id="nfl-send-title">Enviar pick · canal NFL</h2>
    <p>Revisa el mensaje. No se enviará al canal de MLB.</p>
    <div className="nfl-controls">
      <label>Número de pick
        <input type="number" min="1" max="999" value={number} disabled={sending || sent} onChange={e => setNumber(e.target.value)} />
      </label>
      <label>Tipo de pick
        <select value={market} disabled={sending || sent} onChange={e => setMarket(e.target.value as NflTelegramPick['market'])}>
          {MARKETS.map(m => <option key={m.value} value={m.value}>{m.label}</option>)}
        </select>
      </label>
      <label className="nfl-live-toggle">
        <input type="checkbox" checked={isLive} disabled={sending || sent} onChange={e => setIsLive(e.target.checked)} />
        <span>🔴 En vivo</span>
      </label>
      <label>Momio decimal confirmado
        <input type="number" step="0.01" min="1.01" max="100" value={odds} disabled={sending || sent} onChange={e => setOdds(e.target.value)} />
      </label>
    </div>
    {meta.needsTeam && (
      <fieldset className="nfl-side-select">
        <legend>Selecciona el equipo</legend>
        <div className="nfl-side-grid">
          <button type="button" className={side === pick.away ? 'selected' : ''} disabled={sending || sent} onClick={() => setSide(pick.away)}>
            <strong>{pick.away}</strong><small>Visitante</small>
          </button>
          <button type="button" className={side === pick.home ? 'selected' : ''} disabled={sending || sent} onClick={() => setSide(pick.home)}>
            <strong>{pick.home}</strong><small>Local</small>
          </button>
        </div>
      </fieldset>
    )}
    {market === 'LVD' && (
      <fieldset className="nfl-side-select">
        <legend>Margen L/V/D</legend>
        <div className="nfl-side-grid nfl-lvd-grid">
          {LVD_OPTIONS.map(opt => (
            <button key={opt.value} type="button" className={side === opt.value ? 'selected' : ''} disabled={sending || sent} onClick={() => setSide(opt.value)}>
              <strong>{opt.value}</strong><small>{opt.label(pick.home, pick.away)}</small>
            </button>
          ))}
        </div>
      </fieldset>
    )}
    {meta.needsLine && (
      <div className="nfl-controls">
        <label>{meta.lineLabel || 'Línea'}
          <input type="number" step="0.5" min="-100" max="150" value={line} disabled={sending || sent} onChange={e => setLine(e.target.value)} />
        </label>
      </div>
    )}
    {error ? <p role="status" className="nfl-warning">{error}</p> : <pre>{preview}</pre>}
    {!configured && <p className="nfl-warning">Falta NFL_TELEGRAM_CHAT_ID. El bot de MLB se reutiliza, pero el destino NFL debe configurarse por separado.</p>}
    <p role="status">{feedback}</p>
    <footer>
      <button onClick={close} disabled={sending}>Cerrar</button>
      <button onClick={send} disabled={!configured || !!error || sending || sent}>{sent ? 'Enviado' : sending ? 'Enviando…' : 'Confirmar envío al canal NFL'}</button>
    </footer>
  </dialog>;
}

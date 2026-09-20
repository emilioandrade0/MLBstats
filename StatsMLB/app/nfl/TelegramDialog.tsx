'use client';
import { useEffect, useRef, useState } from 'react';
import { nflMessage, type NflTelegramPick } from '../../lib/nfl-telegram';

export default function TelegramDialog({ pick, configured, close }: { pick: NflTelegramPick; configured: boolean; close: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null); const busy = useRef(false);
  const [odds, setOdds] = useState(pick.decimalOdds ? String(pick.decimalOdds.toFixed(2)) : '');
  const [number, setNumber] = useState('1'); const [sending, setSending] = useState(false); const [sent, setSent] = useState(false); const [feedback, setFeedback] = useState('');
  useEffect(() => { dialog.current?.showModal(); }, []);
  const payload = { ...pick, decimalOdds: Number(odds), pickNumber: Number(number) };
  let text = ''; let error = '';
  try { text = nflMessage(payload); } catch (e) { error = e instanceof Error ? e.message : 'Pick inválido'; }
  async function send() {
    if (busy.current || sent || error) return;
    busy.current = true; setSending(true); setFeedback('');
    try {
      const r = await fetch('/api/nfl/telegram', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ pick: payload }) });
      const data = await r.json() as { error?: string }; if (!r.ok) throw Error(data.error || 'No se pudo enviar.');
      setSent(true); setFeedback('Pick enviado al canal NFL.');
    } catch (e) { setFeedback(e instanceof Error ? e.message : 'No se pudo confirmar. Revisa el canal antes de reintentar.'); }
    finally { busy.current = false; setSending(false); }
  }
  return <dialog className="nfl-dialog" ref={dialog} aria-labelledby="nfl-send-title" onCancel={e => { e.preventDefault(); if (!sending) close(); }}><h2 id="nfl-send-title">Enviar pick · canal NFL</h2><p>Revisa el mensaje. No se enviará al canal de MLB.</p><div className="nfl-controls"><label>Número de pick<input type="number" min="1" max="999" value={number} disabled={sending || sent} onChange={e => setNumber(e.target.value)} /></label><label>Momio decimal confirmado<input type="number" step="0.01" min="1.01" max="100" value={odds} disabled={sending || sent} onChange={e => setOdds(e.target.value)} /></label></div>{error ? <p role="status">{error}</p> : <pre>{text}</pre>}{!configured && <p>Falta NFL_TELEGRAM_CHAT_ID. El bot de MLB se reutiliza, pero el destino NFL debe configurarse por separado.</p>}<p role="status">{feedback}</p><footer><button onClick={close} disabled={sending}>Cerrar</button><button onClick={send} disabled={!configured || !!error || sending || sent}>{sent ? 'Enviado' : sending ? 'Enviando…' : 'Confirmar envío al canal NFL'}</button></footer></dialog>;
}

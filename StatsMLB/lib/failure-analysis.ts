// Retrospective diagnostics only. No global ranking or outcome selects a mask.
export const REFERENCE_MASK = 45;
export type FrozenGame = {
  gamePk: number; date: string; trainedThrough: string; away: string; home: string;
  awayScore: number; homeScore: number; actualWinner: string;
  maskProbabilitiesEncoded?: string; maskProbabilities?: (number | null)[];
};
export type FailureGame = {
  gamePk: number; date: string; trainedThrough: string; away: string; home: string;
  pick: string; winner: string; confidence: number; correct: boolean;
  withoutMarketPick: string | null;
};
export type SignalId = 'lowConfidence' | 'highConfidence' | 'away' | 'marketGroupDisagreement';
export const SIGNALS: { id: SignalId; label: string; detail: string }[] = [
  { id: 'lowConfidence', label: 'Confianza menor a 55%', detail: 'Probabilidad del pick por debajo de 55%.' },
  { id: 'highConfidence', label: 'Confianza de 70% o más', detail: 'Permite detectar sobreconfianza comparando probabilidad y acierto.' },
  { id: 'away', label: 'Pick visitante', detail: 'El motor elige al equipo visitante.' },
  { id: 'marketGroupDisagreement', label: 'Cambio al quitar mercado + descanso', detail: 'Las máscaras fijas 45 y 13 eligen lados distintos. No equivale a desacuerdo con las casas de apuestas.' },
];
export function normalizeGame(game: FrozenGame, encoding: string | undefined): FailureGame | null {
  if (!Number.isSafeInteger(game.gamePk) || !/^\d{4}-\d{2}-\d{2}$/.test(game.date) ||
    !/^\d{4}-\d{2}-\d{2}$/.test(game.trainedThrough) || game.trainedThrough >= game.date ||
    !game.away || !game.home || game.away === game.home ||
    !Number.isFinite(game.awayScore) || !Number.isFinite(game.homeScore) || game.awayScore === game.homeScore) return null;
  const winner = game.homeScore > game.awayScore ? game.home : game.away;
  if (game.actualWinner !== winner) return null;
  let packed: string | undefined;
  try {
    if (game.maskProbabilitiesEncoded && encoding === 'uint8-base64-midpoint') packed = atob(game.maskProbabilitiesEncoded);
  } catch { return null; }
  const probability = (mask: number) => {
    const value = packed && packed.length > mask ? (packed.charCodeAt(mask) + .5) / 256 : game.maskProbabilities?.[mask];
    return typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1 ? value : null;
  };
  const p = probability(REFERENCE_MASK);
  if (p == null) return null;
  const other = probability(13);
  const pick = p >= .5 ? game.home : game.away;
  return { gamePk: game.gamePk, date: game.date, trainedThrough: game.trainedThrough,
    away: game.away, home: game.home, pick, winner, correct: pick === winner,
    confidence: Math.max(p, 1-p), withoutMarketPick: other == null ? null : other >= .5 ? game.home : game.away };
}
export function signalValue(game: FailureGame, signal: SignalId): boolean | null {
  switch (signal) {
    case 'lowConfidence': return game.confidence < .55;
    case 'highConfidence': return game.confidence >= .70;
    case 'away': return game.pick === game.away;
    case 'marketGroupDisagreement': return game.withoutMarketPick == null ? null : game.withoutMarketPick !== game.pick;
  }
}
export function metrics(games: FailureGame[]) {
  const n = games.length, correct = games.filter(g => g.correct).length;
  const accuracy = n ? correct / n : null;
  const z2 = 1.96 ** 2;
  const center = accuracy == null ? 0 : (accuracy + z2 / (2*n)) / (1 + z2/n);
  const half = accuracy == null ? 0 : 1.96 * Math.sqrt(accuracy * (1-accuracy)/n + z2/(4*n*n)) / (1+z2/n);
  return { n, correct, failures: n-correct, accuracy,
    averageConfidence: n ? games.reduce((s,g) => s + g.confidence, 0)/n : null,
    interval: n ? [center-half, center+half] : null };
}
export function compareSignal(games: FailureGame[], signal: SignalId) {
  return { withSignal: metrics(games.filter(g => signalValue(g, signal) === true)),
    withoutSignal: metrics(games.filter(g => signalValue(g, signal) === false)),
    unknown: games.filter(g => signalValue(g, signal) == null).length };
}
export function evaluateFilter(games: FailureGame[], signal: SignalId) {
  // Missing signals retain the pick; they are never silently counted as agreement.
  const skipped = games.filter(g => signalValue(g, signal) === true);
  const kept = games.filter(g => signalValue(g, signal) !== true);
  return { before: metrics(games), after: metrics(kept), skipped: metrics(skipped),
    coverage: games.length ? kept.length / games.length : null };
}

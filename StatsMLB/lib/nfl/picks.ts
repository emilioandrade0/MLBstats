// Convert raw model probabilities into concrete picks.
//
// Two modes:
//  - "confidence": pick the side the model considers MOST LIKELY (default, intuitive).
//  - "value":      pick the side with the best EDGE vs. market-implied probability
//                  (only mathematically-recommended bets).

import { Game } from "./api";

export type PickMode = "confidence" | "value";

export type Pick = {
  side: string;
  line: string;
  confidence: number;    // model probability for chosen side
  edge: number | null;   // percentage points vs market-implied
  tone: "signal" | "accent" | "muted";
  recommend: boolean;
};

function americanToProb(ml: number | null | undefined): number | null {
  if (ml == null || !Number.isFinite(ml) || ml === 0) return null;
  return ml > 0 ? 100 / (ml + 100) : -ml / (-ml + 100);
}

export function americanToDecimal(ml: number | null | undefined): number | null {
  if (ml == null || !Number.isFinite(ml) || ml === 0) return null;
  return ml > 0 ? ml / 100 + 1 : 100 / -ml + 1;
}

function fmtAmerican(ml: number | null | undefined): string {
  if (ml == null) return "ML";
  const dec = americanToDecimal(ml);
  return dec == null ? "ML" : dec.toFixed(2);
}

// Color by confidence in "confidence" mode; by edge in "value" mode.
function toneFor(mode: PickMode, conf: number, edge: number | null): Pick["tone"] {
  if (mode === "value") {
    if (edge == null || edge < 2) return "muted";
    if (edge >= 5) return "signal";
    return "accent";
  }
  // confidence mode
  if (conf >= 0.65) return "signal";
  if (conf >= 0.55) return "accent";
  return "muted";
}

export function moneylinePick(g: Game, mode: PickMode = "confidence"): Pick | null {
  const pHome = g.model_home_win_prob;
  if (pHome == null) return null;
  const pAway = 1 - pHome;
  const impHome = americanToProb(g.home_moneyline);
  const impAway = americanToProb(g.away_moneyline);
  const homeEdge = impHome != null ? (pHome - impHome) * 100 : null;
  const awayEdge = impAway != null ? (pAway - impAway) * 100 : null;

  let side: "home" | "away";
  if (mode === "value" && homeEdge != null && awayEdge != null) {
    side = homeEdge >= awayEdge ? "home" : "away";
  } else {
    side = pHome >= 0.5 ? "home" : "away";
  }

  const conf = side === "home" ? pHome : pAway;
  const edge = side === "home" ? homeEdge : awayEdge;
  return {
    side: side === "home" ? g.home_team : g.away_team,
    line: fmtAmerican(side === "home" ? g.home_moneyline : g.away_moneyline),
    confidence: conf,
    edge,
    tone: toneFor(mode, conf, edge),
    recommend: mode === "value" ? (edge != null && edge >= 2) : conf >= 0.55,
  };
}

export function spreadPick(g: Game, mode: PickMode = "confidence"): Pick | null {
  const p = g.model_cover_home_prob;
  if (p == null || g.spread_line == null) return null;
  const IMPLIED = 0.524;
  let takeHome: boolean;
  if (mode === "value") {
    const homeEdge = (p - IMPLIED) * 100;
    const awayEdge = ((1 - p) - IMPLIED) * 100;
    takeHome = homeEdge >= awayEdge;
  } else {
    takeHome = p >= 0.5;
  }
  const conf = takeHome ? p : 1 - p;
  const edge = (conf - IMPLIED) * 100;
  const homeVegas = -g.spread_line;
  const awayVegas = g.spread_line;
  const shown = takeHome ? homeVegas : awayVegas;
  return {
    side: takeHome ? g.home_team : g.away_team,
    line: shown > 0 ? `+${shown.toFixed(1)}` : shown.toFixed(1),
    confidence: conf,
    edge,
    tone: toneFor(mode, conf, edge),
    recommend: mode === "value" ? edge >= 2 : conf >= 0.55,
  };
}

export function totalPick(g: Game, mode: PickMode = "confidence"): Pick | null {
  const p = g.model_total_over_prob;
  if (p == null || g.total_line == null) return null;
  const IMPLIED = 0.524;
  const takeOver = mode === "value"
    ? (p - IMPLIED) >= ((1 - p) - IMPLIED)
    : p >= 0.5;
  const conf = takeOver ? p : 1 - p;
  const edge = (conf - IMPLIED) * 100;
  return {
    side: takeOver ? "OVER" : "UNDER",
    line: g.total_line.toFixed(1),
    confidence: conf,
    edge,
    tone: toneFor(mode, conf, edge),
    recommend: mode === "value" ? edge >= 2 : conf >= 0.55,
  };
}

export function pickResult(pick: Pick, market: "ml" | "spread" | "total", g: Game): "win" | "loss" | "push" | null {
  const hs = g.home_score, as = g.away_score;
  if (hs == null || as == null) return null;
  if (market === "ml") {
    if (hs === as) return "push";
    const homeWon = hs > as;
    return (pick.side === g.home_team ? homeWon : !homeWon) ? "win" : "loss";
  }
  if (market === "spread" && g.spread_line != null) {
    const adjusted = hs - as - g.spread_line;
    if (adjusted === 0) return "push";
    const homeCovered = adjusted > 0;
    return (pick.side === g.home_team ? homeCovered : !homeCovered) ? "win" : "loss";
  }
  if (market === "total" && g.total_line != null) {
    const total = hs + as;
    if (total === g.total_line) return "push";
    const wentOver = total > g.total_line;
    return (pick.side === "OVER" ? wentOver : !wentOver) ? "win" : "loss";
  }
  return null;
}

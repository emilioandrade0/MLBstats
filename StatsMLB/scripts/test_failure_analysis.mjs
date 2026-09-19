import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import { normalizeGame, metrics, compareSignal, evaluateFilter } from '../lib/failure-analysis.ts';

const raw = { gamePk: 1, date: '2026-09-18', trainedThrough: '2026-08-31', away: 'KC', home: 'PIT', awayScore: 1, homeScore: 3, actualWinner: 'PIT', maskProbabilities: Array(46).fill(.53) };
test('rejects results that cannot form a valid pregame diagnostic', () => {
  assert.equal(normalizeGame({ ...raw, trainedThrough: raw.date }), null);
  assert.equal(normalizeGame({ ...raw, actualWinner: 'KC' }), null);
  assert.equal(normalizeGame({ ...raw, homeScore: 1 }), null);
  assert.equal(normalizeGame({ ...raw, maskProbabilities: [] }), null);
});
test('decodes the fixed mask and rejects invalid base64', () => {
  const bytes = Buffer.alloc(46, 130); bytes[45] = 240; bytes[13] = 50;
  const result = normalizeGame({ ...raw, maskProbabilities: undefined, maskProbabilitiesEncoded: bytes.toString('base64') }, 'uint8-base64-midpoint');
  assert.equal(result.confidence, 240.5/256);
  assert.equal(result.pick, 'PIT'); assert.equal(result.withoutMarketPick, 'KC');
  assert.equal(normalizeGame({ ...raw, maskProbabilitiesEncoded: '%!' }, 'uint8-base64-midpoint'), null);
});
test('filter accounts for both avoided failures and discarded wins, retaining unknown signals', () => {
  const base = normalizeGame(raw);
  const games = [{ ...base, correct: false, withoutMarketPick: 'KC' }, { ...base, gamePk: 2, withoutMarketPick: 'KC' }, { ...base, gamePk: 3, withoutMarketPick: null }];
  const result = evaluateFilter(games, 'marketGroupDisagreement');
  assert.equal(result.skipped.failures, 1); assert.equal(result.skipped.correct, 1);
  assert.equal(result.after.n, 1); assert.equal(result.coverage, 1/3);
  assert.equal(compareSignal(games, 'marketGroupDisagreement').unknown, 1);
});
test('empty samples never become zero percent and Wilson intervals contain the measured rate', () => {
  assert.equal(metrics([]).accuracy, null); assert.equal(metrics([]).interval, null);
  const m = metrics([normalizeGame(raw), { ...normalizeGame(raw), correct: false }]);
  assert.ok(m.interval[0] < .5 && m.interval[1] > .5);
});
test('published historical scores independently agree with every normalized pick result', () => {
  const manifest = JSON.parse(readFileSync(new URL('../public/data/walkforward.json', import.meta.url)));
  let count = 0;
  for (const file of manifest.dayFiles) {
    const shard = JSON.parse(readFileSync(new URL(`../public/data/${file}`, import.meta.url)));
    for (const day of Object.values(shard.days)) for (const raw of day.games) {
      const game = normalizeGame(raw, manifest.probabilityEncoding);
      assert.ok(game, `Invalid game ${raw.gamePk}`);
      const winner = raw.homeScore > raw.awayScore ? raw.home : raw.away;
      assert.equal(game.correct, game.pick === winner);
      assert.ok(game.trainedThrough < game.date); count++;
    }
  }
  assert.ok(count > 6000); console.log(`Validated ${count} historical games against final scores.`);
});

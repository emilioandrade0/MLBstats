"""Bounded retrospective experiment. Never changes live recommendations.

Protocol v1: fixed WF45 niche (<.55), fixed feature masks, four candidates,
monthly expanding fits, select on 2025 only, measure selected policy on 2026+.
No hyperparameter search or threshold selection on evaluation outcomes.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
MASKS = [1, 2, 4, 8, 16, 32, 13, 31, 45, 63]
FEATURES = [f'logit_mask_{m}_toward_base_pick' for m in MASKS] + ['base_picks_home']
CANDIDATES = [
    ('niche_ridge_strong', 'Especialista lineal conservador', 'niche', .1),
    ('niche_ridge', 'Especialista lineal', 'niche', 1.),
    ('pooled_ridge', 'Modelo general con énfasis en el nicho', 'pooled', .1),
    ('niche_trees', 'Especialista de árboles pequeños', 'trees', 0.),
]


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def load_data(directory):
    manifest = read(directory/'walkforward.json')
    assert manifest['probabilityEncoding'] == 'uint8-base64-midpoint'
    assert manifest['factorKeys'][:6] == ['localia', 'strength', 'recent', 'runDiff', 'series', 'marketSchedule']
    games, probabilities, hashes, seen = [], [], {}, set()
    for name in ['walkforward.json', *manifest['dayFiles']]:
        assert Path(name).name == name
        hashes[name] = hashlib.sha256((directory/name).read_bytes()).hexdigest()
        if name == 'walkforward.json':
            continue
        for day in read(directory/name)['days'].values():
            for g in day['games']:
                assert g['gamePk'] not in seen, 'Duplicate game'
                seen.add(g['gamePk'])
                assert g['trainedThrough'] < g['date'], 'Non-pregame training cutoff'
                assert g['homeScore'] != g['awayScore'], 'Unresolved game'
                assert g['actualWinner'] == (g['home'] if g['homeScore'] > g['awayScore'] else g['away'])
                p = np.frombuffer(base64.b64decode(g['maskProbabilitiesEncoded'], validate=True), dtype=np.uint8).astype(float)
                assert len(p) > max(MASKS)
                games.append(g)
                probabilities.append((p[MASKS]+.5)/256)
    order = sorted(range(len(games)), key=lambda i: (games[i]['date'], games[i]['gamePk']))
    games = [games[i] for i in order]
    return games, np.array(probabilities)[order], hashes, manifest


def features(probs):
    base = probs[:, MASKS.index(45)]
    side = np.where(base >= .5, 1., -1.)
    return np.column_stack([np.log(probs/(1-probs))*side[:, None], (side > 0).astype(float)])


def predict_monthly(x, base, y, dates, candidate):
    _, _, family, c = candidate
    months = np.array([d[:7] for d in dates])
    niche = np.maximum(base, 1-base) < .55
    target = ((base >= .5) == y).astype(int)
    prediction = np.full(len(y), np.nan)
    folds = []
    for month in sorted(set(months)):
        if month < '2025-01':
            continue
        train = months < month
        if family != 'pooled':
            train &= niche
        test = (months == month) & niche
        if not test.any():
            continue
        assert train.sum() >= 200 and len(np.unique(target[train])) == 2, 'Insufficient warm-up'
        if family == 'trees':
            model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=4, max_depth=2,
                min_samples_leaf=80, l2_regularization=10, learning_rate=.05, early_stopping=False, random_state=190926)
            model.fit(x[train], target[train])
        else:
            model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=1500, random_state=190926))
            weights = np.where(niche[train], 3., 1.) if family == 'pooled' else np.ones(train.sum())
            model.fit(x[train], target[train], logisticregression__sample_weight=weights)
        prediction[test] = model.predict_proba(x[test])[:, 1]
        folds.append({'month': month, 'trainedThrough': str(max(dates[train])),
            'trainGames': int(train.sum()), 'testGames': int(test.sum())})
    return prediction, folds


def apply_policy(base, probability_correct):
    # Fixed threshold: reverse only when the meta-model estimates <45% correctness.
    reverse = np.isfinite(probability_correct) & (probability_correct < .45)
    raw_home = np.where(base >= .5, probability_correct, 1-probability_correct)
    return np.where(reverse, raw_home, base)


def score(base, after, y, dates, mask, bootstrap=False):
    before_hit = (base[mask] >= .5) == y[mask]
    after_hit = (after[mask] >= .5) == y[mask]
    delta = after_hit.astype(int)-before_hit.astype(int)
    n = int(mask.sum())
    result = {'games': n, 'beforeCorrect': int(before_hit.sum()), 'afterCorrect': int(after_hit.sum()),
        'beforeAccuracy': float(before_hit.mean()) if n else None, 'afterAccuracy': float(after_hit.mean()) if n else None,
        'rescued': int((delta > 0).sum()), 'damaged': int((delta < 0).sum()), 'netCorrect': int(delta.sum()),
        'changed': int(((base[mask] >= .5) != (after[mask] >= .5)).sum()),
        'deltaPoints': float(delta.mean()*100) if n else None,
        'beforeBrier': float(np.mean((base[mask]-y[mask])**2)) if n else None,
        'afterBrier': float(np.mean((after[mask]-y[mask])**2)) if n else None}
    if bootstrap and n:
        dd = dates[mask]
        days = np.unique(dd)
        sums = np.array([delta[dd == d].sum() for d in days])
        counts = np.array([(dd == d).sum() for d in days])
        rng = np.random.default_rng(190926)
        sampled = rng.integers(0, len(days), size=(3000, len(days)))
        boot = 100*sums[sampled].sum(axis=1)/counts[sampled].sum(axis=1)
        result['deltaCI95'] = np.quantile(boot, [.025, .975]).tolist()
    return result


def run(directory):
    games, probs, hashes, manifest = load_data(directory)
    base = probs[:, MASKS.index(45)]
    dates = np.array([g['date'] for g in games])
    y = np.array([g['actualWinner'] == g['home'] for g in games], dtype=int)
    assert max(dates) <= datetime.now(timezone.utc).date().isoformat()
    x = features(probs)
    niche = np.maximum(base, 1-base) < .55
    design = niche & (dates >= '2025-01-01') & (dates < '2026-01-01')
    evaluation = niche & (dates >= '2026-01-01')
    predictions, folds, candidates = {}, {}, []
    for candidate in CANDIDATES:
        key, label, _, _ = candidate
        meta, folds[key] = predict_monthly(x, base, y, dates, candidate)
        assert np.isfinite(meta[design | evaluation]).all(), 'Incomplete paired evaluation'
        predictions[key] = apply_policy(base, meta)
        candidates.append({'id': key, 'label': label, 'design': score(base, predictions[key], y, dates, design)})
    # Selection does not inspect any evaluation metric. Include unchanged base as control.
    candidates.append({'id': 'base', 'label': 'Conservar el motor actual', 'design': score(base, base, y, dates, design)})
    predictions['base'] = base.copy()
    selected = min(candidates, key=lambda row: (-row['design']['netCorrect'], row['design']['changed'], row['id']))
    key = selected['id']
    after = predictions[key]
    measured = score(base, after, y, dates, evaluation, bootstrap=True)
    periods = {label: score(base, after, y, dates, evaluation & condition, bootstrap=True) for label, condition in {
        'h1': dates < '2026-07-01', 'h2': dates >= '2026-07-01'}.items()}
    monthly = [{'month': month, **score(base, after, y, dates, evaluation & np.char.startswith(dates, month))}
        for month in sorted(set(d[:7] for d in dates[evaluation]))]
    evidence = bool(key != 'base' and measured['deltaCI95'][0] > 0 and
        all(p['netCorrect'] >= 0 for p in periods.values()) and measured['afterBrier'] <= measured['beforeBrier'])
    report = {'protocol': 'low-confidence-specialist-v1', 'generatedAt': datetime.now(timezone.utc).isoformat(),
        'sourceGeneratedAt': manifest.get('generatedAt'), 'dataThrough': str(max(dates)),
        'versions': {'numpy': np.__version__, 'sklearn': sklearn.__version__},
        'method': {'referenceMask': 45, 'nicheBelow': .55, 'reverseIfCorrectnessBelow': .45,
            'featureMasks': MASKS, 'features': FEATURES, 'warmup': '2024', 'selection': '2025', 'evaluationFrom': '2026-01-01',
            'training': 'Cada mes usa solo partidos de meses anteriores; aciertos y fallos. La familia se selecciona una vez en 2025.',
            'limitations': 'Exploración retrospectiva con histórico ya consultado. No es una prueba prospectiva ni el motor LOCK/FUERTE de la web. Sin datos nuevos de abridores, bullpen o alineaciones. La procedencia pregame de las entradas originales no queda certificada por el corte de entrenamiento.',
            'interval': 'Bootstrap pareado por día, 3000 réplicas; no corrige selección histórica de factores.'},
        'coverage': {'allGames': len(games), 'nicheGames': int(niche.sum()), 'evaluationGames': int(evaluation.sum()),
            'keptFraction': 1., 'outsideNicheChanged': int(np.sum((base[~niche] >= .5) != (after[~niche] >= .5)))},
        'selected': {'id': key, 'label': selected['label'], 'activated': False, 'numericalGatePassed': evidence,
            'reason': 'Solo experimento: falta validación prospectiva y paridad con el motor en vivo.' if evidence else 'No cumple los requisitos de mejora estable; se conserva el motor actual.'},
        'candidates': candidates, 'evaluation': measured, 'periods': periods, 'monthly': monthly,
        'homeOnlyControl': score(base, np.full(len(base), .501), y, dates, evaluation),
        'overallImpact': score(base, after, y, dates, dates >= '2026-01-01'),
        'folds': folds.get(key, []), 'sourceHashes': hashes,
        'changes': [{'gamePk': g['gamePk'], 'date': g['date'], 'away': g['away'], 'home': g['home'],
            'before': g['home'] if base[i] >= .5 else g['away'], 'after': g['home'] if after[i] >= .5 else g['away'],
            'winner': g['actualWinner'], 'effect': 'rescatado' if (after[i] >= .5) == y[i] else 'estropeado'}
            for i, g in enumerate(games) if evaluation[i] and (base[i] >= .5) != (after[i] >= .5)]}
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=ROOT/'public/data')
    parser.add_argument('--output', type=Path, default=ROOT/'public/data/low-confidence-specialist.json')
    args = parser.parse_args()
    report = run(args.data_dir)
    args.output.write_text(json.dumps(report, ensure_ascii=False, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['selected', 'evaluation', 'periods', 'homeOnlyControl']}, ensure_ascii=False, indent=2))

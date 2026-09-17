"""Reproducible retrospective audit. Never rewrites source data or live models.

Run with Python + numpy + pandas + pyarrow. An optional local pyarrow install
under outputs/audit-runtime is supported. Only completed dates before --as-of
are eligible. Candidate families are selected on 2024-2025, then measured on
2026 H1/H2 without reselection. Previously explored data are NOT a new holdout.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parents[1]
DATA = APP / 'public' / 'data'
sys.path.insert(0, str(APP / 'outputs' / 'audit-runtime'))
import pandas as pd


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return round(float(value), 8) if math.isfinite(value) else None
    return value


def metrics(p, y, dates, eligible=None):
    valid = np.isfinite(p)
    if eligible is not None:
        valid &= eligible
    p, y, days = p[valid], y[valid], dates[valid]
    if not len(y):
        return {'games': 0, 'accuracy': None, 'brier': None, 'correct': 0}
    hit = (p >= .5) == y
    n, accuracy = len(y), float(hit.mean())
    z = 1.96
    center = (accuracy + z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(accuracy*(1-accuracy)/n+z*z/(4*n*n))/(1+z*z/n)
    grouped = defaultdict(list)
    for day, win in zip(days, hit):
        grouped[str(day)].append(win)
    conf = np.maximum(p, 1-p)
    bands = []
    ece = 0.
    for lo, hi in [(0.5,.55),(.55,.60),(.60,.65),(.65,.70),(.70,1.01)]:
        ix = (conf >= lo) & (conf < hi)
        if ix.any():
            acc, avg = hit[ix].mean(), conf[ix].mean()
            ece += ix.mean()*abs(acc-avg)
            bands.append({'label': f'{lo*100:.0f}–{min(hi,1)*100:.0f}%', 'games':int(ix.sum()), 'accuracy':acc, 'confidence':avg})
    p = np.clip(p, 1e-6, 1-1e-6)
    return {'games': n, 'correct': int(hit.sum()), 'accuracy': accuracy,
            'accuracyCI95':[center-half,center+half], 'brier':float(np.mean((p-y)**2)),
            'logLoss':float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))), 'ece':ece,
            'days':len(grouped), 'winningDays':sum(np.mean(v)>.5 for v in grouped.values()), 'bands':bands}


def paired(before, after, y, dates, eligible):
    ix = eligible & np.isfinite(before) & np.isfinite(after)
    before, after, yy, dd = before[ix], after[ix], y[ix], dates[ix]
    if not len(yy):
        return {'games':0}
    delta = ((after >= .5) == yy).astype(float)-((before >= .5) == yy)
    days = np.unique(dd)
    sums = np.array([delta[dd == d].sum() for d in days])
    counts = np.array([(dd == d).sum() for d in days])
    rng = np.random.default_rng(20260916)
    samples = rng.integers(0,len(days),size=(2000,len(days)))
    boot = sums[samples].sum(1)/counts[samples].sum(1)*100
    return {'games':len(yy), 'deltaPoints':float(delta.mean()*100),
            'netCorrect':int(delta.sum()), 'rescued':int((delta>0).sum()), 'lost':int((delta<0).sum()),
            'deltaCI95Points':np.quantile(boot,[.025,.975]).tolist(),
            'before':metrics(before,yy,dd), 'after':metrics(after,yy,dd)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--as-of', default=date.today().isoformat())
    args = parser.parse_args()
    asof = args.as_of
    manifest = read(DATA/'walkforward.json')
    sources = ['walkforward.json', *(manifest.get('dayFiles') or ['walkforward-2024.json','walkforward-2025.json','walkforward-2026.json']), 'odds-history.json', 'strikecast-history.json', 'stacked-calibrator.json', 'player-performance.json', 'stats.json']
    source_hashes = {f:hashlib.sha256((DATA/f).read_bytes()).hexdigest() for f in sources}
    all_games = []
    for name in sources[1:1+len(manifest.get('dayFiles') or [2024,2025,2026])]:
        for day in read(DATA/name)['days'].values():
            all_games.extend(day['games'])
    games = sorted([g for g in all_games if g['date'] < asof],key=lambda g:(g['date'],g['gamePk']))
    assert len({g['gamePk'] for g in games}) == len(games), 'Duplicate game IDs'
    assert all(g['trainedThrough'] < g['date'] for g in games), 'Future training boundary'
    dates = np.array([g['date'] for g in games])
    months = np.array([d[:7] for d in dates])
    y = np.array([int(g['actualWinner']==g['home']) for g in games])
    probs = np.array([np.frombuffer(base64.b64decode(g['maskProbabilitiesEncoded']),dtype=np.uint8) for g in games],dtype=np.float64)/256+1/512
    assert probs.shape[1] == 2**len(manifest['factorKeys'])
    probs[:,0] = .5
    base_mask = manifest['defaultMask']
    base = probs[:,base_mask].copy()
    canonical = pd.read_parquet(APP.parent/'data/processed/games.parquet')
    canonical_by_pk = canonical.set_index('game_pk').to_dict('index')
    mismatches = []
    for g in games:
        c = canonical_by_pk.get(g['gamePk'])
        if c and pd.notna(c['home_score']) and pd.notna(c['away_score']):
            if (int(c['home_score']),int(c['away_score'])) != (g['homeScore'],g['awayScore']):
                mismatches.append(g['gamePk'])
    sc = read(DATA/'strikecast-history.json')['games']
    odds = read(DATA/'odds-history.json')['games']
    sc_bad_targets = 0
    sc_raw, sc_blend, market = [], [], []
    for g in games:
        s = sc.get(str(g['gamePk']),{})
        if s.get('homeWin') is not None and s['homeWin'] != int(g['actualWinner']==g['home']):
            sc_bad_targets += 1
            s = {}
        sc_raw.append(s.get('pRaw',np.nan))
        sc_blend.append(s.get('pBlend',np.nan))
        market.append(odds.get(str(g['gamePk']),{}).get('marketPHome',np.nan))
    sc_raw, sc_blend, market = [np.array(v,dtype=float) for v in (sc_raw,sc_blend,market)]
    # Only masks whose declared dependency is satisfiable.
    masks = np.array([m for m in range(1,probs.shape[1]) if not (m & 1024) or (m & 256 and m & 512)])
    hits = (probs >= .5) == y[:,None]
    adaptive = {name:np.full(len(y),np.nan) for name in ['past_best','past_top5','past_brier','past_15','past_days']}
    fold_selection = []
    for month in np.unique(months):
        train, test = months < month, months == month
        if train.sum() < 500:
            continue
        scores = hits[train][:,masks].mean(0)
        order = np.lexsort((masks,-scores))
        best = int(masks[order[0]])
        top = masks[order[:5]]
        best_brier = int(masks[np.argmin(((probs[train][:,masks]-y[train,None])**2).mean(0))])
        day_sizes = {d:int((dates==d).sum()) for d in np.unique(dates[train])}
        d15 = train & np.array([day_sizes.get(d,0)==15 for d in dates])
        best15 = int(masks[np.argmax(hits[d15][:,masks].mean(0))]) if d15.any() else best
        daily_hits = np.array([hits[dates==d][:,masks].mean(0) > .5 for d in day_sizes])
        best_days = int(masks[np.argmax(daily_hits.sum(0))])
        adaptive['past_best'][test] = probs[test,best]
        adaptive['past_top5'][test] = probs[test][:,top].mean(1)
        adaptive['past_brier'][test] = probs[test,best_brier]
        adaptive['past_15'][test] = probs[test,best15]
        adaptive['past_days'][test] = probs[test,best_days]
        fold_selection.append({'month':month,'selectedThrough':str(max(dates[train])),'trainGames':int(train.sum()),'bestMask':best,'top5Masks':top.tolist(),'brierMask':best_brier,'fifteenMask':best15,'daysMask':best_days})
    fallback_market = np.where(np.isfinite(market),market,base)
    fallback_sc = np.where(np.isfinite(sc_blend),sc_blend,base)
    candidates = {'base':base, **adaptive,
        'base_quality_fatigue':probs[:,base_mask|256|512],
        'base_market_25':.75*base+.25*fallback_market,
        'base_market_50':.5*base+.5*fallback_market,
        'base_market_75':.25*base+.75*fallback_market,
        'base_strikecast_50':.5*base+.5*fallback_sc,
        'base_sc_market':.5*base+.25*fallback_sc+.25*fallback_market,
        'past_best_market_50':.5*adaptive['past_best']+.5*fallback_market,
    }
    labels = {'base':'Base activa · máscara 45','past_best':'Mejor combinación seleccionada solo con meses anteriores',
        'past_top5':'Promedio de las 5 mejores combinaciones anteriores','past_brier':'Combinación anterior con menor Brier',
        'past_15':'Mejor combinación anterior en jornadas de 15','past_days':'Más días ganadores en meses anteriores',
        'base_quality_fatigue':'Base + calidad de rotación + fatiga','base_market_25':'75% base + 25% mercado',
        'base_market_50':'50% base + 50% mercado','base_market_75':'25% base + 75% mercado',
        'base_strikecast_50':'50% base + 50% StrikeCast','base_sc_market':'50% base + 25% StrikeCast + 25% mercado',
        'past_best_market_50':'50% mejor combinación previa + 50% mercado'}
    common = np.logical_and.reduce([np.isfinite(v) for v in candidates.values()])
    periods = {'design':common & (dates<'2026-01-01'),'validation2026':common & (dates>='2026-01-01'),
        'h1':common & (dates>='2026-01-01') & (dates<'2026-07-01'),'h2':common & (dates>='2026-07-01')}
    # ONE selection, before evaluating either 2026 period.
    selected = sorted([k for k in candidates if k!='base'],key=lambda k:(-metrics(candidates[k],y,dates,periods['design'])['accuracy'], metrics(candidates[k],y,dates,periods['design'])['brier'],k))[0]
    rows = [{'id':key,'label':labels[key], 'periods':{name:metrics(p,y,dates,ix) for name,ix in periods.items()}} for key,p in candidates.items()]
    before_after = {name:paired(base,candidates[selected],y,dates,ix) for name,ix in periods.items()}
    # A probability-only correction chosen on design data, never on 2026.
    caps = [.60,.65,.70,.75,.80,.90,1.0]
    cap = min(caps, key=lambda c:metrics(np.clip(candidates[selected],1-c,c),y,dates,periods['design'])['brier'])
    capped = np.clip(candidates[selected],1-cap,cap)
    assert np.array_equal(capped[common]>=.5,candidates[selected][common]>=.5)
    probability_correction = {'cap':cap,'selectionPeriod':'2024-2025; exploración retrospectiva',
        'periods':{name:paired(base,capped,y,dates,ix) for name,ix in periods.items()}}
    filters = []
    for threshold in [.50,.55,.60,.65]:
        ix = periods['validation2026'] & (np.maximum(candidates[selected],1-candidates[selected])>=threshold)
        filters.append({'threshold':threshold,'coverage':float(ix.sum()/periods['validation2026'].sum()),
            'beforeSameGames':metrics(base,y,dates,ix),'after':metrics(candidates[selected],y,dates,ix)})
    # Paired ablations: same games, one factor toggled. Descriptive, not new selection.
    ablations = [{'factor':key,'active':bool(base_mask & (1<<i)), 'mask':base_mask^(1<<i),
        'metrics':metrics(probs[:,base_mask^(1<<i)],y,dates,periods['validation2026']),
        'deltaPoints':float((hits[periods['validation2026'],base_mask^(1<<i)].mean()-hits[periods['validation2026'],base_mask].mean())*100)}
        for i,key in enumerate(manifest['factorKeys'])]
    current_top = max(manifest['combinations'],key=lambda c:c['accuracy'])['mask']
    optimistic = {'mask':current_top,'label':'Mejor global actual (selección retrospectiva, no holdout)',
        'all':metrics(probs[:,current_top],y,dates),'chronologicalSelector':metrics(adaptive['past_best'],y,dates,common)}
    # Static buckets fitted using all outcomes vs expanding monthly buckets.
    def bucket(i):
        pp=probs[i,1019]; home=pp>=.5; conf=max(pp,1-pp)
        cb='50-55' if conf<.55 else '55-60' if conf<.60 else '60-65' if conf<.65 else '65-70' if conf<.70 else '70+'
        if np.isfinite(sc_raw[i]):
            agree='y' if (sc_raw[i]>=.5)==home else 'n'
            sp=sc_raw[i] if home else 1-sc_raw[i]
            return f"{'home' if home else 'away'}|{cb}|{agree}|{'w' if sp<.45 else 'm' if sp<.55 else 's'}"
        return f"{'home' if home else 'away'}|{cb}|nosc"
    buckets=np.array([bucket(i) for i in range(len(y))]); cal=read(DATA/'stacked-calibrator.json')
    frozen_rate=np.full(len(y),np.nan); static_rate=np.full(len(y),np.nan)
    static={b['key']:b for b in cal['buckets']}
    for i,b in enumerate(buckets):
        if b in static and static[b]['n']>=40: static_rate[i]=static[b]['winRate']
    for month in np.unique(months):
        for b in np.unique(buckets[months==month]):
            ix=(months<month)&(buckets==b)
            if ix.sum()>=40: frozen_rate[(months==month)&(buckets==b)]=hits[ix,1019].mean()
    cal_rows=[]
    for lo,hi,label in [(.65,1.01,'LOCK'),(.60,.65,'FUERTE'),(.55,.60,'JUEGA'),(0,.55,'JUEGA-B')]:
        cal_rows.append({'tier':label,'static':metrics(probs[:,1019],y,dates,periods['validation2026']&(static_rate>=lo)&(static_rate<hi)),
            'pastOnly':metrics(probs[:,1019],y,dates,periods['validation2026']&(frozen_rate>=lo)&(frozen_rate<hi))})
    # Value predictions are in-sample after retraining on all labeled data; report separately.
    value=pd.read_parquet(APP.parent/'data/processed/value_picks.parquet').set_index('game_pk')
    value_p=np.array([value.loc[g['gamePk'],'p_home_win'] if g['gamePk'] in value.index else np.nan for g in games])
    value_result=metrics(value_p,y,dates,periods['validation2026'])
    # Reconstruct lineup signal with day-specific snapshots and actual starters.
    # Actual starters are known after the game; without announcement timestamps this is exploratory.
    pb=pd.read_parquet(APP.parent/'data/processed/player_box.parquet',columns=['game_pk','side','player_id','started_batting'])
    starters=pb[pb.started_batting.fillna(False)].groupby(['game_pk','side']).player_id.apply(list).to_dict()
    alignment=np.full(len(y),np.nan); alignment_neutral=0; snapshot_violations=0
    snap_cache={}
    records=defaultdict(lambda:[0,0])
    pre_records={}
    for d,group in canonical.assign(day=canonical.game_date.astype(str).str[:10]).sort_values('day').groupby('day'):
        for row in group.itertuples():
            pre_records[row.game_pk]=(records[(row.season,row.home_team_abbrev)].copy(),records[(row.season,row.away_team_abbrev)].copy())
        for row in group.itertuples():
            if pd.notna(row.home_score) and pd.notna(row.away_score) and row.home_score!=row.away_score and row.game_type=='R':
                for team,win in [(row.home_team_abbrev,row.home_score>row.away_score),(row.away_team_abbrev,row.away_score>row.home_score)]:
                    records[(row.season,team)][0]+=int(win); records[(row.season,team)][1]+=1
    def tilt(players,lookup):
        score=0.; count=0
        for player in players:
            entry=lookup.get(str(int(player)),{})
            if entry.get('kind')!='batter': continue
            count+=1
            scores={'elite':{'racha caliente':.030,'sólido':.012,'racha fría':-.025},'weak':{'racha caliente':-.050,'sólido':-.015,'racha fría':.035}}
            score+=scores.get(entry.get('talentTier'),{}).get(entry.get('label'),0)
        return score,count
    for i,g in enumerate(games):
        if not g['date'].startswith('2026'): continue
        if g['date'] not in snap_cache:
            path=DATA/'history'/f"{g['date']}.json"
            snap_cache[g['date']]=read(path) if path.exists() else None
        snap=snap_cache[g['date']]
        if not snap:continue
        if snap['cutoffDate']>=g['date']:
            snapshot_violations+=1;continue
        home=starters.get((g['gamePk'],'home'),[]); away=starters.get((g['gamePk'],'away'),[])
        if len(home)<8 or len(away)<8:continue
        rec=pre_records.get(g['gamePk'],([0,0],[0,0]))
        if all(n and w/n>=.580 for w,n in rec):continue
        ht,hc=tilt(home,snap['players']);at,ac=tilt(away,snap['players'])
        if min(hc,ac)<5:continue
        alignment[i]=np.clip(.5+ht-at,.25,.75)
        if abs(alignment[i]-.5)<1e-9:alignment_neutral+=1
    align_non_neutral=np.where(np.abs(alignment-.5)>1e-9,alignment,np.nan)
    other_engines=[{'id':'strikecast_raw','label':'StrikeCast raw exportado','metrics':metrics(sc_raw,y,dates),'status':'Exportación WF; el endpoint web usa pBlend, no pRaw'},
        {'id':'strikecast_blend','label':'StrikeCast combinado exportado','metrics':metrics(sc_blend,y,dates),'status':'Histórico disponible hasta agosto; confirmar paridad con endpoint'},
        {'id':'market','label':'Favorito del mercado sin margen','metrics':metrics(market,y,dates),'status':'Precios históricos de cierre; no prueban disponibilidad intradía'},
        {'id':'value','label':'Value Model','metrics':value_result,'status':'NO OOS: el entrenador incluye todos los juegos con resultado; excluido de selección'},
        {'id':'alignment','label':'Alineación reconstruida 2026','metrics':metrics(alignment,y,dates),'status':'Exploratorio: titulares reconstruidos del boxscore, sin hora de confirmación'},
        {'id':'alignment_non_neutral','label':'Alineación omitiendo empates 50/50','metrics':metrics(align_non_neutral,y,dates),'status':'Misma señal, abstención cuando no hay ventaja; no es mejora a cobertura igual'}]
    # Activation requires full before/after provenance, not merely a positive backtest.
    numerical_gate=(all(before_after[k]['deltaPoints']>=0 and before_after[k]['after']['brier']<=before_after[k]['before']['brier'] for k in ['h1','h2'])
        and before_after['validation2026']['deltaCI95Points'][0]>0)
    for filename in ['games.parquet','player_box.parquet','value_picks.parquet']:
        source_hashes['../data/processed/'+filename]=hashlib.sha256((APP.parent/'data/processed'/filename).read_bytes()).hexdigest()
    for filename in sorted((DATA/'history').glob('2026-*.json')):
        if filename.stem < asof:
            source_hashes['history/'+filename.name]=hashlib.sha256(filename.read_bytes()).hexdigest()
    report={'generatedAt':datetime.now(timezone.utc).isoformat(),'asOf':asof,'dataThrough':str(max(dates)),
        'method':{'selection':'Familia seleccionada con 2024-2025; medición 2026 H1 y H2 sin reselección. Selectores mensuales usan solo meses previos.',
            'limitations':'Auditoría retrospectiva sobre archivos ya explorados. No equivale a una nueva prueba prospectiva. La web usa modelos fijos entrenados 2023-2025; el calendario usa reentrenamiento mensual: no son el mismo motor.',
            'interval':'Bootstrap pareado por día, 2,000 réplicas, semilla 20260916. No corrige la selección previa de features del histórico.'},
        'coverage':{'games':len(games),'excludedOnOrAfterAsOf':len(all_games)-len(games),'canonicalScoreMismatches':len(mismatches),'mismatchExamples':mismatches[:10],
            'strikecastOutcomeMismatches':sc_bad_targets,'snapshotBoundaryViolations':snapshot_violations,'alignmentNeutralVotes':alignment_neutral,
            'currentPlayerCutoff':read(DATA/'player-performance.json')['cutoffDate'],'maskCount':len(masks)},
        'baseline':metrics(base,y,dates),'optimistic':optimistic,'candidates':rows,'selected':{'id':selected,'label':labels[selected],'numericalGate':numerical_gate,
            'activated':False,'reason':'No se cambia el motor activo: faltan predicciones archivadas equivalentes al runtime y verificación de disponibilidad pregame de las fuentes externas.'},
        'beforeAfter':before_after,'probabilityCorrection':probability_correction,'filters':filters,'ablations':ablations,'engines':other_engines,'calibration':cal_rows,'foldSelections':fold_selection,
        'sourceHashes':source_hashes}
    report=clean(report)
    out=APP/'outputs'/'prediction-audit';out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    (DATA/'prediction-audit.json').write_text(json.dumps(report,ensure_ascii=False,separators=(',',':'),allow_nan=False),encoding='utf-8')
    # Per-game output permits independent paired reproduction without retraining.
    frame=pd.DataFrame({'gamePk':[g['gamePk'] for g in games],'date':dates,'homeWin':y,'before':base,'after':candidates[selected],'afterCapped':capped,'eligible':common})
    frame.to_csv(out/'before-after.csv',index=False)
    print(json.dumps({k:report[k] for k in ['coverage','selected','beforeAfter']},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()

"""Exploratory train-before-July decision rules; no live model changes."""
import hashlib
import json
from pathlib import Path

from audit_lineup_signals import APP, DATA, read, pd, np, clean
from audit_prediction_engines import paired


def sample_band(hot_pa):
    if not hot_pa:return 'sin calientes'
    if any(pa<40 for pa in hot_pa):return 'calientes con muestra pequeña'
    if len(hot_pa)>=2 and all(pa>=60 for pa in hot_pa):return 'calientes con muestra amplia'
    return 'muestra intermedia'


def select_action(d, minimum=40):
    if len(d)<minimum:return 'base'
    high=d.win.mean();base=((d.baseP>=.5)==d.win).mean()
    best=max(high,1-high)
    # Fixed gate chosen before validation: minimum 40 and +2 pp training accuracy.
    if best<base+.02:return 'base'
    return 'superior' if high>.5 else 'inferior'


def evaluate(d,picks):
    baseline=np.where(d.baseP.to_numpy()>=.5,.75,.25)
    candidate=np.where(picks,.75,.25)
    r=paired(baseline,candidate,d.win.to_numpy(),d.date.to_numpy(),np.ones(len(d),dtype=bool))
    if not len(d):return r
    # Scores encode side only. Do not report Brier for invented probability levels.
    for side in ['before','after']:
        r[side]={k:r[side][k] for k in ['games','correct','accuracy']}
    return r


def main():
    source=APP/'outputs/lineup-quality/games.csv';f=pd.read_csv(source)
    meta=read(DATA/'lineup-quality-audit.json');hashes={str(source):hashlib.sha256(source.read_bytes()).hexdigest()}
    pb_path=APP.parent/'data/processed/player_box.parquet';hashes[str(pb_path)]=hashlib.sha256(pb_path.read_bytes()).hexdigest()
    pb=pd.read_parquet(pb_path,columns=['game_pk','side','player_id','batting_order','started_batting'])
    pb=pb[pb.started_batting.fillna(False).astype(bool)]
    groups=pb.groupby(['game_pk','side']);cache={};enriched=[]
    for row in f.itertuples():
        if row.date not in cache:
            path=DATA/'history'/f'{row.date}.json';cache[row.date]=read(path);hashes['history/'+row.date]=hashlib.sha256(path.read_bytes()).hexdigest()
        snap=cache[row.date];assert snap['cutoffDate']<row.date
        hot_pa=[];top_ops=[]
        for p in groups.get_group((row.gamePk,'home' if row.home else 'away')).itertuples():
            e=snap['players'].get(str(int(p.player_id)),{})
            if e.get('kind')!='batter':continue
            if e.get('label')=='racha caliente':hot_pa.append(e.get('pa',0))
            if int(p.batting_order)//100<=4 and e.get('ops') is not None:top_ops.append(e['ops'])
        enriched.append({'gamePk':row.gamePk,'home':row.home,'sample':sample_band(hot_pa),'topOps':float(np.mean(top_ops)) if top_ops else np.nan,'topKnown':len(top_ops)})
    enrichment=pd.DataFrame(enriched);f=f.merge(enrichment,on=['gamePk','home'],validate='one_to_one')
    opposite=enrichment.copy();opposite.home=1-opposite.home
    opposite=opposite.rename(columns={k:'opp_'+k for k in ['sample','topOps','topKnown']})
    f=f.merge(opposite,on=['gamePk','home'],validate='one_to_one')
    # Exact primary cohort from the previous study, not reselected from outcomes.
    f=f[(f.qualityGap>=.025)&(f.formGap>=.050)&(f.unknown<=2)&(f.opp_unknown<=2)&f.baseP.notna()].copy()
    assert not f.gamePk.duplicated().any()
    f['top4Signal']=np.select([(f.topKnown<3)|(f.opp_topKnown<3),f.topOps-f.opp_topOps>=.05,f.topOps-f.opp_topOps<=-.05],['datos insuficientes','top4 confirma superior','top4 favorece inferior'],default='top4 parejo')
    f['baseSignal']=np.where(f.baseP>=.5,'base favorece superior','base favorece inferior')
    tr=f[f.date<'2026-07-01'];te=f[f.date>='2026-07-01'].copy()
    policies={'sample':['sample'],'top4':['top4Signal'],'sample_top4':['sample','top4Signal'],
        'sample_base':['sample','baseSignal'],'top4_base':['top4Signal','baseSignal'],'sample_top4_base':['sample','top4Signal','baseSignal']}
    reports={}; details=[]; predictions=te[['gamePk','date','win','baseP','sample','top4Signal','baseSignal']].copy()
    for name,columns in policies.items():
        def key(row):return ' | '.join(str(row[c]) for c in columns)
        trainkeys=tr.apply(key,axis=1);testkeys=te.apply(key,axis=1)
        actions={k:select_action(tr[trainkeys==k]) for k in trainkeys.unique()}
        chosen=np.array([actions.get(k,'base') for k in testkeys])
        picks=np.where(chosen=='superior',True,np.where(chosen=='inferior',False,te.baseP.to_numpy()>=.5))
        comparisons=evaluate(te,picks);comparisons['policyAppliedGames']=int((chosen!='base').sum())
        comparisons['changedPicks']=int((picks!=(te.baseP.to_numpy()>=.5)).sum())
        reports[name]=comparisons;predictions[name]=picks.astype(int)
        for k in sorted(set(trainkeys)|set(testkeys)):
            a=tr[trainkeys==k];b=te[testkeys==k]
            details.append({'policy':name,'cell':k,'actionChosenBeforeJuly':actions.get(k,'base'),
                'discoveryGames':len(a),'discoveryLowerRate':float(1-a.win.mean()) if len(a) else None,
                'validationGames':len(b),'validationLowerRate':float(1-b.win.mean()) if len(b) else None,
                'validationBaseAccuracy':float(((b.baseP>=.5)==b.win).mean()) if len(b) else None})
    hashes['script']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report=clean({'asOf':meta['asOf'],'discoveryGames':len(tr),'validationGames':len(te),'policies':reports,'cells':details,'activated':False,
        'method':'Seis políticas prefijadas. Cada celda aprende superior/inferior solo antes de julio y requiere ≥40 juegos y ventaja de ≥2 puntos sobre base en entrenamiento; si no, conserva base. Todos se evalúan en los mismos juegos desde julio. PA L30 mide tamaño de muestra, no duración ni persistencia de racha. Top4 usa OPS L30 promedio con ≥3 bateadores conocidos por lado.',
        'limitations':'Seguimiento retrospectivo sobre datos ya explorados. No es un holdout nuevo. Intervalos pareados por día para cada política, sin corrección conjunta de seis comparaciones ni búsquedas previas. Se conservan todos los resultados; ninguna política se selecciona para producción. No se midieron splits por mano, bullpen, rentabilidad ni timestamps de anuncio.',
        'sourceHashes':hashes})
    out=APP/'outputs/lineup-decisions';out.mkdir(parents=True,exist_ok=True)
    predictions.to_csv(out/'predictions.csv',index=False)
    for path in [out/'report.json',DATA/'lineup-decision-audit.json']:path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['cells','sourceHashes']},ensure_ascii=True,indent=2))


if __name__=='__main__':main()

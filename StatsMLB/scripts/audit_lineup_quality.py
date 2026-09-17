"""Test higher-quality + better-form lineup versus lower quality, without live changes."""
from pathlib import Path
import hashlib
import json
import math

from audit_lineup_signals import APP, DATA, pd, np, read, clean, wilson, bh
from audit_prediction_engines import paired


def ops(counts):
    ab,h,tb,bb,hbp,sf,pa=counts.T
    return np.divide(h+bb+hbp,ab+bb+hbp+sf,out=np.zeros_like(h),where=(ab+bb+hbp+sf)>0)+np.divide(tb,ab,out=np.zeros_like(tb),where=ab>0)


def attach_prior_counts(starters,daily):
    starters=starters.copy();daily=daily.copy()
    starters['dateKey']=pd.to_datetime(starters.date);daily['dateKey']=pd.to_datetime(daily.date)
    daily['statisticsThrough']=daily.date
    matched=pd.merge_asof(starters.sort_values('dateKey'),daily.drop(columns='date').sort_values('dateKey'),on='dateKey',by='player_id',allow_exact_matches=False,direction='backward')
    assert (matched.statisticsThrough.dropna()<matched.loc[matched.statisticsThrough.notna(),'date']).all()
    return matched


def summarize(d):
    if not len(d):return {'games':0}
    lower=1-d.win.to_numpy(); p=1-d.baseP.to_numpy();valid=np.isfinite(p)
    residual=lower[valid]-p[valid]; dates=d.date.to_numpy()[valid]
    unique=np.unique(dates); sums=np.array([residual[dates==day].sum() for day in unique]);counts=np.array([(dates==day).sum() for day in unique])
    if len(unique):
        rng=np.random.default_rng(20260916); samples=rng.integers(len(unique),size=(2000,len(unique)))
        ci=np.quantile(sums[samples].sum(1)/counts[samples].sum(1),[.025,.975])
        mean=residual.mean();cluster=sums-counts*mean
        se=math.sqrt(float((cluster**2).sum())*len(unique)/max(1,len(unique)-1))/len(residual)
        pv=math.erfc(abs(mean)/se/math.sqrt(2)) if se else 1.
    else:ci=[None,None];pv=1.
    comparison=paired(p,np.full(len(p),.75),lower, d.date.to_numpy(),valid) if valid.any() else None
    return {'games':len(d),'lowerWins':int(lower.sum()),'lowerWinRate':lower.mean(),'lowerCI95':wilson(int(lower.sum()),len(d)),
        'chooseLowerVsBase':comparison,
        'expectedLower':p[valid].mean() if valid.any() else None,'expectedGames':int(valid.sum()),
        'excessLowerPoints':float(residual.mean()*100) if valid.any() else None,'excessCI95':ci,'pExcess':pv,
        'betterRuns':d.runsFor.mean(),'lowerRuns':d.runsAgainst.mean()}


def main():
    root=APP.parent/'data/processed'; source=APP/'outputs/lineup-audit/lineup-games.csv'
    manifest=read(DATA/'lineup-audit.json');cutoff=manifest['asOf']
    f=pd.read_csv(source);games=pd.read_parquet(root/'games.parquet');games['date']=games.game_date.astype(str).str[:10]
    games=games[(games.game_type=='R')&(games.date<cutoff)&games.home_score.notna()&games.away_score.notna()&(games.home_score!=games.away_score)]
    pb=pd.read_parquet(root/'player_box.parquet').merge(games[['game_pk','date','season']],on='game_pk',validate='many_to_one')
    cols=['bat_atBats','bat_hits','bat_totalBases','bat_baseOnBalls','bat_hitByPitch','bat_sacFlies','bat_plateAppearances']
    pb[cols]=pb[cols].fillna(0).astype(float)
    prior=pb[pb.season<2026].groupby('player_id')[cols].sum();league=prior.sum().to_numpy();league=league/league[-1]
    # Previous-years rates shrunk to league with 200 PA; a 200-PA prior stabilizes 2026.
    prior_rates=(prior.to_numpy()+200*league)/(prior.iloc[:,-1].to_numpy()[:,None]+200)
    prior_lookup=dict(zip(prior.index,prior_rates));prior_pa=prior.iloc[:,-1].to_dict()
    season=pb[pb.season==2026]
    daily=season.groupby(['player_id','date'],as_index=False)[cols].sum().sort_values(['player_id','date'])
    daily[cols]=daily.groupby('player_id')[cols].cumsum()
    starters=season[season.started_batting.fillna(False).astype(bool)&season.game_pk.isin(f.gamePk)][['game_pk','date','side','player_id']].copy()
    matched=attach_prior_counts(starters,daily)
    current=matched[cols].fillna(0).to_numpy();historical=np.array([prior_lookup.get(pid,league) for pid in matched.player_id])
    matched['quality']=ops(current+200*historical)
    matched['priorQuality']=ops(historical)
    matched['rated']=np.array([prior_pa.get(pid,0) for pid in matched.player_id])+current[:,-1]>=100
    quality=matched.groupby(['game_pk','side']).agg(quality=('quality','mean'),priorQuality=('priorQuality','mean'),rated=('rated','sum')).reset_index()
    quality['home']=(quality.side=='home').astype(int);quality=quality.rename(columns={'game_pk':'gamePk'}).drop(columns='side')
    f=f.merge(quality,on=['gamePk','home'],validate='one_to_one')
    other=quality.copy();other.home=1-other.home;other=other.rename(columns={k:'opp_'+k for k in ['quality','priorQuality','rated']})
    f=f.merge(other,on=['gamePk','home'],validate='one_to_one');f=f[(f.rated>=7)&(f.opp_rated>=7)].copy()
    f['qualityGap']=f.quality-f.opp_quality;f['priorGap']=f.priorQuality-f.opp_priorQuality
    f['formGap']=f.mean_ops-f.opp_mean_ops
    conditions={
        'Mayor calidad + OPS L30 superior':(f.qualityGap>=.025)&(f.formGap>=.050)&(f.unknown<=2)&(f.opp_unknown<=2),
        'Mayor calidad + al menos 2 calientes adicionales':(f.qualityGap>=.025)&(f.hot-f.opp_hot>=2),
        'Mayor calidad + al menos 2 positivos adicionales':(f.qualityGap>=.025)&(f.positive-f.opp_positive>=2),
        'Mayor calidad + mejor balance calientes menos fríos':(f.qualityGap>=.025)&(f.net_form-f.opp_net_form>=2),
        'Calidad y forma: diferencias grandes':(f.qualityGap>=.050)&(f.formGap>=.100)&(f.unknown<=2)&(f.opp_unknown<=2),
        'Calidad de años anteriores + OPS L30 superior':(f.priorGap>=.025)&(f.formGap>=.050)&(f.unknown<=2)&(f.opp_unknown<=2),
    }
    primary=conditions['Mayor calidad + OPS L30 superior'];splits={
        'Superior local':primary&(f.home==1),'Superior visitante':primary&(f.home==0),
        'Inferior con pitcher dominante':primary&(f.opp_sp_dominant==1),
        'Inferior sin pitcher dominante':primary&(f.opp_sp_dominant==0)&(f.opp_sp_unknown==0),
        'Superior con pitcher en problemas':primary&(f.sp_trouble==1),
        'Superior sin pitcher en problemas':primary&(f.sp_trouble==0)&(f.sp_unknown==0),
        'Superior favorito del modelo':primary&(f.baseP>.5),'Superior no favorito del modelo':primary&(f.baseP<.5)}
    rules=[]
    for name,mask in {**conditions,**splits}.items():
        d=f[mask];assert not d.gamePk.duplicated().any()
        rules.append({'name':name,'discovery':summarize(d[d.date<'2026-07-01']),'validation':summarize(d[d.date>='2026-07-01'])})
    q=bh([r['validation'].get('pExcess',1.) for r in rules])
    for r,v in zip(rules,q):r['validation']['qExcess']=v
    monthly={month:summarize(d) for month,d in f[primary].groupby(f[primary].date.str[:7])}
    report=clean({'asOf':cutoff,'gamesWithQuality':int(f.gamePk.nunique()),'hypotheses':rules,'monthlyPrimary':monthly,'activated':False,
        'method':'Calidad = OPS ofensivo 2026 previo al día, suavizado con 200 PA de historial 2023–2025 (a su vez suavizado hacia liga). OBP incluye HBP/SF. Promedio de los nueve titulares; ambos lados requieren ≥7 jugadores con ≥100 PA previas en total. Forma = etiquetas o OPS L30. Cortes y reglas explícitos, sin seleccionar el mejor después de validar.',
        'limitations':'Exploración posterior al estudio anterior sobre datos ya vistos. Ajuste BH limitado a las 14 hipótesis de este seguimiento, no a todas las búsquedas históricas previas. Titulares reconstruidos de boxscore; sin timestamp de anuncio. No prueba causalidad. OPS L30 y OPS 2026 se superponen; la variante de años anteriores comprueba sensibilidad. Comparación con probabilidades base WF, no con mercado ni picks en vivo.',
        'sources':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [source,root/'games.parquet',root/'player_box.parquet',Path(__file__)]}})
    out=APP/'outputs/lineup-quality';out.mkdir(parents=True,exist_ok=True)
    f.to_csv(out/'games.csv',index=False)
    for path in [out/'report.json',DATA/'lineup-quality-audit.json']:path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True,indent=2))


if __name__=='__main__':main()

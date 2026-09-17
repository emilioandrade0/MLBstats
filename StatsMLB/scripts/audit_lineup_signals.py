"""Lineup research, not a live betting policy. Uses strictly pre-date snapshots.

Discovery: 2026 through June. Validation: July through day before --as-of.
Actual starters/probable pitchers and closing totals lack announcement timestamps;
all results remain retrospective. No production predictions are overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import date
from pathlib import Path

from audit_prediction_engines import APP, DATA, clean, read, pd, np, metrics, paired

LABELS = {'hot':'racha caliente','solid':'sólido','normal':'normal','cold':'racha fría','unknown':'sin etiqueta'}
PITCHERS = {'dominant':'dominante','solid':'sólido','normal':'normal','trouble':'en problemas','unknown':'sin etiqueta'}


def summarize(players, lookup, pitcher_id):
    f = {label:0 for label in LABELS}
    f.update({'top4_'+label:0 for label in LABELS})
    f.update({'elite_'+label:0 for label in LABELS})
    f.update({'weak_'+label:0 for label in LABELS})
    ops, pa = [], []
    f['low_sample'] = 0
    for row in players.itertuples():
        entry = lookup.get(str(int(row.player_id)), {})
        if entry.get('kind') != 'batter': entry = {}
        label = next((k for k,v in LABELS.items() if entry.get('label')==v), 'unknown')
        f[label] += 1
        if int(row.batting_order)//100 <= 4: f['top4_'+label] += 1
        if entry.get('talentTier')=='elite': f['elite_'+label] += 1
        if entry.get('talentTier')=='weak': f['weak_'+label] += 1
        if entry.get('pa',0)<40: f['low_sample'] += 1
        if entry.get('ops') is not None: ops.append(entry['ops'])
        if entry.get('pa') is not None: pa.append(entry['pa'])
    f['positive'] = f['hot']+f['solid']
    f['net_form'] = f['hot']-f['cold']
    f['mean_ops'] = float(np.mean(ops)) if ops else .7
    f['mean_pa'] = float(np.mean(pa)) if pa else 0.
    entry = lookup.get(str(int(pitcher_id)),{}) if pd.notna(pitcher_id) else {}
    if entry.get('kind') != 'pitcher': entry = {}
    label = next((k for k,v in PITCHERS.items() if entry.get('label')==v), 'unknown')
    f.update({'sp_'+k:int(label==k) for k in PITCHERS})
    return f


def wilson(w,n):
    if not n: return [None,None]
    p=w/n; z=1.96; den=1+z*z/n
    c=(p+z*z/(2*n))/den; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [c-h,c+h]


def score_rule(frame, mask):
    d=frame.loc[mask].copy()
    # Each game contributes once to inference: ambiguous rules matching both teams abstain.
    d=d[~d.gamePk.duplicated(keep=False)]
    if not len(d): return {'n':0}
    n=len(d); w=int(d.win.sum()); valid=d.baseP.notna(); b=d.loc[valid]
    baseline_correct=np.where(b.baseP>=.5,b.win,1-b.win).astype(float)
    diff=b.win.to_numpy(dtype=float)-baseline_correct
    # Cluster by day to account for shared conditions and paired residuals.
    def cluster_p(values, dates):
        t=pd.DataFrame({'v':values,'date':dates})
        if len(t)<2:return 1.
        mean=t.v.mean(); groups=t.assign(res=t.v-mean).groupby('date').res.sum()
        if len(groups)<2:return 1.
        se=math.sqrt(float((groups**2).sum())*len(groups)/(len(groups)-1))/len(t)
        return math.erfc(abs(mean)/se/math.sqrt(2)) if se else (0. if mean else 1.)
    total=d.homeRuns+d.awayRuns
    market=d[d.totalLine.notna()]; resolved=market[(market.homeRuns+market.awayRuns)!=market.totalLine]
    over=((resolved.homeRuns+resolved.awayRuns)>resolved.totalLine).astype(float)
    return {'n':n,'wins':w,'winRate':w/n,'winCI95':wilson(w,n),'runsFor':d.runsFor.mean(),'runsAgainst':d.runsAgainst.mean(),
        'totalRuns':total.mean(),'over7_5':(total>7.5).mean(),'over8_5':(total>8.5).mean(),'over9_5':(total>9.5).mean(),
        'marketGames':len(market),'marketResolved':len(resolved),'pushes':len(market)-len(resolved),
        'marketOver':((resolved.homeRuns+resolved.awayRuns)>resolved.totalLine).mean() if len(resolved) else None,
        'marketOverCI95':wilson(int(over.sum()),len(over)),
        'pTotalVs50':cluster_p(over.to_numpy()-.5,resolved.date.to_numpy()) if len(over) else 1.,
        'baseGames':len(b),'baseAccuracy':baseline_correct.mean() if len(b) else None,
        'deltaVsBasePoints':float(diff.mean()*100) if len(b) else None,
        'pVsBase':cluster_p(diff,b.date.to_numpy()) if len(b) else 1.}


def bh(values):
    p=np.array(values); order=np.argsort(p); q=np.empty(len(p)); last=1.
    for rank in range(len(p),0,-1):
        idx=order[rank-1]; last=min(last,p[idx]*len(p)/rank);q[idx]=last
    return q


def fit_predict(x, y, test, offset=None, test_offset=None, logistic=True, penalty=100.):
    mean=x.mean(0); scale=x.std(0);scale[scale<1e-8]=1
    x=np.column_stack([np.ones(len(x)),(x-mean)/scale]);test=np.column_stack([np.ones(len(test)),(test-mean)/scale])
    reg=np.eye(x.shape[1])*penalty;reg[0,0]=1e-6
    offset=np.zeros(len(x)) if offset is None else offset
    test_offset=np.zeros(len(test)) if test_offset is None else test_offset
    if not logistic:
        w=np.linalg.solve(x.T@x+reg,x.T@(y-offset));return test@w+test_offset
    w=np.zeros(x.shape[1])
    for _ in range(40):
        p=1/(1+np.exp(-np.clip(x@w+offset,-25,25)))
        step=np.linalg.solve(x.T@((p*(1-p))[:,None]*x)+reg,x.T@(p-y)+reg@w)
        w-=step
        if np.max(np.abs(step))<1e-7:break
    return 1/(1+np.exp(-np.clip(test@w+test_offset,-25,25)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--as-of',default=date.today().isoformat());args=parser.parse_args()
    root=APP.parent/'data/processed'; hashes={}
    def parquet(name):
        path=root/(name+'.parquet');hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest();return pd.read_parquet(path)
    games=parquet('games');games['date']=games.game_date.astype(str).str[:10]
    games=games[(games.season==2026)&(games.game_type=='R')&(games.date<args.as_of)&games.home_score.notna()&games.away_score.notna()&(games.home_score!=games.away_score)]
    pre_records={}; records={}
    for day,group in games.sort_values('date').groupby('date'):
        for g in group.itertuples():
            pre_records[g.game_pk]={side:(records.get(getattr(g,side+'_team_abbrev'),[0,0])[0]/max(1,records.get(getattr(g,side+'_team_abbrev'),[0,0])[1])) for side in ['home','away']}
        for g in group.itertuples():
            for side,other in [('home','away'),('away','home')]:
                rec=records.setdefault(getattr(g,side+'_team_abbrev'),[0,0]);rec[0]+=int(getattr(g,side+'_score')>getattr(g,other+'_score'));rec[1]+=1
    pb=parquet('player_box'); starters=pb[pb.started_batting.fillna(False).astype(bool)].groupby(['game_pk','side'])
    odds=parquet('odds_close');xref=parquet('games_xref').dropna(subset=['espn_event_id','game_pk'])
    # Exclude live providers. Require one-to-one event mapping. Median pregame
    # closing lines across providers is a research reference, not an executable price.
    xref=xref[['espn_event_id','game_pk']].drop_duplicates()
    xref=xref[~xref.espn_event_id.duplicated(keep=False)&~xref.game_pk.duplicated(keep=False)]
    odds=odds[~odds.provider_name.str.contains('Live',case=False,na=False)&odds.total_close.notna()].copy()
    odds.total_close=pd.to_numeric(odds.total_close,errors='coerce')
    lines=odds.groupby('espn_event_id').total_close.median().reset_index().merge(xref,on='espn_event_id').set_index('game_pk').total_close.to_dict()
    baseline_path=APP/'outputs/prediction-audit/before-after.csv'
    hashes['baselineCSV']=hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    hashes['script']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    baseline=pd.read_csv(baseline_path).set_index('gamePk').before.to_dict()
    rows=[];excluded={'snapshot':0,'lineup':0};snapshots={}
    for g in games.sort_values(['date','game_pk']).itertuples():
        if g.date not in snapshots:
            path=DATA/'history'/f'{g.date}.json'
            snapshots[g.date]=read(path) if path.exists() else None
            if path.exists():hashes['history/'+g.date]=hashlib.sha256(path.read_bytes()).hexdigest()
        snap=snapshots[g.date]
        if not snap or snap['cutoffDate']>=g.date or snap.get('date')!=g.date:
            excluded['snapshot']+=1;continue
        sides={}
        for side in ['home','away']:
            try: lineup=starters.get_group((g.game_pk,side))
            except KeyError:break
            if len(lineup)!=9 or lineup.player_id.nunique()!=9 or set(lineup.batting_order.astype(int)//100)!=set(range(1,10)):break
            sides[side]=summarize(lineup,snap['players'],getattr(g,'probable_'+side+'_pitcher_id'))
        if len(sides)!=2:excluded['lineup']+=1;continue
        for side,other in [('home','away'),('away','home')]:
            p=baseline.get(g.game_pk,np.nan)
            row={'gamePk':g.game_pk,'date':g.date,'snapshotCutoff':snap['cutoffDate'],'home':int(side=='home'),'team':getattr(g,side+'_team_abbrev'),'opponent':getattr(g,other+'_team_abbrev'),
                'win':int(getattr(g,side+'_score')>getattr(g,other+'_score')),'runsFor':getattr(g,side+'_score'),'runsAgainst':getattr(g,other+'_score'),
                'homeRuns':g.home_score,'awayRuns':g.away_score,'totalLine':lines.get(g.game_pk,np.nan),'baseP':p if side=='home' else 1-p}
            row.update(sides[side]);row.update({'opp_'+k:v for k,v in sides[other].items()});rows.append(row)
            def tilt(s):return s['elite_hot']*.030+s['elite_solid']*.012-s['elite_cold']*.025-s['weak_hot']*.050-s['weak_solid']*.015+s['weak_cold']*.035
            old_p=float(np.clip(.5+tilt(sides[side])-tilt(sides[other]),.25,.75))
            permitted=not all(v>=.580 for v in pre_records[g.game_pk].values()) and sides[side]['unknown']<=4 and sides[other]['unknown']<=4
            row['oldAlignmentP']=old_p if permitted and abs(old_p-.5)>1e-9 else np.nan
    f=pd.DataFrame(rows);assert not f[['gamePk','home']].duplicated().any()
    train=f.date<'2026-07-01';valid=~train; rules=[]
    def rule(name, mask):rules.append((name,np.asarray(mask,dtype=bool)))
    for label in [*LABELS,'positive']:
        for n in range(1,6):
            rule(f'{label} >= {n}',f[label]>=n)
            rule(f'Ventaja {label} >= {n}',f[label]-f['opp_'+label]>=n)
        for n in range(1,4):rule(f'Top4 {label} >= {n}',f['top4_'+label]>=n) if label in LABELS else None
    for own in LABELS:
        for opp in LABELS:
            for a in [2,3,4]:
                for b in [2,3,4]:rule(f'{own} >= {a} vs {opp} >= {b}',(f[own]>=a)&(f['opp_'+opp]>=b))
    for label in LABELS:
        for n in [2,3,4]:
            for sp in PITCHERS:rule(f'{label} >= {n} vs pitcher {sp}',(f[label]>=n)&(f['opp_sp_'+sp]==1))
    for own in PITCHERS:
        for opp in PITCHERS:rule(f'Pitcher {own} vs {opp}',(f['sp_'+own]==1)&(f['opp_sp_'+opp]==1))
    for n in [1,2,3]:
        for label in LABELS:rule(f'Elite {label} >= {n}',f['elite_'+label]>=n)
        rule(f'Calientes >= {n}, sin fríos',(f.hot>=n)&(f.cold==0))
        rule(f'Calientes >= {n}, cobertura completa',(f.hot>=n)&(f.unknown==0)&(f.opp_unknown==0))
    for n in [2,3,4]:
        rule(f'Ambos hot >= {n}',(f.hot>=n)&(f.opp_hot>=n)&(f.home==1))
        rule(f'Ambos cold >= {n}',(f.cold>=n)&(f.opp_cold>=n)&(f.home==1))
        rule(f'Ambos positive >= {n}',(f.positive>=n)&(f.opp_positive>=n)&(f.home==1))
    discovery=[]; low_sample_rules=[]
    for name,mask in rules:
        s=score_rule(f,mask&train)
        if s['n']>=100:discovery.append({'name':name,'discovery':s,'mask':mask})
        else:low_sample_rules.append({'name':name,'discoveryGames':s['n']})
    # Select on discovery only, before consulting validation. Separate winner and
    # totals shortlists; BH controls tests across the entire retained validation family.
    win_top=sorted([r for r in discovery if not r['name'].startswith('Ambos')],key=lambda r:-r['discovery']['deltaVsBasePoints'])[:20]
    total_top=sorted([r for r in discovery if r['discovery']['marketResolved']>=80],key=lambda r:-abs(r['discovery']['marketOver']-.5))[:20]
    selected={r['name'] for r in win_top+total_top}
    records=[]
    for r in discovery:
        v=score_rule(f,r['mask']&valid)
        records.append({'name':r['name'],'discovery':r['discovery'],'validation':v,'selectedBeforeValidation':r['name'] in selected,
            'selectedWinner':r['name'] in {x['name'] for x in win_top},'selectedTotals':r['name'] in {x['name'] for x in total_top}})
    q=bh([r['validation'].get('pVsBase',1.) for r in records])
    for r,value in zip(records,q):r['validation']['qVsBaseAllRules']=value
    qt=bh([r['validation'].get('pTotalVs50',1.) for r in records])
    for r,value in zip(records,qt):
        r['validation']['qTotalVs50AllRules']=value
        r['totalDirection']='over' if (r['discovery'].get('marketOver') or 0)>=.5 else 'under'
    # Few fixed model families, not a further search tuned on the validation set.
    h=f[(f.home==1)&f.baseP.notna()].copy();tr=h.date<'2026-07-01';te=~tr
    keys=[*LABELS,'positive','net_form','low_sample','mean_ops','mean_pa',*['top4_'+k for k in LABELS],*['elite_'+k for k in LABELS],*['sp_'+k for k in PITCHERS]]
    diff=np.column_stack([h[k]-h['opp_'+k] for k in keys]);sums=np.column_stack([h[k]+h['opp_'+k] for k in keys])
    interactions=np.column_stack([h[k]*h['opp_sp_'+sp]-h['opp_'+k]*h['sp_'+sp] for k in ['hot','cold','positive','unknown'] for sp in ['dominant','trouble','unknown']])
    x=np.column_stack([diff,interactions]);y=h.win.to_numpy();base=h.baseP.to_numpy();logit=np.log(np.clip(base,.001,.999)/(1-np.clip(base,.001,.999)))
    candidates={'base':base[te], 'lineup':fit_predict(x[tr],y[tr],x[te]),'base_plus_lineup':fit_predict(x[tr],y[tr],x[te],logit[tr],logit[te])}
    model_rows={name:paired(base[te],p,y[te],h.date.to_numpy()[te],np.ones(te.sum(),dtype=bool)) for name,p in candidates.items()}
    old=h.oldAlignmentP.to_numpy()[te]
    old_comparison={name:paired(old,p,y[te],h.date.to_numpy()[te],np.isfinite(old)) for name,p in candidates.items()}
    total=h.homeRuns.to_numpy()+h.awayRuns.to_numpy();tx=np.column_stack([sums, np.abs(diff)])
    total_pred=fit_predict(tx[tr],total[tr],tx[te],logistic=False)
    total_eval={'games':int(te.sum()),'baselineMean':float(total[tr].mean()),'baselineMAE':float(np.abs(total[te]-total[tr].mean()).mean()),
        'lineupMAE':float(np.abs(total[te]-total_pred).mean()),'baselineRMSE':float(np.sqrt(np.mean((total[te]-total[tr].mean())**2))),
        'lineupRMSE':float(np.sqrt(np.mean((total[te]-total_pred)**2)))}
    market_train=tr&h.totalLine.notna();market_test=te&h.totalLine.notna()
    adjusted_market_prediction=np.full(int(te.sum()),np.nan)
    if market_train.sum()>=100 and market_test.sum():
        pred=fit_predict(tx[market_train],total[market_train],tx[market_test],h.totalLine.to_numpy()[market_train],h.totalLine.to_numpy()[market_test],logistic=False)
        adjusted_market_prediction[h.totalLine.notna().to_numpy()[te]]=pred
        line=h.totalLine.to_numpy()[market_test];actual=total[market_test];resolved=actual!=line
        total_eval['market']={'games':int(market_test.sum()),'resolved':int(resolved.sum()),'pushes':int((~resolved).sum()),'lineMAE':float(np.abs(actual-line).mean()),
            'adjustedMAE':float(np.abs(actual-pred).mean()),'overRate':float((actual[resolved]>line[resolved]).mean()),
            'directionAccuracy':float(((pred[resolved]>line[resolved])==(actual[resolved]>line[resolved])).mean())}
        train_actual=total[market_train];train_lines=h.totalLine.to_numpy()[market_train];rt=train_actual!=train_lines
        baseline_over=bool((train_actual[rt]>train_lines[rt]).mean()>=.5)
        comparison=paired(np.full(int(resolved.sum()),.75 if baseline_over else .25),np.where(pred[resolved]>line[resolved],.75,.25),
            (actual[resolved]>line[resolved]).astype(int),h.date.to_numpy()[market_test][resolved],np.ones(int(resolved.sum()),dtype=bool))
        total_eval['market']['baselineDirection']='over' if baseline_over else 'under'
        total_eval['market']['baselineDirectionAccuracy']=comparison['before']['accuracy']
        total_eval['market']['alwaysUnderDescriptive']=float((actual[resolved]<line[resolved]).mean())
        total_eval['market']['deltaCI95Points']=comparison['deltaCI95Points']
        total_eval['market']['netCorrect']=comparison['netCorrect']
    out=APP/'outputs/lineup-audit';out.mkdir(parents=True,exist_ok=True)
    f.to_csv(out/'lineup-games.csv',index=False)
    pd.DataFrame({'gamePk':h.gamePk[te],'date':h.date[te],'homeWin':y[te],**candidates,'oldAlignment':old,'totalRuns':total[te],'predictedTotal':total_pred,
        'totalLine':h.totalLine[te],'adjustedMarketTotal':adjusted_market_prediction}).to_csv(out/'validation-predictions.csv',index=False)
    report=clean({'asOf':args.as_of,'split':'2026-07-01','games':len(f)//2,'discoveryGames':int(train.sum()/2),'validationGames':int(valid.sum()/2),
        'excluded':excluded,'rulesTested':len(rules),'rulesWith100DiscoveryGames':len(records),'shortlistSize':len(selected),
        'method':'Etiquetas L30 estrictamente anteriores al día; titulares reales reconstruidos de boxscore; pitcher probable sin timestamp. Exploratorio, no prueba prospectiva. Reglas ambiguas se abstienen; ambos lados en totales se cuentan una sola vez. BH aplicado a toda la familia elegible para victoria.',
        'totalsMethod':'Mediana de total_close de proveedores sin Live; sin hora verificable ni precio de ejecución. Push excluido de porcentajes; sin ROI. Cortes 7.5/8.5/9.5 descriptivos, no líneas reales.',
        'models':model_rows,'versusOldAlignmentSameGames':old_comparison,'totals':total_eval,'rules':records,'lowSampleRules':low_sample_rules,'sourceHashes':hashes,'activated':False})
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    (DATA/'lineup-audit.json').write_text(json.dumps(report,ensure_ascii=False,separators=(',',':'),allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['rules','sourceHashes','lowSampleRules','models','versusOldAlignmentSameGames']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()

"""Export existing NFL artifacts to the shared web. Does not train or send messages."""
import ast
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

APP=Path(__file__).resolve().parents[1]
_audit=APP/'outputs/audit-runtime'
if _audit.is_dir():
    # Append (not insert) para no shadowear pyarrow del site-packages.
    sys.path.append(str(_audit))
import pandas as pd


def records(frame):
    return json.loads(frame.to_json(orient='records',date_format='iso'))


def main():
    root=APP/'StrikeCast NFL';data=root/'data/processed';out=APP/'public/data/nfl';out.mkdir(parents=True,exist_ok=True)
    games=pd.read_parquet(data/'games.parquet')
    assert games.game_id.is_unique
    cols=['game_id','model_home_win_prob','model_cover_home_prob','model_total_over_prob']
    path=data/'walkforward_preds.parquet'
    if path.exists():
        predictions=pd.read_parquet(path)[cols];assert predictions.game_id.is_unique
        games=games.merge(predictions,on='game_id',how='left',validate='one_to_one')
    else:
        for c in cols[1:]:games[c]=None
    # Multi-motor: solo agregamos consenso al catalogo si su OOS supera al mejor individual.
    multi_meta={};multi_used=False
    if (data/'nfl_multi_meta.json').exists():
        multi_meta=json.loads((data/'nfl_multi_meta.json').read_text(encoding='utf-8'));multi_used=multi_meta.get('consensusUsed',False)
    if (data/'nfl_multi_preds.parquet').exists():
        multi=pd.read_parquet(data/'nfl_multi_preds.parquet');assert multi.game_id.is_unique
        motor_cols=[c for c in multi.columns if c.endswith('_home_win')]
        if not multi_used:multi=multi.drop(columns=['consensus_home_win'],errors='ignore');motor_cols=[c for c in motor_cols if c!='consensus_home_win']
        games=games.merge(multi[['game_id']+motor_cols],on='game_id',how='left',validate='one_to_one')
    filters={};catalog={};extras={}
    if (data/'game_filters.parquet').exists():
        gf=pd.read_parquet(data/'game_filters.parquet')
        for row in records(gf):filters[row['game_id']]=[key for key,value in row.items() if value is True]
    syntax=ast.parse((root/'packages/ml/strikecast_nfl/features/filters.py').read_text(encoding='utf-8'))
    for node in syntax.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FILTER_CATALOG' for t in node.targets):catalog=ast.literal_eval(node.value)
    if (data/'lvd_preds.parquet').exists():
        lvd=pd.read_parquet(data/'lvd_preds.parquet');assert lvd.game_id.is_unique
        extras={r['game_id']:r for r in records(lvd)}
    seasons=[]
    for season,frame in games.groupby('season'):
        rows=records(frame.sort_values(['kickoff_utc','game_id']))
        for row in rows:row['filters']=filters.get(row['game_id'],[]);row['lvd']=extras.get(row['game_id'])
        valid=frame[(frame.status=='final')&frame.home_score.notna()&frame.away_score.notna()&frame.model_home_win_prob.notna()&(frame.home_score!=frame.away_score)]
        metric={'season':int(season),'games':len(frame),'evaluated':len(valid),'accuracy':None,'brier':None}
        if len(valid):
            y=(valid.home_score>valid.away_score).astype(int);p=valid.model_home_win_prob
            metric.update(accuracy=float(((p>=.5)==y).mean()),brier=float(((p-y)**2).mean()))
        seasons.append(metric)
        (out/f'{int(season)}.json').write_text(json.dumps({'games':rows},ensure_ascii=False,allow_nan=False),encoding='utf-8')
    reports={}
    for name in ['roi_summary','roi_by_season','pattern_robust','pattern_leaderboard','pattern_oos']:
        path=data/(name+'.parquet')
        if path.exists():reports[name]=records(pd.read_parquet(path))
    winner_models=[{'id':'strikecast','label':'StrikeCast NFL','target':'winner','field':'model_home_win_prob'}]
    if (data/'nfl_multi_preds.parquet').exists():
        motor_labels={'m_full':'Motor Completo','m_market':'Motor Mercado','m_elo':'Motor ELO','m_epa':'Motor EPA'}
        for mid,label in motor_labels.items():
            winner_models.append({'id':mid,'label':label,'target':'winner','field':f'{mid}_home_win'})
        if multi_used:
            winner_models.append({'id':'consensus','label':'Consenso (promedio)','target':'winner','field':'consensus_home_win'})
    manifest={'generatedAt':datetime.now(timezone.utc).isoformat(),'source':'StrikeCast NFL · snapshot de archivos locales',
        'seasons':sorted(seasons,key=lambda s:-s['season']),'filters':catalog,'reports':reports,
        'models':winner_models,'multiMeta':multi_meta}
    (out/'index.json').write_text(json.dumps(manifest,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(f'Exported {len(games)} NFL games, {len(seasons)} seasons. No models retrained.')


if __name__=='__main__':main()

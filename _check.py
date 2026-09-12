"""A/B test: comparar pipelines con y sin overrides retirados.

Config A: pipeline actual (post-retiros + re-tunes)
Config B: A + burn_resilience restaurado
Config C: A + burn_resilience + umpire_market restaurados
Config D: Codex original (todos activos, catcher/circadian en baseline)
"""
import pandas as pd, numpy as np, pickle, sys, importlib
sys.path.insert(0, '.')
if 'src.serve.api' in sys.modules:
    importlib.reload(sys.modules['src.serve.api'])
from src.serve.api import (
    _winner_threshold, _flip_reason,
    _h2h_bias, _pitcher_bias,
    _home_cold_streak_bias, _away_hot_streak_bias,
    _home_blowout_momentum_bias, _away_cold_streak_bias,
    _weather_extreme_bias, _home_team_calibration_bias,
    _home_band_calibration_bias,
    _interleague_nl_bias, _umpire_market_bias,
    _travel_resilience_bias,
    _pythag_luck_bias, _babip_persistence_bias,
    _lob_persistence_bias, _cws_night_bias,
    _comeback_deficit_regression_bias,
    _runs_median_persistence_bias, _burn_resilience_bias,
    _starter_workload_regression_bias, _highlev_fatigue_bias,
    _catcher_battery_regime_bias, _circadian_extreme_bias,
    CATCHER_BATTERY_REGIME_BIAS, CATCHER_BATTERY_REGIME_THRESHOLD,
    CIRCADIAN_HOME_BIAS,
    _h2h_venue_regression_bias, _series_double_down_market,
)

with open('data/models/lgb_cls.pkl','rb') as f:
    b = pickle.load(f)
model = b['model']

tr = pd.read_parquet('data/processed/train.parquet')
tr = tr[tr['home_score'].notna() & tr['away_score'].notna()].copy()
tr['game_date'] = pd.to_datetime(tr['game_date'])
tr['home_won'] = (tr['home_score'] > tr['away_score']).astype(int)
tr['home_win'] = tr['home_won']
tr['season'] = tr['game_date'].dt.year
tr['p_home'] = model.predict_proba(tr[b['feature_names']])[:,1]

# Merges necesarios
gm = pd.read_parquet('data/processed/games.parquet')[['game_pk','day_night']]
tr = tr.merge(gm, on='game_pk', how='left')
tr['day_night'] = tr['day_night'].str.lower()

tv = pd.read_parquet('data/processed/features_travel.parquet')[['game_pk','side','travel_dist_miles']]
tv_wide = tv.pivot(index='game_pk', columns='side', values='travel_dist_miles').reset_index()
tv_wide.columns = ['game_pk','travel_dist_miles_a','travel_dist_miles_h']
tr = tr.merge(tv_wide, on='game_pk', how='left')

for filename, val, alias_a, alias_h in [
    ('features_circadian.parquet', 'circ_x_day_home', None, None),
    ('features_luck.parquet', 'babip_luck_net', 'babip_luck_net_a', 'babip_luck_net_h'),
]:
    try:
        if val == 'circ_x_day_home':
            tr = tr.merge(pd.read_parquet(f'data/processed/{filename}', columns=['game_pk', val]), on='game_pk', how='left')
        else:
            luck = pd.read_parquet(f'data/processed/{filename}')[['game_pk','side',val]]
            luck_wide = luck.pivot(index='game_pk', columns='side', values=val).reset_index()
            luck_wide.columns = ['game_pk', alias_a, alias_h]
            tr = tr.merge(luck_wide, on='game_pk', how='left')
    except Exception as e:
        print(f"warn: {filename}: {e}")

try:
    cluster = pd.read_parquet('data/processed/features_cluster_luck.parquet')[['game_pk','team','lob_off_dev','lob_def_dev']]
    hc = cluster.rename(columns={'team':'home_team_abbrev','lob_off_dev':'lob_off_dev_h','lob_def_dev':'lob_def_dev_h'})
    ac = cluster.rename(columns={'team':'away_team_abbrev','lob_off_dev':'lob_off_dev_a','lob_def_dev':'lob_def_dev_a'})
    tr = tr.merge(hc, on=['game_pk','home_team_abbrev'], how='left')
    tr = tr.merge(ac, on=['game_pk','away_team_abbrev'], how='left')
except Exception as e:
    print(f"warn cluster: {e}")

try:
    cb = pd.read_parquet('data/processed/features_catcher_control.parquet', columns=['game_pk','side','battery_kbb_l8'])
    cb_wide = cb.pivot(index='game_pk', columns='side', values='battery_kbb_l8').reset_index()
    cb_wide = cb_wide.rename(columns={'away':'battery_kbb_l8_a','home':'battery_kbb_l8_h'})
    tr = tr.merge(cb_wide, on='game_pk', how='left')
except Exception as e:
    print(f"warn cb: {e}")

for filename, val in [
    ('features_comeback.parquet','cb_avg_def_l30'),
    ('features_game_flow.parquet','runs_median_l20'),
    ('features_burn.parquet','burn_score'),
]:
    try:
        s = pd.read_parquet(f'data/processed/{filename}')[['game_pk','side',val]]
        sw = s.pivot(index='game_pk', columns='side', values=val).reset_index()
        sw = sw.rename(columns={'away':f'{val}_a','home':f'{val}_h'})
        tr = tr.merge(sw, on='game_pk', how='left')
    except Exception as e:
        print(f"warn {filename}: {e}")

for filename in ['features_starter_workload.parquet','features_high_leverage.parquet']:
    try:
        tr = tr.merge(pd.read_parquet(f'data/processed/{filename}'), on='game_pk', how='left')
    except Exception as e:
        print(f"warn {filename}: {e}")

c = {'train': tr}

# Baseline catcher/circadian (Codex original)
def catcher_orig(r, p):
    try:
        if int(r.get('season')) != 2026 or not 0.50 <= float(p) < 0.60:
            return 0.0
        adv = float(r.get('battery_kbb_l8_h')) - float(r.get('battery_kbb_l8_a'))
        if abs(adv) >= CATCHER_BATTERY_REGIME_THRESHOLD:
            return -float(np.sign(adv)) * CATCHER_BATTERY_REGIME_BIAS
    except Exception:
        pass
    return 0.0

def circadian_orig(r, p):
    try:
        if int(r.get('season')) >= 2025 and 0.40 <= float(p) < 0.50 and float(r.get('circ_x_day_home')) >= 2:
            return CIRCADIAN_HOME_BIAS
    except Exception:
        pass
    return 0.0

def make_pick(r, config):
    p = r['p_home']
    if not np.isfinite(p): return None
    ht = r['home_team_abbrev']
    at = r['away_team_abbrev']
    mp = r.get('market_p_home', float('nan'))
    season = int(r['season'])
    gd = r['game_date']
    dn = r.get('day_night')
    hpid = r.get('probable_home_pitcher_id')
    apid = r.get('probable_away_pitcher_id')
    try:
        bias = (
            _h2h_bias(ht, at, season)
            + _h2h_venue_regression_bias(r, p)
            + _pitcher_bias(hpid, apid, season)
            + _home_cold_streak_bias(c, r)
            + _away_hot_streak_bias(c, r)
            + _home_blowout_momentum_bias(c, r)
            + _away_cold_streak_bias(c, r)
            + _home_team_calibration_bias(r, p)
            + _home_band_calibration_bias(r, p)
            + _interleague_nl_bias(r, p)
            + _travel_resilience_bias(r)
            + _pythag_luck_bias(r, p)
            + _babip_persistence_bias(r, p)
            + _lob_persistence_bias(r, p)
            + _cws_night_bias(r, p)
            + _comeback_deficit_regression_bias(r, p)
            + _runs_median_persistence_bias(r, p)
            + _starter_workload_regression_bias(r, p)
            + _highlev_fatigue_bias(r, p)
        )
        # A: usa las funciones actuales (retuneadas)
        if config in ('A','B','C'):
            bias += _catcher_battery_regime_bias(r, p)  # ya re-tuneada
            bias += _circadian_extreme_bias(r, p)  # ya re-tuneada
        # D: baseline Codex (no retuneado)
        else:
            bias += catcher_orig(r, p)
            bias += circadian_orig(r, p)
        # Restaurar burn_resilience si B/C/D
        if config in ('B','C','D'):
            bias += _burn_resilience_bias(r, p)
        # Restaurar umpire_market si C/D
        if config in ('C','D'):
            bias += _umpire_market_bias(r)
        # Restaurar weather_extreme solo D
        if config == 'D':
            bias += _weather_extreme_bias(r)
    except Exception:
        bias = 0.0
    p_adj = float(np.clip(p + bias, 0.001, 0.999)) if bias else p
    raw = bool(p_adj >= _winner_threshold(mp, p_adj, ht, at, season))
    reason = _flip_reason(raw, ht, at, gd, dn, p, mp)
    pick = (not raw) if reason else raw
    # Series DD
    try:
        if _series_double_down_market(c, r, pick):
            if pd.notna(mp):
                pick = bool(float(mp) > 0.5)
    except Exception:
        pass
    return pick

# Solo 2026 para velocidad
sub_26 = tr[tr['season']==2026].copy()
print(f"N 2026: {len(sub_26)}")

for cfg in ['A','B','C','D']:
    sub_26[f'pick_{cfg}'] = sub_26.apply(lambda r: make_pick(r, cfg), axis=1)
    sub_26[f'hit_{cfg}'] = (sub_26[f'pick_{cfg}'].astype('boolean') == sub_26['home_won'].astype('boolean')).astype('Int64')

print("\n═══ Comparativa configs (in-sample 2026) ═══")
for cfg, label in [
    ('A', 'ACTUAL (retirados + retuneados)'),
    ('B', 'A + burn_resilience restaurado'),
    ('C', 'B + umpire_market restaurado'),
    ('D', 'CODEX original (todo + catcher/circadian baseline)'),
]:
    hits = sub_26[f'hit_{cfg}'].sum()
    n = sub_26[f'hit_{cfg}'].notna().sum()
    acc = hits/n if n else 0
    print(f"  {cfg}: {label}")
    print(f"     n={n}  hits={hits}  acc={acc*100:.2f}%")

# Differences
print("\n═══ Diferencias vs config A ═══")
base = sub_26['hit_A'].sum()
for cfg in ['B','C','D']:
    delta = sub_26[f'hit_{cfg}'].sum() - base
    n = sub_26[f'hit_{cfg}'].notna().sum()
    print(f"  {cfg} vs A: {delta:+d} hits ({delta/n*100:+.2f}pp)")

"""Independent order arithmetic and episode-budget audits from saved ledgers."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parent
checks=[];stress=[]
selected=json.loads((OUT/'selected_before_tests.json').read_text())
for period in ['train_2016_2019','test_2020_2023','test_2024_2026']:
 dca=pd.read_csv(OUT/f'{period}_DCA_daily.csv',parse_dates=['date'])
 trough=dca.loc[(dca.twr/dca.twr.cummax().clip(lower=1)-1).idxmin(),'date']
 for c in selected:
  prefix=OUT/f'{period}_{c["id"]}'
  o=pd.read_csv(str(prefix)+'_orders.csv',parse_dates=['signal_date'])
  t=pd.read_csv(str(prefix)+'_trades.csv',parse_dates=['date','signal_date'])
  d=pd.read_csv(str(prefix)+'_daily.csv',parse_dates=['date'])
  if c['family']=='depth':expected=o.cash*c['a']*o.depth/(c['h']+o.depth)
  elif c['family']=='percentile':expected=o.cash*c['a']*(1-o.percentile/c['p0']).clip(lower=0)**c['r']
  else:
   expected=np.minimum(o.cash,(o.K*o.depth/(c['h']+o.depth)-o.D).clip(lower=0))
   e=pd.read_csv(str(prefix)+'_episodes.csv',parse_dates=['start','end'])
   for ep in e.itertuples():
    ds=d[d.episode==ep.id]
    assert ds.episode_K.nunique()==1
    assert abs(t.loc[t.episode==ep.id,'amount'].sum()-ep.D)<1e-6
    assert ep.D<=ep.K+1e-6
    # New deposits do not enlarge the episode budget.
    assert ds.episode_D.diff().dropna().ge(-1e-8).all()
  error=float(np.max(abs(expected-o.amount)));assert error<1e-7
  if c['gate']:assert o.strict_gate.all()
  joined=t.merge(o[['signal_date','amount']],on='signal_date',validate='one_to_one',suffixes=('_filled','_ordered'))
  assert np.max(abs(joined.amount_filled-joined.amount_ordered))<1e-7
  assert np.max(abs(t.units-t.amount/(1+.001425)/t.adjusted_open))<1e-8
  checks.append(dict(period=period,id=c['id'],formula_error=error,orders=len(o),trades=len(t),episode_and_cash_audit=True))
  row=d[d.date==trough].iloc[0]
  stress.append(dict(period=period,id=c['id'],dca_trough_date=str(trough.date()),cash=row.cash,cash_pct=100*row.cash/row.nav,ratio=row.ratio,depth=row.depth,episode_K=row.episode_K,episode_D=row.episode_D))
pd.DataFrame(checks).to_csv(OUT/'order_validation.csv',index=False)
pd.DataFrame(stress).to_csv(OUT/'trough_cash.csv',index=False)
print('Passed independent order/episode audits:',len(checks))

"""Predeclared direct-maintenance-ratio sizing comparison, never sell."""
from pathlib import Path
import sys,json,itertools,hashlib
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
sys.path.insert(0,str(ROOT/'dynamic_budget'))
import model
FEE=.001425
DESIGN=json.loads((OUT/'design.json').read_text())

def features(m):
 f=model.features(m)
 past=m.shift(1).rolling(252)
 for q in [.2,.25,.5,.75]:f[f'q{int(100*q)}']=past.quantile(q)
 arr=m.to_numpy()
 f['percentile']=[np.mean(arr[i-252:i]<=v) if i>=252 else np.nan for i,v in enumerate(arr)]
 f['depth']=((f.q20-m)/(f.q75-f.q25).clip(lower=1e-6)).clip(lower=0)
 f['k2_fraction']=-np.expm1(-2*f.hazard)
 return f

def policies():
 out=[]
 for family in ['percentile','depth','target']:
  grid=DESIGN[family];keys=list(grid)
  for vals in itertools.product(*(grid[k] for k in keys)):
   for gate in DESIGN['gates']:
    cfg=dict(family=family,gate=gate,**dict(zip(keys,vals)))
    cfg['id']=family+('_strict' if gate else '_open')+'_'+ '_'.join(f'{k}{v:g}' for k,v in zip(keys,vals))
    out.append(cfg)
 return out

def simulate(p,f,cfg,start,end):
 q=p.loc[start:end];ff=f.reindex(q.index)
 assert ff[['depth','percentile','k2_fraction']].notna().all().all()
 cash=units=contrib=spent=fees=previous=0.;twr=1.;last_month=None
 pending=None;active=None;episodes=[];trades=[];rows=[];orders=[];flows=[]
 for date,r in q.iterrows():
  s=ff.loc[date];preflow=cash+units*r.adjusted_open
  inflow=0.
  if r.tradable and date.to_period('M')!=last_month:
   inflow=10000.;cash+=inflow;contrib+=inflow;flows.append((date,-inflow));last_month=date.to_period('M')
  fill=None
  if cfg['family']=='DCA' and inflow:fill=dict(amount=inflow,signal=date,episode=None)
  elif pending is not None and r.tradable:fill=pending;pending=None
  if fill is not None:
   amount=min(cash,fill['amount']);fee=amount*FEE/(1+FEE);added=amount/(1+FEE)/r.adjusted_open
   assert amount>=0 and amount<=cash+1e-7
   cash-=amount;units+=added;spent+=amount;fees+=fee
   if fill['episode'] is not None:fill['episode']['D']+=amount
   trades.append(dict(date=date,signal_date=fill['signal'],amount=amount,fee=fee,units=added,adjusted_open=r.adjusted_open,episode=fill['episode']['id'] if fill['episode'] else 0))
  nav=cash+units*r.adjusted_close
  if inflow:
   factor=(preflow/previous if previous>0 else 1.)*nav/(preflow+inflow)
  else:factor=nav/previous if previous>0 else 1.
  twr*=factor;previous=nav
  # Episodes are governed by ratio alone, independent of the original entry gate.
  if cfg['family']=='target':
   if active is not None and s.ratio>=s.q50:
    active['end']=date;active=None
   if active is None and s.depth>0:
    reserved=pending['amount'] if pending else 0.
    active=dict(id=len(episodes)+1,start=date,end=pd.NaT,K=max(0,cash-reserved),D=0.)
    episodes.append(active)
  amount=0.;target=np.nan
  eligible=(not cfg.get('gate',False)) or bool(s.buy_state)
  if pending is None and eligible:
   if cfg['family']=='percentile':amount=cash*cfg['a']*max(0,1-s.percentile/cfg['p0'])**cfg['r']
   elif cfg['family']=='depth':amount=cash*cfg['a']*s.depth/(cfg['h']+s.depth)
   elif cfg['family']=='target' and active is not None:
    target=active['K']*s.depth/(cfg['h']+s.depth)
    amount=min(cash,max(0,target-active['D']))
   elif cfg['family']=='kappa2' and s.buy_state:amount=cash*s.k2_fraction
  if amount>1e-8:
   assert amount<=cash+1e-7
   pending=dict(amount=amount,signal=date,episode=active if cfg['family']=='target' else None)
   orders.append(dict(signal_date=date,amount=amount,cash=cash,depth=s.depth,percentile=s.percentile,strict_gate=bool(s.buy_state),episode=active['id'] if active else 0,K=active['K'] if active else np.nan,D=active['D'] if active else np.nan,target=target))
  rows.append(dict(date=date,inflow=inflow,contributions=contrib,spent=spent,fees=fees,cash=cash,units=units,fund=units*r.adjusted_close,nav=nav,twr=twr,ratio=s.ratio,depth=s.depth,percentile=s.percentile,strict_gate=bool(s.buy_state),episode=active['id'] if active else 0,episode_K=active['K'] if active else np.nan,episode_D=active['D'] if active else np.nan,pending=pending['amount'] if pending else 0.))
 d=pd.DataFrame(rows);t=pd.DataFrame(trades);o=pd.DataFrame(orders);e=pd.DataFrame(episodes)
 dd=d.twr/d.twr.cummax().clip(lower=1)-1
 deep=d.percentile<=.05
 ss=dict(id=cfg['id'],family=cfg['family'],gate=cfg.get('gate',False),start=str(q.index[0].date()),end=str(q.index[-1].date()),contributions=contrib,final_nav=nav,xirr_pct=100*model.old.base.original.xirr(flows+[(q.index[-1],nav)]),twr_pct=100*(twr-1),mdd_pct=100*dd.min(),cash=cash,buy_spent=spent,mean_cash_pct=100*(d.cash/d.nav).mean(),deep_mean_cash_pct=100*(d.loc[deep,'cash']/d.loc[deep,'nav']).mean(),deep_min_cash=d.loc[deep,'cash'].min(),buy_days=len(t),episode_count=len(e),params=json.dumps({k:v for k,v in cfg.items() if k not in ['id','family','gate']}))
 assert np.max(abs(d.cash-(d.contributions-d.spent)))<1e-6
 assert d.cash.min()>-1e-7 and d.units.diff().dropna().min()>=0
 assert np.max(abs(d.nav-d.cash-d.fund))<1e-6
 if len(t):
  assert abs(t.units.sum()-units)<1e-7 and abs(t.amount.sum()-spent)<1e-6
  if cfg['family']!='DCA':
   assert (t.signal_date<t.date).all()
   tradable=p.index[p.tradable]
   assert all(tradable[tradable>row.signal_date][0]==row.date for row in t.itertuples())
 if len(e):assert (e.D<=e.K+1e-6).all()
 return ss,d,t,o,e

def save(result,prefix):
 for suffix,frame in zip(['daily','trades','orders','episodes'],result[1:]):frame.to_csv(OUT/f'{prefix}_{suffix}.csv',index=False)

def add_comparison(df):
 dca=df[df.family=='DCA'].iloc[0]
 df['nav_gap']=df.final_nav-dca.final_nav
 df['mdd_gap_pp']=df.mdd_pct-dca.mdd_pct
 df['meets_objective']=(df.nav_gap>1e-6)&(df.mdd_gap_pp>=-1e-9)
 return df

def main():
 p=pd.read_csv(OUT/'prices.csv',parse_dates=['date']).set_index('date')
 m=pd.read_csv(OUT/'margin.csv',parse_dates=['date']).set_index('date').ratio
 f=features(m);f.to_csv(OUT/'features.csv',index_label='date')
 candidates=policies();benchmarks=[dict(id='DCA',family='DCA'),dict(id='kappa2',family='kappa2',gate=True)]
 configs=candidates+benchmarks;byid={c['id']:c for c in configs}
 start,end=DESIGN['train'];train=add_comparison(pd.DataFrame([simulate(p,f,c,start,end)[0] for c in configs]))
 train.to_csv(OUT/'training_grid.csv',index=False)
 selected=[]
 for (family,gate),g in train[~train.family.isin(['DCA','kappa2'])].groupby(['family','gate']):
  feasible=g[g.mdd_gap_pp>=-1e-9]
  winner=(feasible.sort_values(['final_nav','id'],ascending=[False,True]) if len(feasible) else g.sort_values(['mdd_pct','id'],ascending=[False,True])).iloc[0]
  selected.append(dict(byid[winner.id],training_feasible=bool(len(feasible)),training_meets_objective=bool(winner.meets_objective)))
 (OUT/'selected_before_tests.json').write_text(json.dumps(selected,indent=2))
 allsummary=[];robust=[];validations=[]
 for label,(start,end) in zip(['train_2016_2019','test_2020_2023','test_2024_2026'],[DESIGN['train']]+DESIGN['tests']):
  results={c['id']:simulate(p,f,c,start,end) for c in configs}
  grid=add_comparison(pd.DataFrame([r[0] for r in results.values()]));grid['period']=label
  grid['selected']=grid.id.isin([c['id'] for c in selected]+['DCA','kappa2'])
  grid.to_csv(OUT/f'{label}_grid.csv',index=False)
  allsummary.append(grid[grid.selected])
  for c in selected+benchmarks:save(results[c['id']],label+'_'+c['id'])
  for (family,gate),g in grid[~grid.family.isin(['DCA','kappa2'])].groupby(['family','gate']):
   robust.append(dict(period=label,family=family,gate=gate,configs=len(g),successes=int(g.meets_objective.sum()),nav_min=g.final_nav.min(),nav_max=g.final_nav.max(),mdd_min=g.mdd_pct.min(),mdd_max=g.mdd_pct.max()))
  # Reconcile the new accounting implementation with the established DCA/kappa2 engine.
  model.old.base.START=pd.Timestamp(start)
  oldf=model.features(m);oldf['spend_fraction']=-np.expm1(-2*oldf.hazard)
  for cid in ['DCA','kappa2']:
   old=model.simulate(p,oldf,cid,1. if cid=='DCA' else 0.,False,pd.Timestamp(end))[0]
   for col in ['nav','cash','twr']:
    error=float(np.max(abs(old[col].to_numpy()-results[cid][1][col].to_numpy())))
    assert error<1e-6,(label,cid,col,error)
    validations.append(dict(check=f'{label}_{cid}_{col}_old_engine',error=error))
  # Prefix invariance of features, holdings and budget state.
  midpoint=p.loc[start:end].index[len(p.loc[start:end])//2]
  shortf=features(m.loc[:midpoint]);pd.testing.assert_frame_equal(shortf,f.loc[:midpoint])
  for c in selected:
   short=simulate(p.loc[:midpoint],shortf,c,start,midpoint)[1]
   full=results[c['id']][1].set_index('date').loc[:midpoint]
   for col in ['nav','cash','units','twr','episode_K','episode_D']:
    np.testing.assert_allclose(short[col],full[col],rtol=0,atol=1e-8,equal_nan=True)
   validations.append(dict(check=f'{label}_{c["id"]}_prefix',error=0))
 # Independently verify trailing rank and quantile units on sample dates.
 for dt in pd.to_datetime(['2016-01-04','2020-03-19','2022-10-25','2026-07-29']):
  prior=m.loc[m.index<dt].iloc[-252:];depth=max(0,(np.quantile(prior,.2)-m.loc[dt])/max(np.quantile(prior,.75)-np.quantile(prior,.25),1e-6))
  assert abs(depth-f.loc[dt,'depth'])<1e-10
  assert abs((prior<=m.loc[dt]).mean()-f.loc[dt,'percentile'])<1e-10
 pd.concat(allsummary).to_csv(OUT/'selected_summary.csv',index=False)
 pd.DataFrame(robust).to_csv(OUT/'robustness.csv',index=False)
 pd.DataFrame(validations).to_csv(OUT/'validation.csv',index=False)
 hashes={fn.name:hashlib.sha256(fn.read_bytes()).hexdigest() for fn in [Path(__file__),OUT/'prepare.py',OUT/'design.json',OUT/'prices.csv',OUT/'margin.csv']}
 (OUT/'hashes.json').write_text(json.dumps(hashes,indent=2))
 print(pd.concat(allsummary)[['period','id','final_nav','xirr_pct','mdd_pct','cash','mean_cash_pct','meets_objective']].round(3).to_string(index=False))
 print('ROBUSTNESS');print(pd.DataFrame(robust).to_string(index=False));print('validation',len(validations))
if __name__=='__main__':main()

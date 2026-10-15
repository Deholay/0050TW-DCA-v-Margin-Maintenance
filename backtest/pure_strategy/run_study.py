"""All monthly contributions allocated to strategy, zero core DCA.
DCA is a separate benchmark account only. Main model is dynamic_budget tau20
with original mirror-window exits; no coefficient or signal changes.
Buy-only and original full-window policies are diagnostic comparisons.
Monthly deposits, prices, dates and fees match all previous tests.
"""
from pathlib import Path
import sys,json,hashlib
import pandas as pd
import numpy as np
H=Path(__file__).resolve().parent;ROOT=H.parent
sys.path.insert(0,str(ROOT/'dynamic_budget'))
import model
old=model.old
checks=[]
def check(label,a,b):
 error=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
 assert error<1e-7,(label,error)
 checks.append(dict(check=label,max_error=error,passed=True))

def main():
 p0=pd.read_csv(ROOT/'period_2020_2023/prices_reviewed.csv',parse_dates=['date']).set_index('date')
 m0=pd.read_csv(ROOT/'period_2020_2023/finlab_margin.csv',parse_dates=['date']).set_index('date').ratio
 p1,m1=old.base.original.load()
 configs=[('DCA_benchmark',old.simulate,1.,False),('pure_dynamic_roundtrip',model.simulate,0.,True),('pure_dynamic_buy_only',model.simulate,0.,False),('pure_old_roundtrip',old.simulate,0.,True),('pure_old_buy_only',old.simulate,0.,False)]
 summaries=[];annual=[]
 datasets=[('2020_2023_finlab',p0,m0,'2020-01-02','2023-12-29'),('2023_2026_istock',p1,m1,'2023-03-01','2026-10-14'),('2023_2026_finlab',p1,m0.reindex(m1.index),'2023-03-01','2026-10-14')]
 for label,p,m,start,end in datasets:
  old.base.START=pd.Timestamp(start);f=model.features(m)
  assert f.loc[start:end,'ready'].all()
  results={name:fn(p,f,name,core,exits,pd.Timestamp(end)) for name,fn,core,exits in configs}
  dc=results['DCA_benchmark'][0].set_index('date')
  mixed=pd.read_csv(ROOT/f'dynamic_budget/{label}_daily.csv',parse_dates=['date'])
  for name,fn,core,exits in configs:
   r=results[name];d,t,c,s=r
   row=old.summarize(r,end);row['dataset']=label
   row['gap_vs_dca']=row['final_nav']-dc.nav.iloc[-1]
   row['meets_objective']=row['gap_vs_dca']>1e-6 and row['mdd_pct']>=old.summarize(results['DCA_benchmark'],end)['mdd_pct']-1e-9
   row['no_stock_days']=int((d.fund<1e-8).sum());row['total_days']=len(d)
   row['first_buy']=str(t.loc[t.side=='buy','date'].min().date())
   row['last_sell']=str(t.loc[t.side=='sell','date'].max().date()) if (t.side=='sell').any() else None
   summaries.append(row)
   assert d.contributions.equals(results['DCA_benchmark'][0].contributions)
   if core==0:
    assert (d.core_units==0).all() and (d.core_fund==0).all()
    assert (t.sleeve=='tactical').all()
    check(label+name+'_all_deposits_tactical',d.tactical_contributions,d.contributions)
    prior_name={'pure_dynamic_roundtrip':'model_roundtrip','pure_dynamic_buy_only':'model_buy_only','pure_old_roundtrip':'old_roundtrip','pure_old_buy_only':'old_buy_only'}[name]
    mix=mixed.loc[mixed.strategy==prior_name].set_index('date');pure=d.set_index('date')
    for col in ['nav','cash','fund','contributions']:check(label+name+'_mixture_'+col,mix[col],.8*dc[col]+.2*pure[col])
   else:
    baseline=mixed.loc[mixed.strategy=='DCA_100'].set_index('date')
    for col in ['nav','cash','twr']:check(label+'_benchmark_'+col,dc[col],baseline[col])
   previous=1.
   for year,g in d.groupby(d.date.dt.year):
    last=g.iloc[-1];annual.append(dict(dataset=label,strategy=name,year=int(year),nav=last.nav,cash=last.cash,contributions=last.contributions,annual_twr_pct=100*(last.twr/previous-1)));previous=last.twr
  for kind,i in [('daily',0),('trades',1),('orders',2),('signals',3)]:pd.concat([r[i] for r in results.values()],ignore_index=True).to_csv(H/f'{label}_{kind}.csv',index=False)
  # Spot-check pure policy truncation and decisions with no core capital.
  endpoint='2022-12-30' if start.startswith('2020') else '2026-07-29'
  short=model.simulate(p.loc[:endpoint],model.features(m.loc[:endpoint]),'pure_dynamic_roundtrip',0.,True,pd.Timestamp(endpoint))[0].set_index('date')
  full=results['pure_dynamic_roundtrip'][0].set_index('date')
  for col in ['nav','cash','tactical_units','twr']:check(label+'_prefix_'+col,short[col],full.loc[short.index,col])
 summary=pd.DataFrame(summaries);summary.to_csv(H/'summary.csv',index=False)
 pd.DataFrame(annual).to_csv(H/'annual.csv',index=False)
 pd.DataFrame(checks).to_csv(H/'validation.csv',index=False)
 sources=[Path(__file__),ROOT/'dynamic_budget/model.py',ROOT/'dynamic_budget/methodology.json',ROOT/'margin_daily.csv',ROOT/'prices_reviewed.csv',ROOT/'period_2020_2023/finlab_margin.csv',ROOT/'period_2020_2023/prices_reviewed.csv']
 meta=dict(rules=__doc__,main='pure_dynamic_roundtrip',monthly_contribution=10000,core_fraction=0,leverage=False,parameters_changed=False,tau=20,checks=len(checks),caveats=['2020–2023 uses FinLab combined TWSE+TPEx; 2023–2026 primary uses listed iStock, with a matched-calendar FinLab check.','Periods overlap in 2023 and are exploratory, not independent out-of-sample.','Adjusted total-return fractional units, buy/sell fee 0.1425%, ETF sales tax 0.1%, zero cash yield; no minimum fee or spread.','Monthly contributions enter cash; no routine purchases in the pure strategy.','Dynamic-model order counts are daily orders, not multi-day entry-cycle counts.'],sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
 (H/'methodology.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
 print(summary[['dataset','strategy','contributions','final_nav','xirr_pct','mdd_pct','mean_cash_pct','cash','no_stock_days','total_days','first_buy','meets_objective']].round(4).to_string(index=False));print('checks',len(checks))
if __name__=='__main__':main()

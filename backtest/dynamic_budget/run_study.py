"""Exploratory mathematical cash-control model; no optimized coefficients.

At close t define q10 = prior 252 observations' 10th percentile of market
margin-maintenance ratio m. sigma = standard deviation of prior 60 daily
changes; sigma_ref = median of prior 252 sigma observations (same lagged
features as the original signal). Define non-probabilistic stress
z = max(0, (q10-m)/(sigma_ref*sqrt(20))).
H = clip(tau*(sigma/sigma_ref)^2,5,60), lambda=(1+z^2)/H.
When the ORIGINAL full buy condition is true (not just its rising edge),
schedule B = C * (1-exp(-lambda)) at the next tradable open, where C is
remaining tactical cash at the signal close. It spends no new monthly
cash arriving after that close. Recompute every eligible close. A halted
ETF's pending order remains fixed until its next tradable open.

This is an invented, testable cash-depletion control law, not a fitted
expected-return forecast, an optimal-control solution, or probability of a
bottom. sqrt(20) is a scale convention, not a valid asserted diffusion
model. tau=20 is declared main specification; 10/40 are sensitivity checks,
not selection candidates. H is an e-folding time (constant z=0 consumes
63.2% after H eligible buy days), not a deadline to finish all buying.
High variance lengthens H; deeper stress raises spending intensity.

80% of monthly deposits remains core DCA. No leverage or new capital.
Original prices, fees, tax, adjusted-unit assumptions and TWR conventions
are unchanged. Compare buy-only and original mirror-window sells. No new
buy orders during an active sell campaign. Daily buy orders finish before
the next close, so a sell can trigger when the old multi-day buy might
have ignored it; that interaction is part of this changed policy.
Test periods are exploratory retrospective checks and overlap in 2023.
"""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import model
old=model.old
H=Path(__file__).resolve().parent
ROOT=H.parent
checks=[]
def check(name,a,b,tol=1e-7):
 e=float(np.max(np.abs(np.asarray(a)-np.asarray(b))));assert e<tol,(name,e)
 checks.append(dict(check=name,max_error=e,tolerance=tol,passed=True))

def run(label,p,m,start,end):
 old.base.START=pd.Timestamp(start)
 f=model.features(m)
 assert f.loc[start:end,'ready'].all()
 assert f.loc[start:end,['stress','budget_horizon','spend_fraction']].notna().all().all()
 assert f.loc[start:end,'spend_fraction'].between(0,1,inclusive='neither').all()
 configs=[('DCA_100',old.simulate,1.,False),('old_buy_only',old.simulate,.8,False),('old_roundtrip',old.simulate,.8,True),('model_buy_only',model.simulate,.8,False),('model_roundtrip',model.simulate,.8,True)]
 results={n:fn(p,f,n,c,e,pd.Timestamp(end)) for n,fn,c,e in configs}
 summaries=[]
 for n,r in results.items():
  s=old.summarize(r,end);s['dataset']=label
  s['gap_vs_dca']=s['final_nav']-results['DCA_100'][0].iloc[-1].nav
  summaries.append(s)
 summary=pd.DataFrame(summaries)
 dd=summary.loc[summary.strategy=='DCA_100','mdd_pct'].iloc[0]
 summary['meets_objective']=(summary.gap_vs_dca>1e-6)&(summary.mdd_pct>=dd-1e-9)
 summary.to_csv(H/f'{label}_summary.csv',index=False)
 for kind,i in [('daily',0),('trades',1),('orders',2),('signals',3)]:pd.concat([r[i] for r in results.values()],ignore_index=True).to_csv(H/f'{label}_{kind}.csv',index=False)
 f.to_csv(H/f'{label}_features.csv',index_label='date')
 # Core remains identical; all value creation must come from tactical rules.
 for exits in [False,True]:
  n='model_roundtrip' if exits else 'model_buy_only'
  r=results[n];d=r[0].set_index('date');dc=results['DCA_100'][0].set_index('date')
  full=model.simulate(p,f,'diagnostic',0.,exits,pd.Timestamp(end))[0].set_index('date')
  for col in ['nav','cash','fund','contributions']:check(label+n+col,d[col],.8*dc[col]+.2*full[col])
  check(label+n+'_core',d.core_fund,.8*dc.fund)
  tr=r[1];tr=tr[(tr.sleeve=='tactical')&(tr.side=='buy')]
  for row in tr.itertuples():
   prev=d.loc[row.signal_date]
   check(label+n+'_order_'+str(row.date.date()),-row.cash_change,prev.cash*f.loc[row.signal_date,'spend_fraction'])
   assert f.loc[row.signal_date,'buy_state']
  # Prefix invariance includes features and trading, not just final balance.
  for endpoint in (["2021-12-30","2022-12-30"] if start.startswith('2020') else ["2025-12-31","2026-07-29"]):
   ff=model.features(m.loc[:endpoint]);pd.testing.assert_frame_equal(ff,f.loc[:endpoint])
   sub=model.simulate(p.loc[:endpoint],ff,n,.8,exits,pd.Timestamp(endpoint))[0].set_index('date')
   for col in ['nav','cash','tactical_units','twr']:check(label+n+endpoint+col,sub[col],d.loc[sub.index,col])
 sens=[]
 for tau in [10.,20.,40.]:
  ff=model.features(m,tau)
  for exits in [False,True]:
   n='model_roundtrip' if exits else 'model_buy_only'
   rr=model.simulate(p,ff,n,.8,exits,pd.Timestamp(end))
   ss=old.summarize(rr,end);ss.update(dataset=label,tau=tau,gap_vs_dca=ss['final_nav']-results['DCA_100'][0].iloc[-1].nav)
   sens.append(ss)
 pd.DataFrame(sens).to_csv(H/f'{label}_sensitivity.csv',index=False)
 return summary,results,f

def main():
 p0=pd.read_csv(ROOT/'period_2020_2023/prices_reviewed.csv',parse_dates=['date']).set_index('date')
 m0=pd.read_csv(ROOT/'period_2020_2023/finlab_margin.csv',parse_dates=['date']).set_index('date').ratio
 p1,m1=old.base.original.load()
 all_s=[]
 datasets=[('2020_2023_finlab',p0,m0,'2020-01-02','2023-12-29'),('2023_2026_istock',p1,m1,'2023-03-01','2026-10-14'),('2023_2026_finlab',p1,m0.reindex(m1.index),'2023-03-01','2026-10-14')]
 for label,p,m,start,end in datasets:
  s,r,f=run(label,p,m,start,end);all_s.append(s)
  previous=pd.read_csv(ROOT/('review_20261015/window_roundtrip_summary.csv' if label=='2023_2026_istock' else 'period_2020_2023/'+('period_summary.csv' if label=='2020_2023_finlab' else 'overlap_finlab_combined_summary.csv'))).set_index('strategy')
  for new,prior in [('DCA_100','DCA_100'),('old_buy_only','DCA_80_window_buy_only'),('old_roundtrip','DCA_80_window_roundtrip')]:
   for col in ['final_nav','mdd_pct','xirr_pct']:check(label+'_baseline_'+new+col,s.set_index('strategy').loc[new,col],previous.loc[prior,col])
  if label=='2023_2026_istock':
   tr=r['model_roundtrip'][1];sub=tr[(tr.sleeve=='tactical')&tr.date.between('2026-07-20','2026-07-31')];sub.to_csv(H/'july_model_buys.csv',index=False)
   print('July actual fills\n',sub[['date','signal_date','adjusted_open','cash_change']].to_string(index=False))
   for day in ['2026-07-20','2026-07-28','2026-07-29']:
    print(day,f.loc[day,['ratio','stress','budget_horizon','spend_fraction']].to_dict())
 summary=pd.concat(all_s);summary.to_csv(H/'summary.csv',index=False)
 pd.DataFrame(checks).to_csv(H/'validation.csv',index=False)
 files=[Path(__file__),H/'model.py',ROOT/'margin_daily.csv',ROOT/'prices_reviewed.csv',ROOT/'period_2020_2023/finlab_margin.csv',ROOT/'period_2020_2023/prices_reviewed.csv']
 meta=dict(rules=__doc__,main_tau=20,sensitivity_tau=[10,20,40],parameter_search=False,return_forecast_estimated=False,causal_claim=False,data_limitations='2020–2023 uses user-approved FinLab combined TWSE+TPEx; 2023–2026 listed iStock and a matched-calendar FinLab robustness run. No claim of independent out-of-sample validation. All previous price/cost/source-revision caveats remain.',checks=len(checks),source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
 (H/'methodology.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
 print(summary[['dataset','strategy','final_nav','gap_vs_dca','xirr_pct','mdd_pct','cash','mean_cash_pct','buy_fill_days','meets_objective']].round(4).to_string(index=False));print('checks',len(checks))
if __name__=='__main__':main()

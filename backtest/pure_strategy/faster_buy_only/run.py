"""User-requested more generous cash deployment, pure strategy, never sell.
Keep original buy eligibility, stress z and lagged volatility horizon H.
Spend fraction = 1 - exp(-kappa * (1+z^2)/H).
Declare kappa=2 as new main specification before comparing outcomes;
1 baseline, 1.5 and 3 sensitivities. Not a fitted optimal parameter.
Unlike reducing tau before clipping H, kappa acts AFTER the 5..60 clipping.
At kappa2, spend fraction is 2*f-f^2, not exactly twice the dollar order.
All other cashflows, execution, data and cost assumptions are unchanged.
"""
from pathlib import Path
import sys,json,hashlib
import pandas as pd
import numpy as np
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT/'dynamic_budget'))
import model
PRIMARY_KAPPA=2.
checks=[]
def check(name,a,b):
 e=float(np.max(np.abs(np.asarray(a)-np.asarray(b))));assert e<1e-7,(name,e)
 checks.append(dict(check=name,max_error=e,passed=True))
def features(m,k):
 f=model.features(m)
 f['base_horizon']=f.budget_horizon
 f['budget_horizon']=f.base_horizon/k
 f['hazard']=f.hazard*k
 f['spend_fraction']=-np.expm1(-f.hazard)
 f['kappa']=k
 return f

def main():
 p0=pd.read_csv(ROOT/'period_2020_2023/prices_reviewed.csv',parse_dates=['date']).set_index('date')
 m0=pd.read_csv(ROOT/'period_2020_2023/finlab_margin.csv',parse_dates=['date']).set_index('date').ratio
 p1,m1=model.old.base.original.load()
 summaries=[];julys=[]
 datasets=[('2020_2023_finlab',p0,m0,'2020-01-02','2023-12-29'),('2023_2026_istock',p1,m1,'2023-03-01','2026-10-14'),('2023_2026_finlab',p1,m0.reindex(m1.index),'2023-03-01','2026-10-14')]
 for label,p,m,start,end in datasets:
  model.old.base.START=pd.Timestamp(start)
  frames={};results={}
  for k in [1.,1.5,2.,3.]:
   f=features(m,k);frames[k]=f
   assert f.loc[start:end,'spend_fraction'].between(0,1,inclusive='neither').all()
   assert f.buy_state.equals(frames[1.].buy_state)
   assert (f.spend_fraction.dropna()>=frames[1.].spend_fraction.dropna()).all()
   r=model.simulate(p,f,'pure_dynamic_buy_only',0.,False,pd.Timestamp(end));results[k]=r
   d,t,c,s=r
   assert t.side.eq('buy').all() and t.sleeve.eq('tactical').all()
   assert (d.core_units==0).all() and (d.sale_net==0).all() and (d.sell_tax==0).all()
   check(label+str(k)+'_cash',d.cash,d.contributions-d.buy_spent)
   check(label+str(k)+'_unit_ledger',d.tactical_units.iloc[-1],t.units.sum())
   if k==1:
    prior=pd.read_csv(ROOT/f'pure_strategy/{label}_daily.csv',parse_dates=['date']);prior=prior[prior.strategy=='pure_dynamic_buy_only'].reset_index(drop=True)
    for col in ['nav','cash','fund','twr']:check(label+'_baseline_'+col,d[col],prior[col])
   ss=model.old.summarize(r,end);ss.update(dataset=label,kappa=k,buy_spent=d.iloc[-1].buy_spent,ending_cash_pct=100*d.iloc[-1].cash/d.iloc[-1].nav)
   summaries.append(ss)
   tag=str(k).replace('.','p')
   for kind,df in zip(['daily','trades','orders','signals'],r):df.to_csv(OUT/f'{label}_k{tag}_{kind}.csv',index=False)
   f.to_csv(OUT/f'{label}_k{tag}_features.csv',index_label='date')
   if label=='2023_2026_istock':
    july=t[t.date.between('2026-07-20','2026-07-31')].copy();july['kappa']=k;julys.append(july)
  # Analytical fraction identity at the declared main kappa.
  valid=frames[1.].spend_fraction.notna()
  check(label+'_kappa2_identity',frames[2.].loc[valid,'spend_fraction'],2*frames[1.].loc[valid,'spend_fraction']-frames[1.].loc[valid,'spend_fraction']**2)
  endpoint='2022-12-30' if start.startswith('2020') else '2026-07-29'
  ff=features(m.loc[:endpoint],2.);pd.testing.assert_frame_equal(ff,frames[2.].loc[:endpoint])
  prefix=model.simulate(p.loc[:endpoint],ff,'pure_dynamic_buy_only',0.,False,pd.Timestamp(endpoint))[0].set_index('date')
  main=results[2.][0].set_index('date')
  for col in ['nav','cash','tactical_units','twr']:check(label+'_prefix_'+col,prefix[col],main.loc[prefix.index,col])
  baseline=model.old.simulate(p,frames[1.],'DCA_benchmark',1.,False,pd.Timestamp(end))
  ss=model.old.summarize(baseline,end);ss.update(dataset=label,kappa=0,buy_spent=ss['contributions'],ending_cash_pct=0)
  summaries.append(ss)
 summary=pd.DataFrame(summaries)
 for label in summary.dataset.unique():
  base=summary[(summary.dataset==label)&(summary.strategy=='DCA_benchmark')].iloc[0]
  mask=summary.dataset==label
  summary.loc[mask,'gap_vs_dca']=summary.loc[mask,'final_nav']-base.final_nav
  summary.loc[mask,'meets_objective']=(summary.loc[mask,'final_nav']>base.final_nav+1e-6)&(summary.loc[mask,'mdd_pct']>=base.mdd_pct-1e-9)
 summary.to_csv(OUT/'summary.csv',index=False)
 pd.concat(julys).to_csv(OUT/'july_buys.csv',index=False)
 pd.DataFrame(checks).to_csv(OUT/'validation.csv',index=False)
 files=[Path(__file__),ROOT/'dynamic_budget/model.py',ROOT/'prices_reviewed.csv',ROOT/'margin_daily.csv',ROOT/'period_2020_2023/prices_reviewed.csv',ROOT/'period_2020_2023/finlab_margin.csv']
 rules=dict(description=__doc__,main_kappa=PRIMARY_KAPPA,kappa_comparisons=[1,1.5,2,3],monthly_deposit=10000,core_fraction=0,exits_enabled=False,leverage=False,entry_filter_changed=False,checks=len(checks),caveats=['Exploratory historical comparison, not a fitted optimum or independent out-of-sample test.','2020–2023 FinLab combined TWSE+TPEx; 2023–2026 listed iStock plus matched-calendar FinLab robustness. Periods overlap in 2023.','Fractional adjusted total-return units, fee 0.1425%, no cash interest, no minimum fee or spread.'],sha256={str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in files})
 (OUT/'rules.json').write_text(json.dumps(rules,ensure_ascii=False,indent=2))
 print(summary[['dataset','strategy','kappa','final_nav','xirr_pct','mdd_pct','cash','buy_spent','mean_cash_pct','meets_objective']].round(4).to_string(index=False));print('checks',len(checks))
 print(pd.concat(julys)[['kappa','date','cash_change']].round(2).to_string(index=False))
if __name__=='__main__':main()

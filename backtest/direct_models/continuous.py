"""User-requested continuous 2020–2026 test; no reselection or account reset."""
import json
import numpy as np
import pandas as pd
import run as eng

def main():
 out=eng.OUT
 p=pd.read_csv(out/'prices.csv',parse_dates=['date']).set_index('date')
 m=pd.read_csv(out/'margin.csv',parse_dates=['date']).set_index('date').ratio
 f=eng.features(m)
 selected=json.loads((out/'selected_before_tests.json').read_text())
 ids={c['id'] for c in selected}|{'DCA','kappa2'}
 configs=eng.policies()+[dict(id='DCA',family='DCA'),dict(id='kappa2',family='kappa2',gate=True)]
 results={c['id']:eng.simulate(p,f,c,'2020-01-01','2026-10-14') for c in configs}
 grid=eng.add_comparison(pd.DataFrame([r[0] for r in results.values()]))
 grid['selected']=grid.id.isin(ids)
 assert grid.contributions.eq(820000).all()
 grid.to_csv(out/'continuous_2020_2026_grid.csv',index=False)
 grid[grid.selected].to_csv(out/'continuous_2020_2026_summary.csv',index=False)
 checks=[];annual=[]
 for cid in ids:
  r=results[cid];eng.save(r,'continuous_2020_2026_'+cid)
  d=r[1]
  # Continuous extension must preserve every earlier result and budget state.
  old=pd.read_csv(out/f'test_2020_2023_{cid}_daily.csv',parse_dates=['date'])
  early=d[d.date<='2023-12-31']
  for col in ['nav','cash','units','twr','episode_K','episode_D']:
   np.testing.assert_allclose(early[col],old[col],atol=1e-8,rtol=0,equal_nan=True)
  checks.append(dict(id=cid,prefix_matches_2020_2023=True))
  for year,g in d.groupby(d.date.dt.year):
   row=g.iloc[-1]
   annual.append(dict(id=cid,year=int(year),date=row.date,contributions=row.contributions,nav=row.nav,cash=row.cash,spent=row.spent,twr=row.twr))
 # Independent closed-form DCA terminal wealth with same deposit dates/costs.
 dc=results['DCA'][1];buys=dc[dc.inflow>0]
 units=sum(row.inflow/(1+eng.FEE)/p.loc[row.date,'adjusted_open'] for row in buys.itertuples())
 assert abs(units*p.loc[dc.date.iloc[-1],'adjusted_close']-dc.nav.iloc[-1])<1e-6
 pd.DataFrame(checks).to_csv(out/'continuous_validation.csv',index=False)
 pd.DataFrame(annual).to_csv(out/'continuous_annual.csv',index=False)
 (out/'continuous_rules.json').write_text(json.dumps(dict(start='2020-01-01',end='2026-10-14',monthly_deposit=10000,total_contributions=820000,parameter_selection='Unchanged 2016–2019 training selection',account_reset=False,episode_reset_at_2024=False,sales=False,leverage=False,new_target_deposit_rule=False,grid_role='Original 48 configurations, diagnostics only; no reselection',validation='Eight selected/baseline prefixes match prior 2020–2023 run; independent DCA terminal wealth identity passes'),indent=2))
 print(grid[grid.selected][['id','final_nav','xirr_pct','mdd_pct','cash','buy_spent','mean_cash_pct','nav_gap','meets_objective']].round(4).to_string(index=False))
 candidates=grid[~grid.family.isin(['DCA','kappa2'])]
 print('48-grid successes:',int(candidates.meets_objective.sum()))
 print('Grid maximum NAV:',candidates.loc[candidates.final_nav.idxmax(),['id','final_nav','mdd_pct']].to_dict())
if __name__=='__main__':main()

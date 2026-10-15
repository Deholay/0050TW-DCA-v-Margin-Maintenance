"""User correction: 100% dynamic buying, no selling and no core DCA."""
import sys,json
import pandas as pd
import numpy as np
from pathlib import Path
out=Path(__file__).resolve().parent;h=out.parents[1]
sys.path.insert(0,str(h/'dynamic_budget'))
import model
p0=pd.read_csv(h/'period_2020_2023/prices_reviewed.csv',parse_dates=['date']).set_index('date')
m0=pd.read_csv(h/'period_2020_2023/finlab_margin.csv',parse_dates=['date']).set_index('date').ratio
p1,m1=model.old.base.original.load()
rows=[]
for label,p,m,start,end in [('2020_2023_finlab',p0,m0,'2020-01-02','2023-12-29'),('2023_2026_istock',p1,m1,'2023-03-01','2026-10-14')]:
 model.old.base.START=pd.Timestamp(start)
 r=model.simulate(p,model.features(m),'pure_dynamic_buy_only',0.,False,pd.Timestamp(end))
 d,t,c,s=r
 assert t.side.eq('buy').all() and t.sleeve.eq('tactical').all()
 assert (d.core_units==0).all() and (d.sale_net==0).all() and (d.sell_tax==0).all()
 assert np.allclose(d.cash,d.contributions-d.buy_spent)
 previous=pd.read_csv(h/f'pure_strategy/{label}_daily.csv',parse_dates=['date']);previous=previous[previous.strategy=='pure_dynamic_buy_only'].reset_index(drop=True)
 previous['active_side']=previous.active_side.fillna('')
 pd.testing.assert_frame_equal(d,previous,check_dtype=False,atol=1e-7,rtol=1e-12)
 summary=model.old.summarize(r,end);last=d.iloc[-1]
 summary.update(dataset=label,buy_spent=last.buy_spent,sale_net=last.sale_net,ending_cash_pct=100*last.cash/last.nav,never_invested_principal_pct=100*last.cash/last.contributions,all_trades_buy=True,core_fraction=0,exits_enabled=False)
 rows.append(summary)
 for kind,df in zip(['daily','trades','orders','signals'],r):df.to_csv(out/f'{label}_{kind}.csv',index=False)
result=pd.DataFrame(rows);result.to_csv(out/'summary.csv',index=False)
(out/'rules.json').write_text(json.dumps(dict(main_strategy='pure_dynamic_buy_only',core_fraction=0,exits_enabled=False,monthly_deposit=10000,tau=20,signal_and_sizing_unchanged=True,verified=['No sell trades','All core holdings zero','Sale proceeds and sale tax zero','Cash = contributed principal - buy spending','Daily results reproduce earlier buy-only diagnostic after CSV empty-string normalization'],data_caveat='2020–2023 FinLab TWSE+TPEx; 2023–2026 iStock TWSE. Historical exploratory tests, overlapping in 2023.',costs='Buy fee 0.1425%, adjusted fractional total-return units, cash yield zero, no minimum fee or spread'),ensure_ascii=False,indent=2))
print(result[['dataset','contributions','buy_spent','cash','tactical_fund','final_nav','xirr_pct','mdd_pct','ending_cash_pct','mean_cash_pct','all_trades_buy','exits_enabled']].round(4).to_string(index=False))

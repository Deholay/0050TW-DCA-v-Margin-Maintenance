from pathlib import Path
import pandas as pd
import numpy as np
import run_dynamic as bt
R=Path(__file__).resolve().parent
p,m=bt.load();f=bt.features(m)
old=pd.read_csv(R/'summary.csv').set_index('strategy')
results=[];trades=[];daily=[]
for kind in ['DCA','fixed','normalized','dynamic']:
    dates=[] if kind=='DCA' else bt.event_dates(f,kind)
    for partial in ([False] if kind=='DCA' else [False,True]):
        r,d,e=bt.simulate(p,dates,kind,partial_cash=partial)
        if not partial:
            for col in ['nav','cash','buys','spent','xirr_pct','mdd_pct']:
                assert np.isclose(r[col],old.loc[kind,col]),(kind,col)
        r['cash_rule']='up_to_100k' if partial else 'full_100k_required'
        d['cash_rule']=r['cash_rule'];e['cash_rule']=r['cash_rule']
        assert r['contributions']==440000 and np.isclose(r['spent']+r['cash'],440000)
        assert (e.amount<=100000).all() and (e.amount<=e.available_cash+1e-7).all()
        results.append(r);trades.append(e);daily.append(d)
summary=pd.DataFrame(results)
summary.to_csv(R/'cash_rule_summary.csv',index=False)
pd.concat(trades).to_csv(R/'cash_rule_trades.csv',index=False)
pd.concat(daily).to_csv(R/'cash_rule_daily.csv',index=False)
print(summary[['strategy','cash_rule','signals','buys','skips','spent','cash','nav','xirr_pct','mdd_pct']].round(3).to_string(index=False))
print('\nDynamic new trades:')
print(trades[-1].to_string(index=False))

from pathlib import Path
import pandas as pd
import run_dynamic as bt
R=Path(__file__).resolve().parent
p,m=bt.load();f=bt.features(m)
daily=pd.read_csv(R/'daily.csv',parse_dates=['date'])
events=pd.read_csv(R/'events.csv',parse_dates=['signal_date','execution_date'])
rows=[]
for kind,col,cut in [('fixed','fixed_diff',-10),('normalized','z',-1.5),('dynamic','dynamic_diff',-10)]:
    condition=f.ready & (f.ratio<f.low_threshold)&(f[col]<cut)
    crosses=condition&~condition.shift(1,fill_value=False)
    episode=crosses.cumsum()
    d=daily[daily.strategy==kind].set_index('date')
    before=d.cash.shift(1,fill_value=0)+d.contributions.diff().fillna(d.contributions.iloc[0])
    for day in f.index[condition & (f.index>=bt.START)]:
        future=p.index[(p.index>day)&p.tradable&(p.index<=bt.END)]
        if len(future)==0:continue
        execution=future[0]
        cash=before.loc[execution]
        if not crosses.loc[day] and cash>0:
            rows.append(dict(strategy=kind,episode=int(episode.loc[day]),condition_date=day,possible_buy_date=execution,cash_before=cash,enough_100k=cash>=100000,reason='condition_continues_no_new_cross'))
a=pd.DataFrame(rows);a.to_csv(R/'blocked_cash_days.csv',index=False)
for kind in ['fixed','normalized','dynamic']:
    v=a[(a.strategy==kind)&a.enough_100k]
    print('\nENOUGH CASH',kind,'days',len(v),'episodes',v.episode.nunique(),'months',sorted(v.possible_buy_date.dt.strftime('%Y-%m').unique()))
    print(v.groupby('episode').agg(first=('possible_buy_date','min'),last=('possible_buy_date','max'),days=('possible_buy_date','size'),cash_min=('cash_before','min'),cash_max=('cash_before','max')).to_string())
    allv=a[a.strategy==kind]
    print('ANY POSITIVE CASH',len(allv),'episodes',allv.episode.nunique(),'months',sorted(allv.possible_buy_date.dt.strftime('%Y-%m').unique()))
    print('SKIPPED CROSSINGS')
    print(events[(events.strategy==kind)&~events.bought][['signal_date','execution_date','available_cash']].to_string(index=False))

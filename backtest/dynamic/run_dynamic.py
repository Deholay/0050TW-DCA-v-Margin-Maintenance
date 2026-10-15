"""Exploratory causal diff/window comparison. No network or parameter optimization."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_backtest import xirr,FEE
ROOT=Path(__file__).resolve().parent
START=pd.Timestamp('2023-03-01');END=pd.Timestamp('2026-10-14')

def load():
    p=pd.read_csv(ROOT.parent/'prices_reviewed.csv',parse_dates=['date']).set_index('date')
    m=pd.read_csv(ROOT.parent/'margin_daily.csv',parse_dates=['date']).set_index('date').ratio
    assert p.index.equals(m.index) and len(m)==1200 and not m.isna().any()
    vals=' '.join(str(round(x*100)) for x in m)
    h=0
    for c in vals:h=(31*h+ord(c))&0xffffffff
    assert h==1286114740
    old=pd.read_csv(ROOT.parent/'signals.csv',parse_dates=['signal_date'])
    for k in [140,150,160]:
        generated=m.index[(m<k)&(m.shift()>=k)]
        assert list(generated)==list(old.loc[old.threshold==k,'signal_date'])
    return p,m

def features(m,w=20,lookback=252,smooth=1,step=None):
    f=pd.DataFrame({'ratio':m})
    f['endpoint']=m.rolling(smooth).mean()
    f['sigma']=m.diff().shift(1).rolling(60).std(ddof=1)
    # sigma[t] itself only uses changes through t-1; historical median also ends at t-1.
    f['sigma_ref']=f.sigma.shift(1).rolling(252).median()
    f['low_threshold']=m.shift(1).rolling(lookback).quantile(.1)
    target=(w*(f.sigma_ref/f.sigma)**2).clip(5,60).round()
    windows=[];prev=None
    for v in target:
        if pd.isna(v):windows.append(np.nan);continue
        current=int(v) if prev is None or step is None else int(np.clip(v,prev-step,prev+step))
        windows.append(current);prev=current
    f['window']=windows
    f['fixed_diff']=f.endpoint-f.endpoint.shift(w)
    f['z']=f.fixed_diff/(f.sigma*np.sqrt(w))
    a=f.endpoint.to_numpy()
    f['dynamic_diff']=[a[i]-a[i-int(v)] if pd.notna(v) and i>=v else np.nan for i,v in enumerate(windows)]
    f['ready']=f[['dynamic_diff','z','low_threshold']].notna().all(axis=1)
    return f

def event_dates(f,kind,k=10,z=1.5,gate=True):
    low=(f.ratio<f.low_threshold) if gate else pd.Series(True,index=f.index)
    if kind=='fixed':condition=f.fixed_diff < -k
    elif kind=='normalized':condition=f.z < -z
    elif kind=='dynamic':condition=f.dynamic_diff < -k
    elif kind=='low_only':condition=pd.Series(True,index=f.index)
    elif kind.startswith('level_'):condition=f.ratio < int(kind.split('_')[1]);low=pd.Series(True,index=f.index)
    else:raise ValueError(kind)
    condition=condition&low&f.ready
    # A signal is the transition of the entire buy condition from false to true.
    crosses=condition & ~condition.shift(1,fill_value=False)
    return list(f.index[crosses])

def simulate(p,dates,label,start=START,ticket=100000,cash_rate=0,partial_cash=False):
    q=p.loc[start:END];trading=p.index[p.tradable]
    schedules={};unexecuted=[]
    for d in dates:
        if d<start or d>END:continue
        future=trading[(trading>d)&(trading<=END)]
        if not len(future):unexecuted.append(d);continue
        schedules.setdefault(future[0],[]).append(d)
    cash=units=contrib=spent=0.;last_month=None;last_date=None;previous=0.;twr=1.
    daily=[];events=[];flows=[]
    for date,r in q.iterrows():
        if last_date is not None:cash*=(1+cash_rate)**((date-last_date).days/365.25)
        inflow=0.
        if date.to_period('M')!=last_month and r.tradable:
            inflow=10000.;cash+=inflow;contrib+=inflow;flows.append((date,-inflow));last_month=date.to_period('M')
        pending=[pd.NaT] if label=='DCA' and inflow else schedules.get(date,[]) if label!='DCA' else []
        for signal in pending:
            amount=10000. if label=='DCA' else ticket
            if partial_cash and label!='DCA':amount=min(amount,max(0.,cash))
            bought=amount>1e-7 and cash+1e-7>=amount
            events.append(dict(strategy=label,signal_date=signal,execution_date=date,bought=bought,available_cash=cash,amount=amount if bought else 0.,adjusted_open=r.adjusted_open))
            if bought:
                assert r.tradable
                units+=amount/(1+FEE)/r.adjusted_open;cash-=amount;spent+=amount
        nav=cash+units*r.adjusted_close
        if previous+inflow:twr*=nav/(previous+inflow)
        daily.append(dict(date=date,strategy=label,cash=cash,nav=nav,fund=units*r.adjusted_close,contributions=contrib,twr=twr))
        previous=nav;last_date=date
        assert cash>=-1e-6
    d=pd.DataFrame(daily);e=pd.DataFrame(events)
    buys=sum(x['bought'] for x in events)
    fund_check=sum(x['amount']/(1+FEE)/x['adjusted_open']*q.adjusted_close.iloc[-1] for x in events)
    assert np.isclose(fund_check,units*q.adjusted_close.iloc[-1])
    dd=d.twr/d.twr.cummax()-1
    result=dict(strategy=label,start=str(q.index[0].date()),end=str(q.index[-1].date()),signals=len(events),buys=buys,skips=len(events)-buys,unexecuted_end=len(unexecuted),contributions=contrib,spent=spent,cash=cash,nav=nav,xirr_pct=100*xirr(flows+[(q.index[-1],nav)]),mdd_pct=100*dd.min(),mean_cash_pct=100*(d.cash/d.nav).mean(),ticket=ticket,cash_rate=cash_rate)
    return result,d,e

def main():
    p,m=load();f=features(m)
    assert f.loc[START:,'ready'].all()
    # Prefix invariance: appending future observations cannot change any past feature or signal.
    for n in [400,800,1000]:
        pd.testing.assert_frame_equal(features(m.iloc[:n]),f.iloc[:n])
    f.to_csv(ROOT/'features.csv',index_label='date')
    base=[('DCA',[]),('Level 140',event_dates(f,'level_140')),('Level 150',event_dates(f,'level_150')),('Level 160',event_dates(f,'level_160')),('Low only',event_dates(f,'low_only'))]
    base += [(kind,event_dates(f,kind)) for kind in ['fixed','normalized','dynamic']]
    results=[];daily=[];events=[]
    for label,dates in base:
        r,d,e=simulate(p,dates,label);results.append(r);daily.append(d);events.append(e)
    summary=pd.DataFrame(results);summary.to_csv(ROOT/'summary.csv',index=False)
    pd.concat(daily).to_csv(ROOT/'daily.csv',index=False)
    pd.concat(events).to_csv(ROOT/'events.csv',index=False)
    assert summary.contributions.nunique()==1 and summary.contributions.iloc[0]==440000
    sensitivity=[]
    # One-factor checks, not selection of an optimum. Each comparison uses matching dates/cash.
    configs=[dict(case='base',w=20,k=10,z=1.5,gate=True,smooth=1,step=None,lookback=252,ticket=100000,cash_rate=0)]
    changes=[('no_low_filter',{'gate':False}),('window10',{'w':10}),('window40',{'w':40}),('loose',{'k':5,'z':1}),('strict',{'k':15,'z':2}),('endpoint_SMA3',{'smooth':3}),('window_step_limit',{'step':1}),('ticket50000',{'ticket':50000}),('cash2pct',{'cash_rate':.02}),('history3years',{'lookback':756})]
    for name,update in changes:configs.append(dict(configs[0],**update,case=name))
    for cfg in configs:
        ff=features(m,cfg['w'],cfg['lookback'],cfg['smooth'],cfg['step'])
        start=START if cfg['lookback']==252 else pd.Timestamp('2025-01-02')
        assert ff.loc[start:,'ready'].all()
        for kind in ['DCA','low_only','fixed','normalized','dynamic']:
            dates=[] if kind=='DCA' else event_dates(ff,kind,cfg['k'],cfg['z'],cfg['gate'])
            rr,_,_=simulate(p,dates,kind,start,cfg['ticket'],cfg['cash_rate'])
            sensitivity.append(dict(rr,case=cfg['case']))
    pd.DataFrame(sensitivity).to_csv(ROOT/'sensitivity.csv',index=False)
    # Endpoint robustness, including the full 2025 drawdown, with identical start/cash for every method.
    checkpoints=pd.concat(daily)
    checkpoints=checkpoints[checkpoints.date.isin(pd.to_datetime(['2024-12-31','2025-12-31','2026-10-14']))]
    checkpoints.to_csv(ROOT/'checkpoints.csv',index=False)
    print(summary.round(3).to_string(index=False))
    print('\nSensitivity ending NAV:')
    print(pd.DataFrame(sensitivity).pivot(index='case',columns='strategy',values='nav').round())
    return summary,f

if __name__=='__main__':main()

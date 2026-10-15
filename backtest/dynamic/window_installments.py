"""Freeze cash and horizon at signal close; spend over the next W tradable opens."""
from pathlib import Path
import pandas as pd
import numpy as np
import run_dynamic as bt
from run_backtest import FEE,xirr
R=Path(__file__).resolve().parent

def simulate_installments(p,f,mode):
    triggers=set(bt.event_dates(f,'dynamic'))
    cash=units=contributed=spent=0.;month=None;previous=0.;twr=1.;active=None
    flows=[];daily=[];trades=[];campaigns=[];signals=[]
    for date,r in p.loc[bt.START:bt.END].iterrows():
        inflow=0.
        if r.tradable and month!=date.to_period('M'):
            inflow=10000.;cash+=inflow;contributed+=inflow;flows.append((date,-inflow));month=date.to_period('M')
        if active is not None and r.tradable:
            a=active
            amount=a['budget']-a['spent'] if a['remaining']==1 else a['daily_amount']
            assert amount>0 and cash+1e-6>=amount and date>a['signal_date']
            new_units=amount/(1+FEE)/r.adjusted_open
            units+=new_units;cash-=amount;spent+=amount;a['spent']+=amount;a['remaining']-=1
            a['last_buy_date']=date
            if pd.isna(a['first_buy_date']):a['first_buy_date']=date
            trades.append(dict(strategy=mode,campaign=a['campaign'],signal_date=a['signal_date'],date=date,amount=amount,adjusted_open=r.adjusted_open,units=new_units,fee=amount-amount/(1+FEE)))
            if a['remaining']==0:
                assert np.isclose(a['spent'],a['budget'])
                active=None
        # New information is only observed after the day's open transactions.
        if date in triggers:
            status='ignored_active' if active is not None else 'no_cash' if cash<1e-6 else 'opened'
            signals.append(dict(strategy=mode,signal_date=date,cash=cash,window=int(f.loc[date,'window']),status=status))
            if status=='opened':
                horizon=int(f.loc[date,'window']) if mode=='window_installments' else 1
                active=dict(campaign=len(campaigns)+1,signal_date=date,window_at_signal=int(f.loc[date,'window']),horizon=horizon,budget=cash,daily_amount=cash/horizon,spent=0.,remaining=horizon,first_buy_date=pd.NaT,last_buy_date=pd.NaT)
                campaigns.append(active)
        nav=cash+units*r.adjusted_close
        if previous+inflow:twr*=nav/(previous+inflow)
        daily.append(dict(date=date,strategy=mode,cash=cash,fund=units*r.adjusted_close,nav=nav,contributions=contributed,twr=twr))
        previous=nav
        assert cash>=-1e-6 and np.isclose(cash+spent,contributed)
    d=pd.DataFrame(daily);t=pd.DataFrame(trades);c=pd.DataFrame(campaigns);s=pd.DataFrame(signals)
    assert np.isclose(t.units.sum()*p.adjusted_close.iloc[-1],d.fund.iloc[-1])
    assert np.isclose(t.amount.sum(),spent)
    assert len(s)==10
    for a in campaigns:
        tr=t[t.campaign==a['campaign']]
        expected=p.index[(p.index>a['signal_date'])&p.tradable][:a['horizon']]
        assert list(tr.date)==list(expected)
    result=dict(strategy=mode,signals=len(s),campaigns=len(c),buy_days=len(t),ignored_active=(s.status=='ignored_active').sum(),no_cash=(s.status=='no_cash').sum(),contributions=contributed,spent=spent,cash=cash,nav=nav,xirr_pct=100*xirr(flows+[(bt.END,nav)]),mdd_pct=100*(d.twr/d.twr.cummax()-1).min(),mean_cash_pct=100*(d.cash/d.nav).mean(),unfinished_campaigns=int((c.remaining>0).sum()))
    return result,d,t,c,s

def main():
    p,m=bt.load();f=bt.features(m)
    summaries=[]
    for mode in ['all_cash_once','window_installments']:
        result,d,t,c,s=simulate_installments(p,f,mode)
        summaries.append(result)
        for name,table in [('daily',d),('trades',t),('campaigns',c),('signals',s)]:table.to_csv(R/f'{mode}_{name}.csv',index=False)
    prior=pd.read_csv(R/'cash_rule_summary.csv')
    for name,select in [('DCA',prior.strategy=='DCA'),('dynamic_up_to_100k',(prior.strategy=='dynamic')&(prior.cash_rule=='up_to_100k'))]:
        row=prior.loc[select].iloc[0].to_dict();row['strategy']=name;row['buy_days']=row.pop('buys');summaries.append(row)
    summary=pd.DataFrame(summaries);summary.to_csv(R/'installment_summary.csv',index=False)
    print(summary[['strategy','signals','buy_days','spent','cash','nav','xirr_pct','mdd_pct']].round(3).to_string(index=False))
    print('\nCampaigns:');print(pd.read_csv(R/'window_installments_campaigns.csv').to_string(index=False))
    print('\nSignals:');print(pd.read_csv(R/'window_installments_signals.csv').to_string(index=False))
    return summary
if __name__=='__main__':main()

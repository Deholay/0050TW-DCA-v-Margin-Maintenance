"""Exploratory 0050 threshold-crossing backtest. Run locally from any directory.
Inputs: browser-extracted signal evidence from 1,200 consecutive istock daily rows;
Yahoo adjusted OHLC; TWSE 2026-10-14 final price. No network calls.
Only 2021-11-03 onward is used: the Yahoo pre-2014 scale inconsistency is excluded.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import brentq
ROOT=Path(__file__).resolve().parent
START=pd.Timestamp('2021-11-03'); END=pd.Timestamp('2026-10-14')
DEPOSIT=10000.; FEE=.001425; THRESHOLDS=[130,140,150,160]

def prepare_prices():
    raw=json.loads((ROOT/'raw/0050-yahoo2.json').read_text())['chart']['result'][0]
    idx=pd.to_datetime(raw['timestamp'],unit='s',utc=True).tz_convert('Asia/Taipei').tz_localize(None).normalize()
    prices=pd.DataFrame(raw['indicators']['quote'][0],index=idx)
    prices['adjusted_close']=raw['indicators']['adjclose'][0]['adjclose']
    prices=prices.loc[START:END].copy()
    # Official latest daily raw close repairs Yahoo's missing final day. No dividends
    # intervene since the last Yahoo adjusted/raw factor of exactly 1 on Oct 13.
    twse_path=ROOT.parent/'data/twse-20261012-15/prices-20261014.json'
    twse=json.loads(twse_path.read_text())
    table=next(t for t in twse['tables'] if '收盤價' in t.get('fields',[]))
    r=next(r for r in table['data'] if r[0]=='0050')
    assert prices.loc['2026-10-13','adjusted_close']/prices.loc['2026-10-13','close']==1
    for key,k in [('open',5),('high',6),('low',7),('close',8),('volume',2)]:
        prices.loc[END,key]=float(r[k].replace(',',''))
    prices.loc[END,'adjusted_close']=prices.loc[END,'close']
    assert not prices.index.duplicated().any()
    assert prices[['open','close','adjusted_close']].notna().all().all()
    prices['tradable']=prices['volume']>0
    suspended=prices.loc['2025-06-11':'2025-06-17']
    assert not suspended['tradable'].any()
    prices['adjusted_open']=prices['open']*prices['adjusted_close']/prices['close']
    assert prices['adjusted_close'].pct_change().abs().max()<.25
    prices.to_csv(ROOT/'prices_reviewed.csv',index_label='date')
    return prices

def load_signals(prices):
    signals=pd.read_csv(ROOT/'signals.csv',parse_dates=['previous_date','signal_date','next_market_date'])
    assert not signals.duplicated(['threshold','signal_date']).any()
    assert ((signals.previous_ratio>=signals.threshold)&(signals.ratio<signals.threshold)).all()
    assert (signals.previous_date<signals.signal_date).all()
    assert (signals.signal_date<signals.next_market_date).all()
    assert signals.groupby('threshold').size().to_dict()=={140:6,150:11,160:27}
    assert all(d in prices.index for d in signals.next_market_date)
    # Buy at the first tradable ETF open after the published signal.
    signals['execution_date']=[prices.index[(prices.index>d)&prices.tradable][0] for d in signals.signal_date]
    assert (signals.execution_date==signals.next_market_date).all()
    return signals

def xirr(flows):
    if not flows or not any(v<0 for d,v in flows) or not any(v>0 for d,v in flows): return np.nan
    t0=flows[0][0]; t=np.array([(d-t0).days/365.25 for d,v in flows]); v=np.array([v for d,v in flows])
    f=lambda rate: np.sum(v/np.power(1+rate,t))
    return brentq(f,-.9999,100)

def simulate(prices,signals,threshold=None,ticket=100000,cash_rate=0):
    label='DCA' if threshold is None else f'Below {threshold}%'
    schedule={r.execution_date:r for r in signals.itertuples() if r.threshold==threshold}
    cash=units=contributed=spent=0.; last_month=None; last_date=None; nav_before=0.; twr=1.
    daily=[];trades=[];skips=[];flows=[];purchaseflows=[]
    for date,r in prices.iterrows():
        if last_date is not None:cash*= (1+cash_rate)**((date-last_date).days/365.25)
        inflow=0.
        if date.to_period('M')!=last_month:
            inflow=DEPOSIT;cash+=inflow;contributed+=inflow;flows.append((date,-inflow));last_month=date.to_period('M')
        if threshold is None and inflow and r.tradable:
            amount=DEPOSIT
        elif threshold is not None and date in schedule:
            if cash+1e-7<ticket:
                skips.append({'strategy':label,'date':date,'available_cash':cash,'required':ticket});amount=0.
            else:amount=ticket
        else:amount=0.
        if amount:
            assert r.tradable and cash+1e-7>=amount
            new_units=amount/(1+FEE)/r.adjusted_open
            units+=new_units;cash-=amount;spent+=amount;purchaseflows.append((date,-amount))
            trades.append({'strategy':label,'date':date,'signal_date':schedule[date].signal_date if threshold else pd.NaT,
                           'signal_ratio':schedule[date].ratio if threshold else np.nan,'amount_including_fee':amount,
                           'adjusted_open':r.adjusted_open,'normalized_units':new_units,
                           'buy_fee':amount-amount/(1+FEE)})
        nav=cash+units*r.adjusted_close
        if nav_before+inflow>0:twr*=nav/(nav_before+inflow)
        daily.append({'date':date,'strategy':label,'cash':cash,'fund_value':units*r.adjusted_close,
                      'nav':nav,'contributed':contributed,'twr_index':twr})
        assert cash>=-1e-6
        nav_before=nav;last_date=date
    daily=pd.DataFrame(daily).set_index('date');nav=daily.nav.iloc[-1];fund=daily.fund_value.iloc[-1]
    mdd=(daily.twr_index/daily.twr_index.cummax()-1).min()
    result={'strategy':label,'threshold':threshold,'ticket':ticket if threshold else DEPOSIT,'cash_rate':cash_rate,
            'signals':len(schedule) if threshold else len(trades),'buys':len(trades),'skipped_cash':len(skips),
            'total_contributions':contributed,'purchase_spend':spent,'ending_cash':cash,'ending_fund':fund,'ending_nav':nav,
            'profit':nav-contributed,'total_profit_pct':(nav/contributed-1)*100,
            'portfolio_xirr_pct':xirr(flows+[(END,nav)])*100,
            'invested_only_xirr_pct':xirr(purchaseflows+[(END,fund)])*100 if fund else np.nan,
            'flow_adjusted_mdd_pct':mdd*100,'mean_cash_weight_pct':(daily.cash/daily.nav).mean()*100}
    # Independent accounting check of terminal purchased adjusted units.
    independent=sum(t['amount_including_fee']/(1+FEE)/t['adjusted_open']*prices.adjusted_close.iloc[-1] for t in trades)
    assert np.isclose(independent,fund)
    assert len(trades)+len(skips)==result['signals']
    return result,daily,pd.DataFrame(trades),pd.DataFrame(skips)

def event_study(prices,signals):
    events=[]
    for s in signals.itertuples():
        entry=prices.loc[s.execution_date,'adjusted_open'];row={'threshold':s.threshold,'signal_date':s.signal_date,'buy_date':s.execution_date,'ratio':s.ratio}
        future=prices.loc[s.execution_date:]
        row['to_end_return_pct']=(prices.adjusted_close.iloc[-1]/entry/(1+FEE)-1)*100
        row['worst_close_vs_entry_pct']=(future.adjusted_close.min()/entry/(1+FEE)-1)*100
        for year in [1,3,5]:
            target=s.execution_date+pd.DateOffset(years=year)
            eligible=prices.index[prices.index>=target]
            row[f'return_{year}y_pct']=(prices.loc[eligible[0],'adjusted_close']/entry/(1+FEE)-1)*100 if len(eligible) else np.nan
        events.append(row)
    return pd.DataFrame(events)

def main():
    prices=prepare_prices();signals=load_signals(prices)
    results=[];dailies=[];trades=[];skips=[]
    for threshold in [None]+THRESHOLDS:
        result,daily,trade,skip=simulate(prices,signals,threshold)
        results.append(result);dailies.append(daily);trades.append(trade);skips.append(skip)
    pd.DataFrame(results).to_csv(ROOT/'summary.csv',index=False)
    pd.concat(dailies).to_csv(ROOT/'daily_portfolios.csv')
    pd.concat(trades).to_csv(ROOT/'trades.csv',index=False)
    pd.concat(skips).to_csv(ROOT/'skipped_signals.csv',index=False)
    events=event_study(prices,signals);events.to_csv(ROOT/'signal_forward_returns.csv',index=False)
    sensitivity=[]
    for ticket in [10000,50000,100000]:
        for threshold in THRESHOLDS:
            sensitivity.append(simulate(prices,signals,threshold,ticket=ticket)[0])
    for threshold in [None]+THRESHOLDS:
        sensitivity.append(simulate(prices,signals,threshold,cash_rate=.02)[0])
    pd.DataFrame(sensitivity).to_csv(ROOT/'sensitivity.csv',index=False)
    print(pd.DataFrame(results).round(3).to_string(index=False))
    print('\nForward-return event study (overlapping events, not independent):')
    print(events.groupby('threshold').agg(events=('signal_date','count'),one_year_n=('return_1y_pct','count'),one_year_mean=('return_1y_pct','mean'),three_year_n=('return_3y_pct','count'),three_year_mean=('return_3y_pct','mean')).round(2))
    return prices,signals,pd.DataFrame(results)

if __name__=='__main__': main()

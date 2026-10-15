"""Reconstruct listed margin market value / actual aggregate margin debt from official raw data."""
from pathlib import Path
import gzip,json,datetime as dt
import pandas as pd
import numpy as np
R=Path(__file__).resolve().parent

def read(kind,day):
    for p in [R/'raw'/f'{kind}-{day}.json.gz',R/'raw'/f'{kind}-{day}.json',R.parent/'raw/twse'/f'{kind}-{day}.json.gz']:
        if p.exists():return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())
    return None

def numeric(s):
    try:return float(str(s).replace(',',''))
    except ValueError:return np.nan

def main(require_complete=True):
    cached={};rows=[];miss=[];carried=[];prices0050=[];missing_dates=[]
    start=dt.date(2004,2,11);end=dt.date(2010,12,31)
    for i in range((end-start).days+1):
        date=start+dt.timedelta(days=i)
        if date.weekday()==6:continue
        day=date.strftime('%Y%m%d');m=read('margin',day)
        if m is None:missing_dates.append(day);continue
        if m.get('stat')!='OK':continue
        p=read('prices',day)
        if p is None:missing_dates.append(day);continue
        assert m['date']==p['date']==day and p['stat']=='OK'
        summary=next(t for t in m['tables'] if any('融資金額' in str(r[0]) for r in t.get('data',[])))
        detail=next(t for t in m['tables'] if '代號' in t.get('fields',[]))
        pt=next(t for t in p['tables'] if '收盤價' in t.get('fields',[]))
        assert len({r[0] for r in pt['data']})==len(pt['data'])
        assert len({r[0] for r in detail['data']})==len(detail['data'])
        price_map={r[0]:numeric(r[8]) for r in pt['data']}
        for code,close in price_map.items():
            if np.isfinite(close) and close>0:cached[code]=(close,date)
        debt=numeric(next(r for r in summary['data'] if '融資金額' in r[0])[5])*1000
        lots=sum(numeric(r[6]) for r in detail['data'])
        official_lots=numeric(next(r for r in summary['data'] if r[0].startswith('融資('))[5])
        assert lots==official_lots,(day,lots,official_lots)
        value=same_day_value=unknown_lots=carry_lots=0.
        for r in detail['data']:
            code=r[0];q=numeric(r[6])
            if not q:continue
            close=price_map.get(code,np.nan)
            if np.isfinite(close) and close>0:same_day_value+=q*1000*close
            elif code in cached:
                close,old_date=cached[code];carry_lots+=q
                carried.append(dict(date=date,code=code,name=r[1],lots=q,carried_close=close,last_close_date=old_date,age_calendar_days=(date-old_date).days))
            else:
                unknown_lots+=q;miss.append(dict(date=date,code=code,name=r[1],lots=q));continue
            value+=q*1000*close
        row=dict(date=date,ratio=value/debt*100,same_day_ratio=same_day_value/debt*100,debt=debt,market_value=value,lots=lots,unknown_lots=unknown_lots,carry_lots=carry_lots,known_quantity_pct=(1-unknown_lots/lots)*100)
        rows.append(row)
        e=next((r for r in pt['data'] if r[0]=='0050'),None)
        assert e is not None,day
        prices0050.append(dict(date=date,open=numeric(e[5]),close=numeric(e[8]),volume=numeric(e[2])))
    daily=pd.DataFrame(rows);daily.to_csv(R/'margin_reconstructed.csv',index=False)
    pd.DataFrame(prices0050).to_csv(R/'0050_raw_prices.csv',index=False)
    pd.DataFrame(miss).to_csv(R/'unknown_prices.csv',index=False)
    pd.DataFrame(carried).to_csv(R/'carried_prices.csv',index=False)
    print('Trading days',len(daily),'missing calendar dates',len(missing_dates),'unknown observations',len(miss),'carried observations',len(carried))
    print(daily.groupby(daily.date.astype(str).str[:4]).agg(days=('date','size'),minimum_ratio=('ratio','min'),maximum_unknown_lots=('unknown_lots','max'),minimum_known_quantity_pct=('known_quantity_pct','min')).round(4))
    (R/'coverage.json').write_text(json.dumps(dict(missing_calendar_dates=missing_dates,trading_days=len(daily),unknown_observations=len(miss),carried_observations=len(carried)),ensure_ascii=False,indent=2))
    if require_complete:assert not missing_dates,'Full download is not complete; no backtest allowed.'
    return daily
if __name__=='__main__':
    import sys
    main('--partial' not in sys.argv)

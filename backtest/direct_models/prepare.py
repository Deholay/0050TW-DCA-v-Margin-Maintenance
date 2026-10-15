"""Review 2014+ price history; preserve official exceptional trading days."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parent
ROOT=OUT.parent

def main():
 j=json.loads((ROOT/'period_2020_2023/raw/finlab-margin.json').read_text())
 m=pd.Series(np.array(j['series']['v'])*100,index=pd.to_datetime(j['series']['d']),name='ratio').loc['2014-01-01':'2026-10-14']
 raw=json.loads((ROOT/'raw/0050-yahoo2.json').read_text())['chart']['result'][0]
 idx=pd.to_datetime(raw['timestamp'],unit='s',utc=True).tz_convert('Asia/Taipei').tz_localize(None).normalize()
 p=pd.DataFrame(raw['indicators']['quote'][0],index=idx)
 p['adjusted_close']=raw['indicators']['adjclose'][0]['adjclose']
 p=p.loc[m.index.min():m.index.max()].copy()
 # Reuse previous repairs before this pass; newer official checks take precedence.
 for fn in [ROOT/'period_2020_2023/prices_reviewed.csv',ROOT/'prices_reviewed.csv']:
  reviewed=pd.read_csv(fn,parse_dates=['date']).set_index('date')
  p=p.reindex(p.index.union(reviewed.index));p.loc[reviewed.index,reviewed.columns]=reviewed
 repairs=[];checks=[]
 for fn in sorted((OUT/'raw').glob('twse-0050-*.json')):
  data=json.loads(fn.read_text())
  assert data['stat']=='OK'
  for row in data['data']:
   yy,mm,dd=map(int,row[0].split('/'));dt=pd.Timestamp(yy+1911,mm,dd)
   official={key:float(row[k].replace(',',''))/4 for key,k in [('open',3),('high',4),('low',5),('close',6)]}
   if dt in p.index and pd.notna(p.loc[dt,'open']) and p.loc[dt,'volume']>0:
    error=max(abs(p.loc[dt,k]-v) for k,v in official.items())
    assert error<1e-5,(dt,error)
    checks.append(dict(date=dt,max_ohlc_error=error))
   elif dt in m.index:
    ordered=p.sort_index()
    before=ordered.loc[ordered.index<dt].iloc[-1];after=ordered.loc[ordered.index>dt].iloc[0]
    fa=before.adjusted_close/before.close;fb=after.adjusted_close/after.close
    assert abs(fa-fb)<1e-6,(dt,fa,fb)
    for k,v in official.items():p.loc[dt,k]=v
    p.loc[dt,'volume']=float(row[1].replace(',',''))*4
    p.loc[dt,'adjusted_close']=official['close']*fa
    repairs.append(dict(date=str(dt.date()),source=fn.name,adjustment_factor=fa,neighbor_factor_difference=abs(fa-fb)))
 p=p.sort_index();p['adjusted_open']=p.open*p.adjusted_close/p.close;p['tradable']=p.volume>0
 missing=m.index.difference(p.index)
 assert not len(missing),missing
 p=p.reindex(m.index)
 assert p[['open','close','adjusted_open','adjusted_close']].notna().all().all(), p[p[['open','close','adjusted_open','adjusted_close']].isna().any(axis=1)].to_string()
 assert m.index.is_unique and m.index.is_monotonic_increasing and m.notna().all()
 assert p.index.equals(m.index)
 assert p.loc[:'2024-12-31','tradable'].all(),p.loc[:'2024-12-31'].loc[lambda x: ~x.tradable.astype(bool)].to_string()
 assert p.adjusted_close.pct_change().abs().max()<.2
 p.to_csv(OUT/'prices.csv',index_label='date');m.to_csv(OUT/'margin.csv',index_label='date')
 pd.DataFrame(checks).to_csv(OUT/'official_price_checks.csv',index=False)
 (OUT/'data_quality.json').write_text(json.dumps(dict(rows=len(p),repairs=repairs,official_monthly_checks=len(checks),max_error=max(x['max_ohlc_error'] for x in checks),signal_source=j['source'],caveats=['FinLab combined TWSE+TPEx aggregate, not listed-only. Historical publication time and revisions are not independently archived.','Adjusted total-return fractional units, zero cash interest, fee 0.1425%; excludes minimum fee, spread and dividend tax.','Yahoo pre-2014 price scaling not used.']),ensure_ascii=False,indent=2))
 print('rows',len(p),'repairs',repairs,'official checks',len(checks))
if __name__=='__main__':main()

"""Fixed-rule period test; user-approved FinLab TWSE+TPEx signal alternative.
Reuses the original engine without changing it. No parameter search.
"""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'review_20261015'))
import window_roundtrip as eng
POLICIES=[('DCA_100',1.,False),('DCA_80_window_buy_only',.8,False),('DCA_80_window_roundtrip',.8,True),('diagnostic_tactical_100',0.,True),('diagnostic_buy_only_100',0.,False)]
checks=[]
def check(label,a,b,tol=1e-7):
 e=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
 assert e<tol,(label,e)
 checks.append(dict(check=label,max_error=e,tolerance=tol,passed=True))

def run(p,m,start,end,prefix):
 eng.base.START=pd.Timestamp(start)
 f=eng.make_features(m)
 assert f.loc[start:end,'ready'].all()
 assert p.loc[start:end].index.isin(f.index).all()
 results={name:eng.simulate(p,f,name,frac,exits,pd.Timestamp(end)) for name,frac,exits in POLICIES}
 summary=pd.DataFrame([eng.summarize(r,end) for r in results.values()]);dca=summary.iloc[0]
 summary['gap_vs_dca']=summary.final_nav-dca.final_nav
 summary['meets_objective']=(summary.final_nav>dca.final_nav+1e-6)&(summary.mdd_pct>=dca.mdd_pct-1e-9)
 for kind,i in [('daily',0),('trades',1),('campaigns',2),('signals',3)]:
  pd.concat([r[i] for r in results.values()],ignore_index=True).to_csv(HERE/f'{prefix}_{kind}.csv',index=False)
 summary.to_csv(HERE/f'{prefix}_summary.csv',index=False);f.to_csv(HERE/f'{prefix}_features.csv',index_label='date')
 ds={n:r[0].set_index('date') for n,r in results.items()};dc=ds['DCA_100']
 for n,full in [('DCA_80_window_buy_only','diagnostic_buy_only_100'),('DCA_80_window_roundtrip','diagnostic_tactical_100')]:
  for col in ['nav','cash','fund','contributions']:check(prefix+'_'+n+'_'+col,ds[n][col],.8*dc[col]+.2*ds[full][col])
 # Independent closed-form DCA holdings check.
 buys=p.loc[dc.index][dc.inflow>0]
 units=(10000/(1+eng.FEE)/buys.adjusted_open).sum()
 check(prefix+'_direct_dca_nav',dc.nav.iloc[-1],units*p.loc[dc.index[-1],'adjusted_close'])
 return results,summary,f

def main():
 j=json.loads((HERE/'raw/finlab-margin.json').read_text())
 series=j['series'];m=pd.Series(np.array(series['v'])*100,index=pd.to_datetime(series['d']),name='ratio').loc['2018-01-01':]
 assert m.index.is_unique and m.index.is_monotonic_increasing and m.notna().all()
 m.to_csv(HERE/'finlab_margin.csv',index_label='date')
 raw=json.loads((ROOT/'raw/0050-yahoo2.json').read_text())['chart']['result'][0]
 idx=pd.to_datetime(raw['timestamp'],unit='s',utc=True).tz_convert('Asia/Taipei').tz_localize(None).normalize()
 p=pd.DataFrame(raw['indicators']['quote'][0],index=idx)
 p['adjusted_close']=raw['indicators']['adjclose'][0]['adjclose']
 p=p.loc['2020-01-01':'2023-12-31'].copy()
 # Yahoo misses 2021-04-06. TWSE contemporaneous nominal OHLC / 4 matches
 # Yahoo split-adjusted OHLC. Neighbor adjustment factors agree (no dividend).
 official=json.loads((HERE/'raw/twse-0050-202104.json').read_text())
 rr=next(r for r in official['data'] if r[0]=='110/04/06')
 factor=p.loc['2021-04-01','adjusted_close']/p.loc['2021-04-01','close']
 check('repair_neighbor_adjustment_factor',factor,p.loc['2021-04-07','adjusted_close']/p.loc['2021-04-07','close'],1e-6)
 for key,k in [('open',3),('high',4),('low',5),('close',6)]:p.loc['2021-04-06',key]=float(rr[k].replace(',',''))/4
 p.loc['2021-04-06','volume']=float(rr[1].replace(',',''))*4
 p.loc['2021-04-06','adjusted_close']=p.loc['2021-04-06','close']*factor
 p=p.sort_index();p['adjusted_open']=p.open*p.adjusted_close/p.close;p['tradable']=p.volume>0
 assert p.index.is_unique and p[['open','close','adjusted_open','adjusted_close']].notna().all().all()
 assert p.index.equals(m.loc['2020':'2023'].index)
 assert p.tradable.all() and (p.adjusted_close.pct_change().abs().dropna()<.2).all()
 p.to_csv(HERE/'prices_reviewed.csv',index_label='date')
 results,summary,f=run(p,m,'2020-01-02','2023-12-29','period')
 assert (summary.contributions==480000).all()
 # Future data must not change past features or trades.
 for end in ['2020-12-31','2021-12-30','2022-12-30']:
  short=eng.make_features(m.loc[:end]);pd.testing.assert_frame_equal(short,f.loc[:end])
  d=eng.simulate(p.loc[:end],short,'DCA_80_window_roundtrip',.8,True,pd.Timestamp(end))[0]
  full=results['DCA_80_window_roundtrip'][0]
  for col in ['nav','cash','tactical_units','twr']:check('prefix_'+end+'_'+col,d[col],full.loc[full.date<=end,col])
 annual=[]
 for name,r in results.items():
  d=r[0];previous=1.
  for year,g in d.groupby(d.date.dt.year):
   end=g.iloc[-1];annual.append(dict(strategy=name,year=int(year),final_nav=end.nav,contributions=end.contributions,annual_twr_pct=100*(end.twr/previous-1),cash=end.cash));previous=end.twr
 pd.DataFrame(annual).to_csv(HERE/'annual.csv',index=False)
 # Equal-start, equal-warmup overlap comparison isolates signal-provider effect.
 op,om=eng.base.original.load();fm=m.reindex(om.index);assert fm.notna().all()
 overlap=[]
 for provider,mm in [('istock_listed',om),('finlab_combined',fm)]:
  _,s,_=run(op,mm,'2023-03-01','2026-10-14','overlap_'+provider);s['provider']=provider;overlap.append(s)
 pd.concat(overlap).to_csv(HERE/'overlap_summary.csv',index=False)
 old=pd.read_csv(ROOT/'review_20261015/window_roundtrip_summary.csv')
 for name in [x[0] for x in POLICIES[:3]]:
  for col in ['final_nav','xirr_pct','mdd_pct']:check('original_engine_'+name+'_'+col,overlap[0].set_index('strategy').loc[name,col],old.set_index('strategy').loc[name,col])
 diff=fm-om
 quality=dict(primary_price_rows=len(p),warmup_observations=int((m.index<p.index[0]).sum()),signal_start=str(m.index[0].date()),overlap_rows=len(om),overlap_mean_absolute_ratio_diff_pp=float(diff.abs().mean()),overlap_max_absolute_ratio_diff_pp=float(diff.abs().max()),overlap_correlation=float(fm.corr(om)),price_repair='2021-04-06: TWSE STOCK_DAY nominal OHLC divided by 4, common neighboring Yahoo dividend-adjustment factor.',missing_price_rows_after_repair=0,source='https://finlab.finance/api/market/margin-maintenance-ratio',definition=j['source'],parameters_optimized=False,monthly_inflow=10000,contribution_count=48,signal_caveat='User-approved TWSE+TPEx proxy; not the original listed-only series. Historical source revisions and survivorship treatment cannot be independently established from the aggregate API.',costs='Buy/sell fee 0.1425%, ETF sell tax 0.1%, zero cash yield; fractional adjusted total-return units; no minimum fee or spread.',warmup_note='Margin-only historical observations retained for rolling features; prices needed only during the tested portfolio interval.')
 files=[HERE/'raw/finlab-margin.json',HERE/'raw/twse-0050-202104.json',ROOT/'raw/0050-yahoo2.json',Path(eng.__file__),Path(__file__)]
 quality['sha256']={str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in files}
 (HERE/'methodology.json').write_text(json.dumps(quality,ensure_ascii=False,indent=2))
 pd.DataFrame(checks).to_csv(HERE/'validation.csv',index=False)
 print(summary[['strategy','final_nav','gap_vs_dca','xirr_pct','mdd_pct','buy_campaigns','sell_campaigns','cash','meets_objective']].round(4).to_string(index=False))
 print('OVERLAP');print(pd.concat(overlap)[['provider','strategy','final_nav','xirr_pct','mdd_pct']].round(4).to_string(index=False))
 print('CAMPAIGNS');print(results['DCA_80_window_roundtrip'][2].to_string(index=False));print(json.dumps(quality,ensure_ascii=False,indent=2));print('checks',len(checks))
if __name__=='__main__':main()

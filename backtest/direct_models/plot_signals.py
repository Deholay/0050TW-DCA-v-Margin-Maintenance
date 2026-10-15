"""Daily signal and order charts; preserve every observation and actual fill."""
from pathlib import Path
import json
import argparse
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'chart_inputs'

def main(output_dir=None):
 global OUT
 OUT=Path(output_dir).resolve() if output_dir else OUT
 OUT.mkdir(parents=True,exist_ok=True)
 f=pd.read_csv(ROOT/'features.csv',parse_dates=['date']).set_index('date').loc['2020':'2026-10-14']
 p=pd.read_csv(ROOT/'prices.csv',parse_dates=['date']).set_index('date').reindex(f.index)
 selected=json.loads((ROOT/'selected_before_tests.json').read_text())
 configs=selected+[dict(id='kappa2',family='kappa2',gate=True)]
 labels={c['id']: ('原式' if c['family']=='kappa2' else {'percentile':'①','depth':'②','target':'③'}[c['family']]+('嚴' if c['gate'] else '寬')) for c in configs}
 frames={};trades={};orders={};counts=[]
 full=f.copy();full['0050_adjusted_close']=p.adjusted_close;full['0050_adjusted_open']=p.adjusted_open
 amounts=pd.DataFrame(index=f.index);prices=pd.DataFrame({'收盤':p.adjusted_close},index=f.index)
 target=None;alltrades=[]
 for c in configs:
  cid=c['id'];label=labels[cid];prefix=ROOT/f'continuous_2020_2026_{cid}'
  daily=pd.read_csv(str(prefix)+'_daily.csv',parse_dates=['date']).set_index('date')
  t=pd.read_csv(str(prefix)+'_trades.csv',parse_dates=['date','signal_date'])
  o=pd.read_csv(str(prefix)+'_orders.csv',parse_dates=['signal_date'])
  frames[cid]=daily;trades[cid]=t;orders[cid]=o
  assert t.date.is_unique and o.signal_date.is_unique
  assert np.allclose(t.adjusted_open,p.loc[t.date,'adjusted_open'])
  prices[label]=t.set_index('date').adjusted_open.reindex(f.index)
  amounts[label]=o.set_index('signal_date').amount.reindex(f.index)
  full[cid+'_order']=amounts[label].fillna(0)
  full[cid+'_fill']=t.set_index('date').amount.reindex(f.index).fillna(0)
  for col in ['cash','episode_K','episode_D','pending']:full[cid+'_'+col]=daily[col]
  counts.append(dict(model=label,id=cid,order_days=len(o),buy_days=len(t),buy_spent=t.amount.sum()))
  tt=t.copy();tt['model']=label;tt['id']=cid;alltrades.append(tt)
  if cid=='target_open_h0.25':
   target=pd.DataFrame({'K':daily.episode_K,'T':daily.episode_K*f.depth/(.25+f.depth),'D':daily.episode_D,'C':daily.cash})/10000
 full['percentile_fraction']=.2*(1-f.percentile/.2).clip(lower=0)
 full['depth_fraction']=.2*f.depth/(.25+f.depth)
 full.to_csv(ROOT/'all_signal_values_daily.csv',index_label='date')
 pd.concat(alltrades).to_csv(ROOT/'all_model_buys.csv',index=False)
 pd.DataFrame(counts).to_csv(ROOT/'all_signal_counts.csv',index=False)
 manifests=[]
 common=['資料涵蓋2020/1/2–2026/10/14，每日資料未降採樣。','FinLab上市加上櫃維持率口徑；所有歷史分位數、波動估計排除當日與未來資料。','六個候選模型沿用2016–2019選定參數，另含原公式κ=2；不是48組全部參數組合。','各策略是獨立的每月一萬元純策略帳戶，只買不賣；尚未改成DCA加閒錢策略。']
 def chart(slug,title,desc,data,ylabel,right=(),rightlabel='',zero=False,defs=None):
  # Sparse missing values mean no order/fill, never a made-up price or amount.
  rows=[]
  for dt,row in data.iterrows():
   rr={'日':str(dt.date())}
   rr.update({k:round(float(v),3) for k,v in row.items() if pd.notna(v)})
   rows.append(rr)
  fields=list(data.columns)
  source=dict(label='既有回測逐日訊號與交易紀錄',files=[{'label':'direct_models/all_signal_values_daily.csv'},{'label':'direct_models/all_model_buys.csv'},{'label':'direct_models/run.py'}],metricDefinitions=[{'label':'讀圖','definition':desc}]+(defs or []),caveats=common)
  spec=dict(schemaVersion=1,id=slug,title=title,description=desc,chart=dict(type='line',x='日',y=fields[0],fields=fields,rightAxisFields=list(right),rightYAxisLabel=rightlabel,startAtZero=zero,showValues=False,showLegend=True,yLabel=ylabel,showXAxisLabel=False,valueDecimals=2),height=360,columns=['日']+fields,rows=rows,source=source,theme='codex-classic')
  (OUT/f'{slug}.json').write_text(json.dumps(spec,ensure_ascii=False,allow_nan=False,separators=(',',':')))
  receipt=dict(schemaVersion=1,items=[dict(id=slug,title=title,queries=[dict(id=slug+'-data',source=source,reportingPeriod='2020/1/2–2026/10/14',summary=f'共{len(rows)}個逐日觀察；未減少或聚合時間序列。')])])
  (OUT/f'{slug}-sources.json').write_text(json.dumps(receipt,ensure_ascii=False,allow_nan=False,separators=(',',':')))
  manifests.append(dict(slug=slug,rows=len(rows),fields=fields,marks={k:int(data[k].notna().sum()) for k in fields}))
 chart('signals-price','0050價格與全部模型實際買點','收盤線與買入開盤價均為含息／分割調整價格。嚴＝保留舊條件；寬＝移除舊條件。原式＝κ2。同日買點會重疊；無交易日留空。',prices,'調整價格（元）')
 chart('signals-maintenance','大盤維持率與進場門檻','m為當日維持率；q10、q20為先前252筆的第10、20百分位。低於門檻只是條件之一。',pd.DataFrame({'m':f.ratio,'q10':f.low_threshold,'q20':f.q20}),'維持率（%）')
 chart('signals-window','原進場訊號：下跌幅度與動態window','Δm＝當日維持率減W日前維持率；須低於−10個百分點，且m低於q10，舊嚴格條件才成立。W讀右軸。',pd.DataFrame({'Δm':f.dynamic_diff,'門檻':-10.,'W':f.window}),'維持率變化（百分點）',['W'],'W（交易日）')
 chart('signals-depth','百分位與低檔深度','p為維持率在先前252筆中的百分位，讀右軸；①在p<20%買入。d＝max(0,(q20−m)/IQR)；z為原公式的低檔深度，兩者尺度不同。',pd.DataFrame({'d':f.depth,'z':f.stress,'p':100*f.percentile}),'低檔深度（各自標準化）',['p'],'歷史百分位（%）',True)
 chart('signals-volatility','原公式：波動與資金分配時間','σ為先前60筆維持率日變化的標準差；σref為歷史參考波動。H＝clip(20(σ/σref)²,5,60)，讀右軸；κ2再加快投入。',pd.DataFrame({'σ':f.sigma,'σref':f.sigma_ref,'H':f.budget_horizon}),'波動（百分點／日變化）',['H'],'H（交易日）')
 chart('signals-intensity','當日訊號決定的現金使用比例','①、②為移除舊條件版本；原式已套用舊嚴格條件。縱軸是剩餘現金的百分比，不是總資產比重，也不是獲利機率。',pd.DataFrame({'①':full.percentile_fraction*100,'②':full.depth_fraction*100,'原式':f.k2_fraction.where(f.buy_state,0)*100}),'剩餘現金投入比例（%）',zero=True)
 chart('signals-target','③累計目標：預算、應投入、已投入與現金','③寬版本。K＝輪初固定預算；T＝當日累計目標；D＝當輪已投入；C＝現金。新入金不增加當輪K。非訊號輪次的K/T/D留空。',target,'金額（萬元）',zero=True)
 chart('signals-orders','所有模型的逐日買入指令金額','按訊號收盤日繪出成本內含的買入指令；下一可交易日開盤執行。沒有指令的日子留空。這不是每月加碼總額；原式＝κ2。',amounts/10000,'下單金額（萬元）',zero=True)
 (ROOT/'signal_charts_manifest.json').write_text(json.dumps(manifests,ensure_ascii=False,indent=2))
 print(pd.DataFrame(counts).round(2).to_string(index=False));print('daily rows',len(f))
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output-dir',type=Path,default=OUT,help='Destination for optional inline-chart JSON inputs; CSV results stay beside this script.')
 main(parser.parse_args().output_dir)

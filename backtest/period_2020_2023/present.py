from pathlib import Path
import json
import pandas as pd
import numpy as np
H=Path(__file__).resolve().parent
O=Path('/Users/fuchienzou/.codex/visualizations/2026/10/15/01a13e46-d89f-7042-bef2-dd086fde596e')
names={'DCA_100':'全額DCA','DCA_80_window_buy_only':'80/20只買不賣','DCA_80_window_roundtrip':'80/20買賣window'}
d=pd.read_csv(H/'period_daily.csv',parse_dates=['date']);d=d[d.strategy.isin(names)]
s=pd.read_csv(H/'period_summary.csv');s=s[s.strategy.isin(names)]
p=pd.read_csv(H/'prices_reviewed.csv',parse_dates=['date']).set_index('date')
# Official COVID-period OHLC and April-2021 neighboring prices verify scale.
validation=[]
for ym in ['202003','202104']:
 j=json.loads((H/f'raw/twse-0050-{ym}.json').read_text())
 for r in j['data']:
  parts=r[0].split('/');date=pd.Timestamp(f'{int(parts[0])+1911}-{parts[1]}-{parts[2]}')
  for col,k in [('open',3),('close',6)]:
   error=abs(p.loc[date,col]*4-float(r[k].replace(',','')))
   assert error<.00002,(date,col,error)
   validation.append(dict(date=str(date.date()),field=col,error=error))
pd.DataFrame(validation).to_csv(H/'official_price_validation.csv',index=False)
wide=d.pivot(index='date',columns='strategy',values='nav');rows=[]
for date,r in wide.iterrows():rows.append({'日期':str(date.date()),'只買不賣':round(r.DCA_80_window_buy_only-r.DCA_100,2),'買賣window':round(r.DCA_80_window_roundtrip-r.DCA_100,2),'DCA基準':0.})
source={'label':'2020–2023固定規則回測','files':[{'label':x} for x in ['period_daily.csv','period_summary.csv','period_campaigns.csv','prices_reviewed.csv','finlab_margin.csv']],
 'urls':[{'label':'FinLab歷史維持率','url':'https://finlab.finance/tools/margin-maintenance-ratio'},{'label':'FinLab公開資料','url':'https://finlab.finance/api/market/margin-maintenance-ratio'},{'label':'證交所0050月行情','url':'https://www.twse.com.tw/zh/trading/historical/stock-day.html'}],
 'metricDefinitions':[{'label':'相對DCA資產差額','definition':'同日策略總資產減全額DCA總資產；總資產包含0050持倉與現金，三組每月入金均為1萬元。'}],
 'filters':['2020年初從零開始，每月首個交易日入金','80%核心定投、20%機動；進出場沿用既定動態window','2018–2019年僅供訊號暖機；無交易與入金'],
 'caveats':['本次採使用者同意的FinLab上市＋上櫃合計維持率，與原本iStock上市口徑不同。','0050採股息與分割調整價格及零碎等價單位；買賣費率0.1425%、賣出稅0.1%；現金收益0%，未計最低手續費與價差。','歷史聚合序列的修訂與下市標的處理無法由此公開API獨立確認。'],
 'evidenceFlow':[{'kind':'calculation','title':'訊號與執行','detail':'低於前252日10%分位且W日下降超過10百分點時進場；高於90%分位且上升超過10百分點時出場。W依既定歷史波動公式取5–60日；收盤訊號後，下W個開盤等額買入或等股數賣出，活動期間不取消。'},{'kind':'validation','title':'回測檢核','detail':'974個交易日完整對齊，489筆暖機觀察，48次相同入金。49項帳務、線性部位拆分、未來資料截斷及原版結果重現檢核通過。'},{'kind':'validation','title':'價格核對與補值','detail':'以證交所2020年3月及2021年4月官方開收盤核對，分割比例一致；補上Yahoo缺漏的2021-04-06，該日採官方價格與前後一致的股息調整係數。'}]}
chart={'schemaVersion':1,'id':'period-2020-2023-gap','title':'2020–2023：相對全額DCA的資產差額','description':'零線為DCA；低於零代表策略資產較少。每月入金相同。',
 'chart':{'type':'line','x':'日期','y':'只買不賣','fields':['只買不賣','買賣window','DCA基準'],'rightAxisFields':[],'startAtZero':False,'showValues':False,'showLegend':True,'yLabel':'資產差額（元）','showXAxisLabel':False,'valueDecimals':0,
 'annotations':[{'id':'last-sale','kind':'range','at':'2023-02-09','end':'2023-04-17','label':'反向訊號：43日分批賣出'}]},'height':400,'columns':list(rows[0]),'rows':rows,'source':source,'theme':'codex-classic'}
(O/'period-2020-2023-gap.json').write_text(json.dumps(chart,ensure_ascii=False,allow_nan=False))
receipt={'schemaVersion':1,'items':[{'id':'wealth-gap','title':'2020–2023相對DCA資產差額','queries':[{'id':'daily-gaps','source':source,'columns':list(rows[0]),'rows':rows,'reportingPeriod':'2020-01-02 至 2023-12-29'}]}]}
(O/'period-2020-2023-gap-sources.json').write_text(json.dumps(receipt,ensure_ascii=False,allow_nan=False))
cols=['strategy','contributions','final_nav','gap_vs_dca','xirr_pct','mdd_pct','mdd_peak_date','mdd_trough_date','cash','buy_campaigns','sell_campaigns']
ss=s[cols].copy();ss.strategy=ss.strategy.map(names)
ov=pd.read_csv(H/'overlap_summary.csv');ov=ov[ov.strategy.isin(names)]
ov=ov[['provider','strategy','final_nav','xirr_pct','mdd_pct']].copy();ov.strategy=ov.strategy.map(names)
c=pd.read_csv(H/'period_campaigns.csv');c=c[c.strategy=='DCA_80_window_roundtrip'][['side','signal_date','window','frozen_amount','first_fill','last_fill','cash_change']]
receipt2={'schemaVersion':1,'items':[
 {'id':'period-results','title':'報酬與最大回撤未同時達標','queries':[{'id':'summary','source':dict(source,metricDefinitions=[{'label':'年化報酬XIRR','definition':'依實際每月入金日期與期末總資產計算資金加權年化報酬。'},{'label':'最大回撤','definition':'用排除外部入金影響的每日時間加權報酬指數計算，包含起始指數1；入金日以開盤前後分段。'}]),'columns':cols,'rows':ss.to_dict('records'),'reportingPeriod':'2020-01-02 至 2023-12-29'}]},
 {'id':'campaign-explanation','title':'恐慌期間買入與2023年出場後未再進場','queries':[{'id':'campaigns','source':{'label':'機動部位成交紀錄','files':[{'label':'period_campaigns.csv'},{'label':'prices_reviewed.csv'}],'metricDefinitions':[{'label':'出場後價格變化','definition':'2023-04-17最後一筆賣出當日調整收盤至2023-12-29調整收盤上漲14.19%。此區間不含分批賣出期間。'}],'caveats':['2020年3月策略僅累積6000元機動資金，因本測試從2020年1月零資產開始；不代表已有長期部位者的實際交易金額。']},'columns':list(c.columns),'rows':c.to_dict('records'),'reportingPeriod':'2020-01-02 至 2023-12-29'}]},
 {'id':'provider-check','title':'重疊期間換資料後仍輸DCA，但交易細節不同','queries':[{'id':'provider-comparison','source':{'label':'兩種維持率口徑的重疊測試','files':[{'label':'overlap_summary.csv'},{'label':'methodology.json'}],'metricDefinitions':[{'label':'口徑核對','definition':'同價格、同起訖、同暖機日曆與同規則；只替換iStock上市維持率與FinLab上市櫃合計維持率。'}],'caveats':['重疊1200筆維持率相關係數0.9960、平均絕對差0.8893百分點；相關高不等於交易完全相同，也無法證明2020–2023上市單獨口徑的績效。'],'evidenceFlow':[{'kind':'validation','title':'原版重現','detail':'使用iStock資料時，原回測三組期末資產、XIRR、最大回撤重現差異小於1e-7。'}]},'columns':list(ov.columns),'rows':ov.to_dict('records'),'reportingPeriod':'2023-03-01 至 2026-10-14'}]}]}
(O/'period-2020-2023-results-sources.json').write_text(json.dumps(receipt2,ensure_ascii=False,allow_nan=False))
print('official price checks',len(validation),'max error',max(x['error'] for x in validation));print('chart rows',len(rows))

"""Render inputs: observed 0050 prices, actual tactical buys, fixed W intervals.

Use the latest 80/20 symmetric-window strategy. No recomputed or optimized
signals, no substituted close-price fills. The price axis is dividend/split
adjusted, consistent with the backtest. Period-minimum closes are descriptive
and retrospective, not executable fills or a claimed bottom-prediction rate.
"""
from pathlib import Path
import json
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = Path('/Users/fuchienzou/.codex/visualizations/2026/10/15/01a13e46-d89f-7042-bef2-dd086fde596e')
STRATEGY = 'DCA_80_window_roundtrip'
PRICE = '0050含息調整收盤'
BUY = '機動分批買入價'


def main():
    p=pd.read_csv(HERE.parent/'prices_reviewed.csv',parse_dates=['date']).set_index('date')
    c=pd.read_csv(HERE/'window_roundtrip_campaigns.csv',parse_dates=['signal_date','first_fill','last_fill'])
    c=c[(c.strategy==STRATEGY)&(c.side=='buy')].copy()
    c['buy_number']=range(1,len(c)+1)
    t=pd.read_csv(HERE/'window_roundtrip_trades.csv',parse_dates=['date','signal_date'])
    t=t[(t.strategy==STRATEGY)&(t.sleeve=='tactical')&(t.side=='buy')].copy()
    s=pd.read_csv(HERE/'window_roundtrip_signals.csv',parse_dates=['signal_date'])
    s=s[(s.strategy==STRATEGY)&(s.side=='buy')].set_index('signal_date')
    assert len(c)==6 and len(t)==97 and not t.date.duplicated().any()
    assert (t.date>t.signal_date).all()
    assert all(abs(r.adjusted_open-p.loc[r.date,'adjusted_open'])<1e-9 for r in t.itertuples())
    fills=t.set_index('date')
    stats=[]
    for a in c.itertuples():
        tr=t[t.campaign==a.campaign]
        q=p.loc[a.first_fill:a.last_fill]
        q=q[q.tradable]
        bottom=q.adjusted_close.idxmin()
        avg=tr.gross_value.sum()/tr.units.sum()
        assert len(tr)==a.window and bottom in set(tr.date)
        stats.append(dict(buy_number=a.buy_number,campaign=a.campaign,
                          signal_date=str(a.signal_date.date()),first_fill=str(a.first_fill.date()),
                          last_fill=str(a.last_fill.date()),window=a.window,
                          budget=a.frozen_amount,buy_average=avg,
                          period_min_close_date=str(bottom.date()),
                          period_min_close=q.loc[bottom,'adjusted_close'],
                          average_above_period_min_pct=100*(avg/q.loc[bottom,'adjusted_close']-1),
                          first_open_to_period_min_pct=100*(q.loc[bottom,'adjusted_close']/tr.adjusted_open.iloc[0]-1),
                          buy_at_min_close_date=tr.loc[tr.date==bottom,'adjusted_open'].iloc[0],
                          budget_fraction_on_min_day=1/a.window))
    metrics=pd.DataFrame(stats)
    metrics.to_csv(HERE/'entry_window_analysis.csv',index=False)
    # Both dates were checked against their full calendar month's observed closes.
    bottoms=['2024-08-05','2025-04-09']
    for d in bottoms:
        month=p.loc[d[:7]]
        assert str(month.adjusted_close.idxmin().date())==d

    chart_windows=[('entry-windows-overview','0050：六輪機動進場與分批買入','2023-03-01','2026-10-14'),
                   ('entry-windows-detail','放大看：2024年8月與2025年4月低點','2024-07-01','2025-05-30')]
    for slug,title,start,end in chart_windows:
        q=p.loc[start:end]
        rows=[]
        for date,r in q.iterrows():
            if not r.tradable:
                continue
            status=s.loc[date,'status'] if date in s.index else ''
            signal=({'opened':'啟動買入','ignored_active':'已有分批計畫，略過新訊號','no_cash':'無可用現金'}[status] if status else '')
            rows.append({'日期':str(date.date()),PRICE:round(r.adjusted_close,4),
                         BUY:round(fills.loc[date,'adjusted_open'],4) if date in fills.index else None,
                         '進場訊號':signal,
                         '當日買入金額':round(-fills.loc[date,'cash_change'],2) if date in fills.index else None})
        annotations=[]
        selected=c[(c.first_fill>=pd.Timestamp(start))&(c.last_fill<=pd.Timestamp(end))]
        for a in selected.itertuples():
            annotations.append(dict(id=f'buy-window-{a.buy_number}',kind='range',
                                    at=str(a.first_fill.date()),end=str(a.last_fill.date()),
                                    label=f'{a.buy_number}｜{a.signal_date:%m/%d}訊號 → {a.first_fill:%m/%d}–{a.last_fill:%m/%d}買入（W={a.window}）'))
        for d in bottoms:
            if start<=d<=end:
                annotations.append(dict(id='month-low-'+d,kind='point',field=PRICE,at=d,
                                        label=d[5:].replace('-','/')+' 當月最低收盤'))
        assert len(annotations)<=8
        source={'label':'0050價格與機動買入紀錄',
                'files':[{'label':x} for x in ['prices_reviewed.csv','window_roundtrip_trades.csv','window_roundtrip_campaigns.csv','window_roundtrip_signals.csv','entry_window_analysis.csv']],
                'metricDefinitions':[{'label':'價格與成交','definition':'0050曲線採股息與分割調整收盤價；機動買入曲線只在實際買入日顯示當日調整開盤價。'},{'label':'買入window','definition':'陰影涵蓋每批第一筆至最後一筆實際買入日，標籤記錄觸發訊號日及凍結W。'}],
                'filters':['策略：每月80%核心定投、20%機動動態window買賣','圖上僅標示機動買入，不含核心每月定投或賣出','價格只繪ETF可交易日'],
                'caveats':['調整價格是與回測一致的含息等價價格，並非當年未調整的掛牌價格。','當月或分批期間最低收盤價均為事後觀察，不代表當時可預知最低價，也不是實際開盤成交價。','每天分批自然會涵蓋該批期間最低收盤日，不可將此解讀為六輪皆精準抓底。'],
                'evidenceFlow':[{'kind':'validation','title':'交易價格與日期核對','detail':'六批共97筆機動買入，逐筆調整開盤價與價格檔一致；所有買入均晚於訊號日，交易筆數等於各批W。'},{'kind':'validation','title':'兩個月內低點核對','detail':'2024-08-05及2025-04-09分別為當月最低調整收盤日，兩日均有實際機動買入。'}]}
        chart={'schemaVersion':1,'id':slug,'title':title,
               'description':'陰影為買入window；買入價採當日開盤，與收盤價分開顯示。',
               'chart':{'type':'line','x':'日期','y':PRICE,'fields':[PRICE,BUY],
                        'rightAxisFields':[],'startAtZero':False,'showValues':False,
                        'showLegend':True,'yLabel':'含息調整價格（元）','showXAxisLabel':False,
                        'valueDecimals':2,'annotations':annotations},
               'height':440,'columns':['日期',PRICE,BUY,'進場訊號','當日買入金額'],
               'rows':rows,'source':source,'theme':'codex-classic'}
        (OUT/(slug+'.json')).write_text(json.dumps(chart,ensure_ascii=False,indent=2,allow_nan=False))
        dates={row['日期'] for row in rows}
        for a in annotations:
            assert a['at'] in dates and (a['kind']!='range' or a['end'] in dates)
        # The same safe chart-scope rows and definitions support its receipt.
        receipt={'schemaVersion':1,'items':[{'id':slug+'-evidence','title':title,'queries':[
            {'id':slug+'-data','source':source,'columns':chart['columns'],'rows':rows,
             'reportingPeriod':start+' 至 '+end,
             'methods':[{'language':'calculation','code':'成交均價 = 該輪買進總成交金額（不含費）÷ 買入總含息等價單位；距期間最低收盤 = 成交均價 ÷ 分批期間最低調整收盤價 − 1。'}]},
            {'id':slug+'-campaigns','source':{'label':'分批期間成本核對','files':[{'label':'entry_window_analysis.csv'}],
                'caveats':['均價不含買入手續費；比較基準為各批既定分批期間的最低收盤，並非任意前後窗的最低價。']},
             'reportingPeriod':start+' 至 '+end,
             'columns':['buy_number','signal_date','first_fill','last_fill','window','buy_average','period_min_close_date','period_min_close','average_above_period_min_pct','first_open_to_period_min_pct'],
             'rows':metrics[metrics.campaign.isin(selected.campaign)].to_dict('records')}
        ]}]}
        (OUT/(slug+'-sources.json')).write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False))
        print(slug,len(rows),'rows',len(annotations),'annotations')
    print(metrics.round(6).to_string(index=False))


if __name__=='__main__':
    main()

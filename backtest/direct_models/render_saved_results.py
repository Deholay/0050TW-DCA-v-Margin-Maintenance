"""Render portable README figures from saved, validated backtest outputs.

Run from any directory. No network, signal recomputation or parameter selection.
Figures retain all daily observations; labels use English for portable fonts.
"""
from pathlib import Path
import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'margin-strategy-matplotlib'))
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT=Path(__file__).resolve().parent
FIG=ROOT/'figures'
LABELS={
 'DCA':'DCA',
 'kappa2':'Original kappa=2',
 'percentile_open_p00.2_a0.2_r1':'1 Percentile / open',
 'percentile_strict_p00.2_a0.2_r1':'1 Percentile / strict',
 'depth_open_a0.2_h0.25':'2 Depth / open',
 'depth_strict_a0.2_h0.25':'2 Depth / strict',
 'target_open_h0.25':'3 Target / open',
 'target_strict_h0.25':'3 Target / strict',
}
COLORS={'DCA':'#30343b','kappa2':'#7852a3',
 'percentile_open_p00.2_a0.2_r1':'#b8860b','percentile_strict_p00.2_a0.2_r1':'#b8860b',
 'depth_open_a0.2_h0.25':'#286ba6','depth_strict_a0.2_h0.25':'#286ba6',
 'target_open_h0.25':'#d26c28','target_strict_h0.25':'#d26c28'}
STYLE=lambda cid:'--' if 'strict' in cid else '-'
sources=set()
def read(name,dates=None,index=None):
 sources.add(name)
 return pd.read_csv(ROOT/name,parse_dates=dates,index_col=index)
def daily(cid):return read(f'continuous_2020_2026_{cid}_daily.csv',['date'],'date')
def tidy(ax,ylabel):
 ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.2)
 ax.spines[['top','right']].set_visible(False)
def date_axis(ax):
 ax.xaxis.set_major_locator(mdates.YearLocator())
 ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
 ax.set_xlim(pd.Timestamp('2020-01-02'),pd.Timestamp('2026-10-14'))
def save(fig,name):
 fig.savefig(FIG/name,dpi=170,bbox_inches='tight',facecolor='white')
 plt.close(fig)

def main():
 FIG.mkdir(exist_ok=True)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.titleweight':'medium','axes.labelsize':10,'legend.fontsize':9,'lines.linewidth':1.4})
 summary=read('continuous_2020_2026_summary.csv').set_index('id').loc[list(LABELS)]
 f=read('features.csv',['date'],'date').loc['2020':'2026-10-14']
 p=read('prices.csv',['date'],'date').loc[f.index]
 ledgers={cid:daily(cid) for cid in LABELS}
 for cid,d in ledgers.items():
  assert len(d)==1647 and d.index.equals(f.index)
  assert np.isclose(d.nav.iloc[-1],summary.loc[cid,'final_nav'])
  dd=100*(d.twr/d.twr.cummax().clip(lower=1)-1)
  assert np.isclose(dd.min(),summary.loc[cid,'mdd_pct'])

 # Full selected-model comparison; all bars start at zero.
 fig,axes=plt.subplots(1,3,figsize=(14,5.4),sharey=True,layout='constrained')
 fig.suptitle('0050 | Continuous account, 2020-01-02 to 2026-10-14',fontsize=15)
 ys=np.arange(len(summary))
 for ax,col,scale,title,unit in zip(axes,['final_nav','xirr_pct','mdd_pct'],[1e-4,1,-1],['Ending wealth','Annualized cash-flow return','Maximum drawdown'],['TWD 10,000','XIRR (%)','Decline (%) | lower is better']):
  vals=summary[col]*scale
  ax.barh(ys,vals,color=[COLORS[c] for c in summary.index],alpha=.85,height=.68)
  for i,(cid,v) in enumerate(zip(summary.index,vals)):
   if 'strict' in cid:ax.patches[i].set_hatch('///')
   ax.text(v+vals.max()*.015,i,f'{v:,.2f}',va='center',fontsize=9)
  ax.set_xlim(0,vals.max()*1.22);ax.set_title(title);ax.set_xlabel(unit)
  ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True);ax.spines[['top','right','left']].set_visible(False)
 axes[0].set_yticks(ys,[LABELS[c] for c in summary.index]);axes[0].invert_yaxis()
 save(fig,'01-model-comparison.png')

 # Focus on DCA and the three principal strategies to keep paths legible.
 focus=['DCA','kappa2','depth_open_a0.2_h0.25','target_open_h0.25']
 fig,axes=plt.subplots(3,1,figsize=(13,10),sharex=True,layout='constrained')
 fig.suptitle('Account paths | TWD 10,000 monthly, TWD 820,000 contributed',fontsize=15)
 for cid in focus:
  d=ledgers[cid]
  for ax,values in zip(axes,[d.nav/10000,100*(d.twr/d.twr.cummax().clip(lower=1)-1),100*d.cash/d.nav]):
   ax.plot(d.index,values,label=LABELS[cid],color=COLORS[cid])
 axes[0].plot(f.index,ledgers['DCA'].contributions/10000,color='#888888',ls=':',label='Cumulative contributions')
 for ax,label in zip(axes,['Wealth (TWD 10,000)','TWR drawdown (%)','Cash / total wealth (%)']):tidy(ax,label);date_axis(ax)
 axes[0].legend(ncol=3,loc='upper left');axes[0].set_ylim(bottom=0);axes[2].set_ylim(0,105)
 save(fig,'02-account-paths.png')

 # Signal inputs, never retrospectively computed future-return labels.
 fig,axes=plt.subplots(5,1,figsize=(14,15),sharex=True,layout='constrained')
 fig.suptitle('Daily signal inputs | 1,647 observations; no downsampling',fontsize=15)
 axes[0].plot(p.index,p.adjusted_close,color='#333333',label='0050 adjusted close')
 for cid,mark in [('depth_open_a0.2_h0.25','o'),('target_open_h0.25','^'),('kappa2','x')]:
  t=read(f'continuous_2020_2026_{cid}_trades.csv',['date','signal_date'])
  axes[0].scatter(t.date,t.adjusted_open,s=20,marker=mark,c=COLORS[cid],label=LABELS[cid]+' fills',alpha=.72,zorder=3)
 axes[0].set_title('Price and actual next-open fills; coincident buys overlap',loc='left')
 for field,label,color,style in [('ratio','Market maintenance','#333333','-'),('low_threshold','Prior q10','#286ba6','--'),('q20','Prior q20','#d26c28','--'),('q50','Prior median','#777777',':')]:
  axes[1].plot(f.index,f[field],label=label,color=color,ls=style)
 axes[1].fill_between(f.index,f.q25,f.q75,color='#aaaaaa',alpha=.13,label='Prior q25-q75')
 axes[2].plot(f.index,f.dynamic_diff,color='#286ba6',label='Maintenance change over W')
 axes[2].axhline(-10,color='#555555',ls='--',label='Entry threshold: -10 pp')
 right=axes[2].twinx();right.plot(f.index,f.window,color='#b8860b',alpha=.65,label='W');right.set_ylabel('W (trading observations)');right.set_ylim(0,65)
 axes[3].plot(f.index,f.depth,label='Depth d (IQR scale)',color='#286ba6')
 axes[3].plot(f.index,f.stress,label='Original stress z',color='#7852a3')
 right=axes[3].twinx();right.plot(f.index,100*f.percentile,color='#b8860b',alpha=.5,lw=.9,label='Percentile p');right.set_ylabel('Percentile p (%)');right.set_ylim(0,105)
 axes[4].plot(f.index,f.sigma,label='Prior volatility sigma',color='#286ba6')
 axes[4].plot(f.index,f.sigma_ref,label='Reference volatility',color='#777777',ls='--')
 right=axes[4].twinx();right.plot(f.index,f.budget_horizon,color='#b8860b',alpha=.65,label='H');right.set_ylabel('H (trading observations)');right.set_ylim(0,65)
 for ax,label in zip(axes,['Adjusted price (TWD)','Maintenance (%)','Change (percentage points)','Normalized depth','Volatility (percentage points)']):
  tidy(ax,label);date_axis(ax);ax.legend(loc='upper left',ncol=2,framealpha=.85)
 for ax in [axes[0],axes[3],axes[4]]:ax.set_ylim(bottom=0)
 save(fig,'03-signal-inputs.png')

 # Signal-driven budgets and all seven actual model order streams.
 fig,axes=plt.subplots(3,1,figsize=(14,11),sharex=True,layout='constrained')
 fig.suptitle('Cash deployment | signal strength, episode budget, actual orders',fontsize=15)
 fractions={
  'percentile_open_p00.2_a0.2_r1':.2*(1-f.percentile/.2).clip(lower=0),
  'depth_open_a0.2_h0.25':.2*f.depth/(.25+f.depth),
  'kappa2':f.k2_fraction.where(f.buy_state,0)}
 for cid,values in fractions.items():axes[0].plot(f.index,100*values,label=LABELS[cid],color=COLORS[cid])
 axes[0].set_title('Fraction of remaining cash scheduled after the close',loc='left')
 d=ledgers['target_open_h0.25']
 for values,label,color,style in [(d.cash,'Available cash C','#54835b','-'),(d.episode_K,'Frozen budget K','#666666','--'),(d.episode_K*f.depth/(.25+f.depth),'Cumulative target T','#d26c28','-'),(d.episode_D,'Already spent D','#286ba6',':')]:
  axes[1].plot(f.index,values/10000,label=label,color=color,ls=style)
 axes[1].set_title('Model 3 / open: deposits do not increase K during an episode',loc='left')
 for cid in LABELS:
  if cid=='DCA':continue
  o=read(f'continuous_2020_2026_{cid}_orders.csv',['signal_date'])
  # Stems ensure separate orders are not joined across no-signal days.
  axes[2].vlines(o.signal_date,0,o.amount/10000,color=COLORS[cid],linestyle=STYLE(cid),alpha=.4,lw=.8)
  axes[2].scatter(o.signal_date,o.amount/10000,s=17,marker='x' if 'strict' in cid else 'o',color=COLORS[cid],label=LABELS[cid],alpha=.8)
 axes[2].set_title('Every order on its signal date; fills occur at the next tradable open',loc='left')
 for ax,label in zip(axes,['Remaining cash (%)','TWD 10,000','Order budget (TWD 10,000)']):
  tidy(ax,label);date_axis(ax);ax.set_ylim(bottom=0);ax.legend(loc='upper left',ncol=3,framealpha=.85)
 save(fig,'04-cash-deployment.png')

 manifest=dict(period=['2020-01-02','2026-10-14'],daily_rows=len(f),no_downsampling=True,rendered_with=dict(python=__import__('sys').version.split()[0],pandas=pd.__version__,numpy=np.__version__,matplotlib=matplotlib.__version__),sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sorted(sources)},figures={x.name:dict(bytes=x.stat().st_size,sha256=hashlib.sha256(x.read_bytes()).hexdigest()) for x in sorted(FIG.glob('*.png'))})
 (ROOT/'saved_results_manifest.json').write_text(json.dumps(manifest,indent=2))
 print('Saved',len(manifest['figures']),'figures; verified all eight terminal wealth and MDD values.')
if __name__=='__main__':main()

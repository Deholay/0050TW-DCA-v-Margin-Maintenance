"""Prespecified symmetric dynamic-window exit for the 20% tactical sleeve.

2023-03-01..2026-10-14, monthly external deposits TWD 10,000 at the first
tradable open. Core receives 80% and buys monthly forever; tactical receives
20% and retains all sale proceeds. These are contribution weights, not NAV
targets. No borrowing, no parameter search, zero cash yield.

Use the ORIGINAL dynamic entry: ratio < prior-252-day q10 AND
ratio[t]-ratio[t-W] < -10 percentage points. Symmetric exit: ratio > prior
252-day q90 AND the same difference > +10pp. Both trigger only on false to
true transitions. W uses run_dynamic.features unchanged: round(20 *
(prior sigma-reference / prior sigma)^2), clipped to 5..60.

At signal CLOSE, freeze W and all tactical cash (buy) or all tactical
total-return equivalent units (sell). Execute equal cash/units at the next W
tradable OPENS. An active campaign is never cancelled or resized. Ignore
new signals during an active campaign; after its last fill, that day's
closing signal may open the next. New monthly money waits outside the
frozen budget. No expiry rule is added.

Buy and sell fees 0.1425% each, sell ETF tax 0.1%. Fractional adjusted-price
units idealize dividend reinvestment and stock splits, as in prior studies;
frozen equivalent units are not literally frozen broker share counts across
dividends. No minimum fee, bid/ask spread, market impact, or investor tax.
Cash is TRADE-DATE economic cash (including sale receivables), not a bank
balance. No simultaneous buy/sell campaigns: any reinvested sale precedes
the purchase, hence its settlement precedes that purchase's settlement.
Broker prepayment/order limits and the settlement calendar are not modeled.

Compare DCA, 80/20 window buy-only, and the identical rule plus symmetric
window exits. A full tactical account verifies linear sleeve accounting;
it is a diagnostic, not a selected fourth candidate. TWR splits deposit
dates at the open, marking yesterday's holdings at that open before adding
the external flow. MDD uses daily-close TWR including initial index 1.
The earlier start-of-day approximation is retained as twr_legacy for audit.
Nested year-end results are not out-of-sample validation.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import allocation_study as base

HERE = Path(__file__).resolve().parent
FEE = base.FEE
SELL_TAX = .001
EPS = 1e-8


def make_features(m):
    f = base.original.features(m)
    f['high_threshold'] = m.shift(1).rolling(252).quantile(.9)
    for side, state in [
        ('buy', (m < f.low_threshold) & (f.dynamic_diff < -10) & f.ready),
        ('sell', (m > f.high_threshold) & (f.dynamic_diff > 10) & f.ready),
    ]:
        f[side + '_state'] = state
        f[side + '_event'] = state & ~state.shift(1, fill_value=False)
    assert not (f.buy_state & f.sell_state).any()
    return f


def simulate(p, f, name, core_fraction, exits, end=base.END):
    q = p.loc[base.START:end]
    cash = core = tactical = contributed = previous = 0.
    tactical_basis = realized = buy_spent = sale_net = fees = tax = 0.
    tactical_contributed = 0.
    twr = twr_legacy = 1.
    last_month = None
    active = None
    rows, trades, campaigns, signals = [], [], [], []

    for date, r in q.iterrows():
        inflow = 0.
        value_before_open_flow = cash + (core + tactical) * r.adjusted_open if r.tradable else np.nan
        if r.tradable and last_month != date.to_period('M'):
            inflow = base.DEPOSIT
            contributed += inflow
            core_budget = inflow * core_fraction
            if core_budget:
                gross = core_budget / (1 + FEE)
                added = gross / r.adjusted_open
                fee = core_budget - gross
                core += added
                fees += fee
                trades.append(dict(strategy=name, date=date, sleeve='core', side='buy',
                                   campaign=0, signal_date=pd.NaT, units=added,
                                   adjusted_open=r.adjusted_open, gross_value=gross,
                                   fee=fee, tax=0., cash_change=-core_budget,
                                   realized_pnl=0.))
            cash += inflow * (1 - core_fraction)
            tactical_contributed += inflow * (1 - core_fraction)
            last_month = date.to_period('M')

        if active is not None and r.tradable:
            a = active
            assert date > a['signal_date']
            amount = a['frozen_amount'] - a['executed_amount'] if a['remaining'] == 1 else a['per_day']
            assert amount > 0
            if a['side'] == 'buy':
                assert amount <= cash + EPS
                gross = amount / (1 + FEE)
                units = gross / r.adjusted_open
                fee, levy, pnl = amount - gross, 0., 0.
                cash_change = -amount
                cash -= amount
                tactical += units
                tactical_basis += amount
                buy_spent += amount
            else:
                units = amount
                assert units <= tactical + EPS
                gross = units * r.adjusted_open
                fee, levy = gross * FEE, gross * SELL_TAX
                cash_change = gross - fee - levy
                cost = tactical_basis * units / tactical
                pnl = cash_change - cost
                tactical_basis -= cost
                tactical -= units
                if abs(tactical) < EPS:
                    tactical = tactical_basis = 0.
                realized += pnl
                cash += cash_change
                sale_net += cash_change
            fees += fee
            tax += levy
            trades.append(dict(strategy=name, date=date, sleeve='tactical', side=a['side'],
                               campaign=a['campaign'], signal_date=a['signal_date'],
                               units=units, adjusted_open=r.adjusted_open, gross_value=gross,
                               fee=fee, tax=levy, cash_change=cash_change, realized_pnl=pnl))
            a['executed_amount'] += amount
            a['cash_change'] += cash_change
            a['realized_pnl'] += pnl
            a['remaining'] -= 1
            a['last_fill'] = date
            if pd.isna(a['first_fill']):
                a['first_fill'] = date
            if a['remaining'] == 0:
                assert np.isclose(a['executed_amount'], a['frozen_amount'])
                active = None

        # Close-only information cannot influence the current open.
        if core_fraction < 1:
            for side in ('buy', 'sell'):
                if not f.loc[date, side + '_event']:
                    continue
                available = cash if side == 'buy' else tactical
                status = ('disabled' if side == 'sell' and not exits else
                          'ignored_active' if active is not None else
                          'no_cash' if side == 'buy' and available < EPS else
                          'no_units' if side == 'sell' and available < EPS else 'opened')
                window = int(f.loc[date, 'window'])
                signals.append(dict(strategy=name, signal_date=date, side=side,
                                    window=window, available=available, status=status))
                if status == 'opened':
                    active = dict(strategy=name, campaign=len(campaigns) + 1,
                                  side=side, signal_date=date, window=window,
                                  frozen_amount=available, per_day=available/window,
                                  executed_amount=0., remaining=window,
                                  first_fill=pd.NaT, last_fill=pd.NaT,
                                  cash_change=0., realized_pnl=0.)
                    campaigns.append(active)

        core_fund = core * r.adjusted_close
        tactical_fund = tactical * r.adjusted_close
        nav = core_fund + tactical_fund + cash
        legacy_factor = nav/(previous + inflow) if previous + inflow else 1.
        if inflow and previous:
            factor = (value_before_open_flow/previous) * (nav/(value_before_open_flow+inflow))
        else:
            factor = legacy_factor
        daily_return = factor-1
        twr *= factor
        twr_legacy *= legacy_factor
        rows.append(dict(date=date, strategy=name, inflow=inflow,
                         contributions=contributed, tactical_contributions=tactical_contributed,
                         core_units=core, tactical_units=tactical, core_fund=core_fund,
                         tactical_fund=tactical_fund, tactical_nav=tactical_fund+cash,
                         cash=cash, fund=core_fund+tactical_fund, nav=nav,
                         daily_return=daily_return, twr=twr, twr_legacy=twr_legacy, buy_spent=buy_spent,
                         sale_net=sale_net, fees=fees, sell_tax=tax,
                         tactical_basis=tactical_basis, realized_pnl=realized,
                         active_side=active['side'] if active else '',
                         active_campaign=active['campaign'] if active else 0))
        previous = nav
        assert cash > -EPS and core >= 0 and tactical >= 0
        assert abs(cash - (tactical_contributed - buy_spent + sale_net)) < 1e-6
        assert abs(tactical_fund + cash - tactical_contributed - realized
                   - (tactical_fund - tactical_basis)) < 1e-6

    d = pd.DataFrame(rows)
    t = pd.DataFrame(trades)
    c = pd.DataFrame(campaigns)
    s = pd.DataFrame(signals)
    # Every campaign follows the next W eligible opens, including unfinished ones.
    for a in campaigns:
        fills = t.loc[(t.sleeve == 'tactical') & (t.campaign == a['campaign'])]
        eligible = q.index[(q.index > a['signal_date']) & q.tradable][:a['window']]
        assert list(fills.date) == list(eligible)
    return d, t, c, s


def summarize(result, end):
    d, t, c, s = result
    d = d.loc[d.date <= pd.Timestamp(end)]
    t = t.loc[t.date <= pd.Timestamp(end)]
    last = d.iloc[-1]
    flows = [(r.date, -r.inflow) for r in d.itertuples() if r.inflow]
    peak = d.twr.cummax().clip(lower=1.)
    dd = d.twr/peak - 1
    trough = dd.idxmin()
    earlier = d.loc[:trough]
    peak_date = earlier.loc[earlier.twr.idxmax(), 'date']
    tactical_trades = t.loc[t.sleeve == 'tactical']
    return dict(strategy=last.strategy, start=str(d.date.iloc[0].date()),
                requested_end=str(pd.Timestamp(end).date()), end=str(last.date.date()),
                contributions=last.contributions, final_nav=last.nav,
                profit=last.nav-last.contributions,
                xirr_pct=100*base.original.xirr(flows+[(last.date,last.nav)]),
                twr_return_pct=100*(last.twr-1), mdd_pct=100*dd.min(),
                mdd_peak_date=str(peak_date.date()), mdd_trough_date=str(d.loc[trough,'date'].date()),
                cash=last.cash, core_fund=last.core_fund,
                tactical_fund=last.tactical_fund, tactical_nav=last.tactical_nav,
                tactical_contributions=last.tactical_contributions,
                realized_pnl=last.realized_pnl, fees=last.fees, sell_tax=last.sell_tax,
                mean_cash_pct=100*(d.cash/d.nav).mean(),
                buy_fill_days=int((tactical_trades.side=='buy').sum()),
                sell_fill_days=int((tactical_trades.side=='sell').sum()),
                buy_campaigns=0 if c.empty else int(((c.side=='buy') & (c.signal_date<=pd.Timestamp(end))).sum()),
                sell_campaigns=0 if c.empty else int(((c.side=='sell') & (c.signal_date<=pd.Timestamp(end))).sum()))


def main():
    p, m = base.original.load()
    f = make_features(m)
    assert f.loc[base.START:base.END, 'ready'].all()
    for n in (400, 800, 1000):
        pd.testing.assert_frame_equal(make_features(m.iloc[:n]), f.iloc[:n])
    assert list(f.index[f.buy_event]) == base.original.event_dates(f, 'dynamic')

    policies = [('DCA_100',1.,False), ('DCA_80_window_buy_only',.8,False),
                ('DCA_80_window_roundtrip',.8,True), ('diagnostic_tactical_100',0.,True)]
    results = {name: simulate(p,f,name,fraction,exits) for name,fraction,exits in policies}
    checks=[]
    def close_check(label, a, b, tolerance=1e-7):
        error=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
        assert error < tolerance, (label,error)
        checks.append(dict(check=label, max_absolute_error=error, tolerance=tolerance, passed=True))

    dca = results['DCA_100'][0].set_index('date')
    buy = results['DCA_80_window_buy_only'][0].set_index('date')
    rt = results['DCA_80_window_roundtrip'][0].set_index('date')
    full = results['diagnostic_tactical_100'][0].set_index('date')
    old_dca=pd.read_csv(base.BACKTEST/'dynamic/daily.csv',parse_dates=['date'])
    old_dca=old_dca.loc[old_dca.strategy=='DCA'].set_index('date')
    old_buy=pd.read_csv(base.BACKTEST/'dynamic/window_installments_daily.csv',parse_dates=['date']).set_index('date')
    assert dca.index.equals(old_dca.index) and buy.index.equals(old_buy.index)
    for col in ('nav','cash','fund','contributions'):
        close_check('original_DCA_'+col,dca[col],old_dca[col])
    close_check('original_DCA_legacy_twr',dca.twr_legacy,old_dca.twr)
    for col in ('nav','cash','fund','contributions'):
        close_check('buy_only_linear_'+col,buy[col],.8*dca[col]+.2*old_buy[col])
        close_check('roundtrip_linear_'+col,rt[col],.8*dca[col]+.2*full[col])
    close_check('core_never_sold',rt.core_fund,.8*dca.fund)
    # Truncating future observations must not alter the earlier trading state.
    for endpoint in ('2024-12-31','2025-12-31'):
        pp=p.loc[:endpoint]
        ff=make_features(m.loc[:endpoint])
        prefix=simulate(pp,ff,'DCA_80_window_roundtrip',.8,True,pd.Timestamp(endpoint))[0].set_index('date')
        for col in ('nav','cash','core_units','tactical_units','twr'):
            close_check('prefix_'+endpoint+'_'+col,prefix[col],rt.loc[prefix.index,col])

    summary=base.add_comparison(pd.DataFrame([summarize(x,base.END) for x in results.values()]))
    snapshots=base.add_comparison(pd.DataFrame([summarize(x,end) for end in ('2024-12-31','2025-12-31','2026-10-14') for x in results.values()]))
    for name,i in [('daily',0),('trades',1),('campaigns',2),('signals',3)]:
        pd.concat([r[i] for r in results.values()],ignore_index=True).to_csv(HERE/f'window_roundtrip_{name}.csv',index=False)
    for name,table in [('summary',summary),('snapshots',snapshots),('validation',pd.DataFrame(checks))]:
        table.to_csv(HERE/f'window_roundtrip_{name}.csv',index=False)
    f.to_csv(HERE/'window_roundtrip_features.csv',index_label='date')
    paths=[Path(__file__),HERE/'allocation_study.py',base.BACKTEST/'dynamic/run_dynamic.py',
           base.BACKTEST/'prices_reviewed.csv',base.BACKTEST/'margin_daily.csv']
    rules=dict(rules=__doc__,policies=policies,parameter_search=False,
               source_sha256={str(x.relative_to(base.BACKTEST)):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths},
               sell_tax_source='https://www.twse.com.tw/zh/products/securities/etf/overview/rules.html',
               settlement_source='https://www.twse.com.tw/zh/products/system/trading.html')
    (HERE/'window_roundtrip_rules.json').write_text(json.dumps(rules,ensure_ascii=False,indent=2))
    cols=['strategy','final_nav','xirr_pct','mdd_pct','cash','tactical_nav','buy_campaigns','sell_campaigns','fees','sell_tax','higher_return_and_no_worse_drawdown']
    print(summary[cols].round(6).to_string(index=False))
    print(results['DCA_80_window_roundtrip'][2].to_string(index=False))
    print('Validation checks:',len(checks),'maximum error',max(x['max_absolute_error'] for x in checks))


if __name__=='__main__':
    main()

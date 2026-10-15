"""Fixed-rule cash-allocation comparison; local snapshots only, no optimization.

Period: 2023-03-01 through 2026-10-14. Deposit TWD 10,000 on each month's
first tradable ETF date. Contributions arrive before that day's open. Buy fee
0.1425% is included in the cash spent; cash earns zero. No selling, borrowing,
or leverage. Adjusted open/close represent fractional total-return units.

Low condition on t: ratio[t] < quantile(ratio[t-252:t], 0.10), excluding t.
Dynamic condition adds the EXISTING dynamic_diff[t] < -10 percentage points;
window = round(20 * (sigma_ref / sigma)^2), clipped to 5..60, with sigma and
sigma_ref using only prior observations, exactly as in run_dynamic.features.
Conditions are STATES, not false-to-true crossings: every qualifying close
schedules spending remaining cash at the next tradable open. If multiple
signals map to the same open (e.g. suspension), they produce one sweep.

Five predeclared policies: 100% DCA; 80% DCA + low-condition cash sweep;
80% DCA + dynamic-double-condition cash sweep; 0% DCA + low cash sweep;
80% DCA + low sweep with reserve capped at TWD 30,000. For the capped policy,
after the monthly 80% buy, excess reserve is immediately bought at that same
open. Then any scheduled condition spends the rest. No parameter search.

Year-end snapshots use the SAME 2023-03-01 start and are NOT out-of-sample.
TWR assumes deposits at the start of each date, matching existing DCA.
MDD is measured on TWR, not contribution-inflated NAV. The study retains all
limitations of the existing market-wide ratio and adjusted-price snapshots.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BACKTEST = HERE.parent
sys.path.insert(0, str(BACKTEST / 'dynamic'))
import run_dynamic as original

START = pd.Timestamp('2023-03-01')
END = pd.Timestamp('2026-10-14')
FEE = .001425
DEPOSIT = 10000.
POLICIES = [
    ('DCA_100', 1., None, None),
    ('DCA_80_low_sweep', .8, 'low', None),
    ('DCA_80_dynamic_sweep', .8, 'dynamic', None),
    ('Cash_100_low_sweep', 0., 'low', None),
    ('DCA_80_low_cap30000', .8, 'low', 30000.),
]


def make_schedule(prices, condition):
    trading = prices.index[prices.tradable]
    schedule = {}
    for signal in condition.index[condition]:
        later = trading[(trading > signal) & (trading <= END)]
        if len(later) and later[0] >= START:
            schedule.setdefault(later[0], []).append(signal)
    return schedule


def simulate(prices, policy, schedules):
    name, dca_fraction, signal_kind, reserve_cap = policy
    q = prices.loc[START:END]
    schedule = schedules.get(signal_kind, {})
    cash = units = contributed = spent = previous_nav = 0.
    twr = 1.
    last_month = None
    rows, trades = [], []
    for date, r in q.iterrows():
        inflow = 0.
        if r.tradable and date.to_period('M') != last_month:
            inflow = DEPOSIT
            cash += inflow
            contributed += inflow
            last_month = date.to_period('M')

        def buy(amount, reason, signal_dates=()):
            nonlocal cash, units, spent
            if amount <= 1e-8:
                return
            assert r.tradable and amount <= cash + 1e-8
            added = amount / (1 + FEE) / r.adjusted_open
            trades.append(dict(strategy=name, date=date, reason=reason,
                               signal_date=max(signal_dates) if signal_dates else pd.NaT,
                               signal_dates=';'.join(str(x.date()) for x in signal_dates),
                               amount_including_fee=amount, adjusted_open=r.adjusted_open,
                               normalized_units=added, buy_fee=amount-amount/(1+FEE),
                               cash_before=cash, cash_after=cash-amount))
            cash -= amount
            units += added
            spent += amount

        if inflow:
            buy(inflow * dca_fraction, 'monthly_dca')
            if reserve_cap is not None:
                buy(max(0., cash - reserve_cap), 'monthly_reserve_cap')
        if date in schedule:
            buy(cash, 'signal_cash_sweep', schedule[date])
        fund = units * r.adjusted_close
        nav = cash + fund
        daily_return = nav / (previous_nav + inflow) - 1 if previous_nav + inflow else 0.
        twr *= 1 + daily_return
        rows.append(dict(date=date, strategy=name, inflow=inflow, cash=cash,
                         fund=fund, nav=nav, contributions=contributed, spent=spent,
                         normalized_units=units, daily_return=daily_return, twr=twr))
        previous_nav = nav
        assert cash >= -1e-8 and units >= 0
        assert abs(cash + spent - contributed) < 1e-6
        if reserve_cap is not None:
            assert cash <= reserve_cap + 1e-8
    daily, transactions = pd.DataFrame(rows), pd.DataFrame(trades)
    assert np.isclose(sum(t['normalized_units'] for t in trades) * q.adjusted_close.iloc[-1], daily.fund.iloc[-1])
    assert contributed == 440000
    return daily, transactions


def summarize(daily, requested_end):
    d = daily.loc[daily.date <= pd.Timestamp(requested_end)]
    last = d.iloc[-1]
    flows = [(r.date, -r.inflow) for r in d.itertuples() if r.inflow]
    return dict(strategy=last.strategy, start=str(d.date.iloc[0].date()),
                requested_end=str(pd.Timestamp(requested_end).date()), end=str(last.date.date()),
                contributions=last.contributions, spent=last.spent, cash=last.cash,
                final_nav=last.nav, profit=last.nav-last.contributions,
                profit_on_contributions_pct=100*(last.nav/last.contributions-1),
                xirr_pct=100*original.xirr(flows+[(last.date, last.nav)]),
                twr_return_pct=100*(last.twr-1),
                mdd_pct=100*(d.twr/d.twr.cummax()-1).min(),
                mean_cash= d.cash.mean(), mean_cash_pct=100*(d.cash/d.nav).mean())


def add_comparison(frame):
    for end, group in frame.groupby('requested_end'):
        dca = group.loc[group.strategy == 'DCA_100'].iloc[0]
        ix = group.index
        frame.loc[ix, 'nav_minus_dca'] = group.final_nav - dca.final_nav
        frame.loc[ix, 'xirr_minus_dca_pp'] = group.xirr_pct - dca.xirr_pct
        frame.loc[ix, 'mdd_minus_dca_pp'] = group.mdd_pct - dca.mdd_pct
        frame.loc[ix, 'higher_return_and_no_worse_drawdown'] = (
            (group.xirr_pct > dca.xirr_pct + 1e-10) & (group.mdd_pct >= dca.mdd_pct - 1e-10))
    return frame


def main():
    p, m = original.load()
    f = original.features(m)
    assert f.loc[START:END, 'ready'].all()
    # Feature prefix invariance verifies that future rows do not alter signals.
    for n in (400, 800, 1000):
        pd.testing.assert_frame_equal(original.features(m.iloc[:n]), f.iloc[:n])
    low = (m < f.low_threshold) & f.low_threshold.notna()
    dynamic = low & (f.dynamic_diff < -10) & f.ready
    schedules = {k: make_schedule(p, v) for k, v in [('low', low), ('dynamic', dynamic)]}
    results = [simulate(p, policy, schedules) for policy in POLICIES]
    daily = pd.concat([x[0] for x in results], ignore_index=True)
    trades = pd.concat([x[1] for x in results], ignore_index=True)
    summary = add_comparison(pd.DataFrame([summarize(d, END) for d, _ in results]))
    snapshots = add_comparison(pd.DataFrame([summarize(d, e) for e in ('2024-12-31', '2025-12-31', '2026-10-14') for d, _ in results]))

    # Full daily reconciliation to the existing SAME-PERIOD DCA, no refitting.
    old = pd.read_csv(BACKTEST/'dynamic/daily.csv', parse_dates=['date'])
    old = old.loc[old.strategy == 'DCA'].set_index('date')
    new = daily.loc[daily.strategy == 'DCA_100'].set_index('date')
    assert new.index.equals(old.index)
    checks = []
    for col in ['cash', 'fund', 'nav', 'contributions', 'twr']:
        maximum = float((new[col]-old[col]).abs().max())
        assert np.allclose(new[col], old[col], rtol=0, atol=1e-8), (col, maximum)
        checks.append(dict(metric=col, max_absolute_daily_difference=maximum, tolerance=1e-8, passed=True))
    old_summary = pd.read_csv(BACKTEST/'dynamic/summary.csv').set_index('strategy').loc['DCA']
    now = summary.set_index('strategy').loc['DCA_100']
    for new_col, old_col in [('final_nav', 'nav'), ('xirr_pct', 'xirr_pct'), ('mdd_pct', 'mdd_pct')]:
        assert abs(now[new_col]-old_summary[old_col]) < 1e-8
    for name, frame in [('daily', daily), ('trades', trades), ('summary', summary), ('snapshots', snapshots), ('dca_validation', pd.DataFrame(checks))]:
        frame.to_csv(HERE/f'allocation_{name}.csv', index=False)
    inputs = [BACKTEST/'prices_reviewed.csv', BACKTEST/'margin_daily.csv', BACKTEST/'dynamic/run_dynamic.py', BACKTEST/'dynamic/daily.csv', BACKTEST/'dynamic/summary.csv']
    metadata = dict(rules=__doc__, policies=POLICIES, source_sha256={str(x.relative_to(BACKTEST)): hashlib.sha256(x.read_bytes()).hexdigest() for x in inputs},
                    source_latest=str(p.index[-1].date()), network_calls=0, parameter_optimization=False,
                    conditional_execution='next_tradable_open_after_each_qualifying_close',
                    snapshot_interpretation='same-start nested endpoints, not out-of-sample')
    (HERE/'allocation_rules.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(summary.round(6).to_string(index=False))
    print('\nDCA validation: max daily absolute difference', max(x['max_absolute_daily_difference'] for x in checks))


if __name__ == '__main__':
    main()

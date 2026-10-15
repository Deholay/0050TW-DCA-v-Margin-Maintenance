"""Single prespecified 3-calendar-month expiry extension; no parameter search.

Use exactly allocation_study's 2023-03-01..2026-10-14 prices, daily low-state
signals, monthly TWD 10,000 deposits, adjusted total-return units, zero cash
yield, and proportional 0.1425% buy fee. No selling, borrowing, or leverage.
For each month's reserve tranche, expiry is the FIRST TRADABLE DAY of the
deposit month + 3 calendar months (March tranche expires at June's first
tradable open). Low signals execute at the next tradable open, sweeping all
unspent tranches. Otherwise expiring tranches buy at their scheduled open.
An expiry purchase is a known calendar order and does not depend on that
day's closing information. The monthly DCA component buys at deposit open.

Only two variants are added: 80% DCA / 20% reserve with 3-month expiry, and
100% reserve with the same expiry. No alternate expiry horizon is evaluated.
The second variant tests the waiting mechanism and permits independent daily
80/20 accounting reconstruction: 80% of DCA + 20% of all-reserve expiry.
NAV/cash/fund/units are linear; XIRR, TWR and MDD need not be linear.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import allocation_study as base

HERE = Path(__file__).resolve().parent


def simulate(prices, schedule, dca_fraction, name):
    q = prices.loc[base.START:base.END]
    cash = units = contributions = spent = previous_nav = 0.
    twr = 1.
    last_month = None
    pending, batches, rows, trades = [], [], [], []
    for date, r in q.iterrows():
        month = date.to_period('M')
        inflow = 0.

        def buy(amount, reason, batch=None, signal_dates=()):
            nonlocal cash, units, spent
            if amount <= 1e-8:
                return
            assert r.tradable and amount <= cash + 1e-8
            added = amount / (1 + base.FEE) / r.adjusted_open
            units += added
            spent += amount
            cash -= amount
            trades.append(dict(strategy=name, date=date, reason=reason,
                               tranche_month=str(batch['origin']) if batch else '',
                               expiry_month=str(batch['expiry']) if batch else '',
                               signal_date=max(signal_dates) if signal_dates else pd.NaT,
                               amount_including_fee=amount, adjusted_open=r.adjusted_open,
                               normalized_units=added, buy_fee=amount-amount/(1+base.FEE)))
            if batch:
                batches.append(dict(strategy=name, tranche_month=str(batch['origin']),
                                    deposit_date=batch['deposit_date'], expiry_month=str(batch['expiry']),
                                    amount=amount, execution_date=date, reason=reason,
                                    waiting_calendar_days=(date-batch['deposit_date']).days))

        if r.tradable and month != last_month:
            inflow = base.DEPOSIT
            cash += inflow
            contributions += inflow
            last_month = month
            buy(inflow*dca_fraction, 'monthly_dca')
            reserve = inflow-inflow*dca_fraction
            if reserve:
                pending.append(dict(origin=month, expiry=month+3, amount=reserve, deposit_date=date))
        if r.tradable:
            if date in schedule:
                for batch in pending:
                    buy(batch['amount'], 'low_signal', batch, schedule[date])
                pending = []
            else:
                keep = []
                for batch in pending:
                    if month >= batch['expiry']:
                        buy(batch['amount'], 'three_month_expiry', batch)
                    else:
                        keep.append(batch)
                pending = keep
        fund = units*r.adjusted_close
        nav = cash+fund
        daily_return = nav/(previous_nav+inflow)-1 if previous_nav+inflow else 0.
        twr *= 1+daily_return
        rows.append(dict(date=date, strategy=name, inflow=inflow, cash=cash, fund=fund,
                         nav=nav, contributions=contributions, spent=spent,
                         normalized_units=units, daily_return=daily_return, twr=twr,
                         pending_tranches=len(pending)))
        previous_nav = nav
        assert cash >= -1e-7 and abs(cash-sum(b['amount'] for b in pending)) < 1e-7
        assert abs(cash+spent-contributions) < 1e-7
    for batch in pending:
        batches.append(dict(strategy=name, tranche_month=str(batch['origin']),
                            deposit_date=batch['deposit_date'], expiry_month=str(batch['expiry']),
                            amount=batch['amount'], execution_date=pd.NaT, reason='still_waiting_at_end',
                            waiting_calendar_days=(base.END-batch['deposit_date']).days))
    assert contributions == 440000
    assert np.isclose(sum(t['normalized_units'] for t in trades)*q.adjusted_close.iloc[-1], rows[-1]['fund'])
    return pd.DataFrame(rows), pd.DataFrame(trades), pd.DataFrame(batches)


def main():
    p, m = base.original.load()
    f = base.original.features(m)
    low = (m < f.low_threshold) & f.low_threshold.notna()
    schedule = base.make_schedule(p, low)
    results = [simulate(p, schedule, .8, 'DCA_80_low_expiry3m'),
               simulate(p, schedule, 0., 'Cash_100_low_expiry3m')]
    daily = pd.concat([r[0] for r in results], ignore_index=True)
    trades = pd.concat([r[1] for r in results], ignore_index=True)
    batches = pd.concat([r[2] for r in results], ignore_index=True)
    existing = pd.read_csv(HERE/'allocation_summary.csv')
    baselines = existing[existing.strategy.isin(['DCA_100', 'DCA_80_low_sweep', 'Cash_100_low_sweep'])].copy()
    summary = pd.concat([baselines, pd.DataFrame([base.summarize(r[0], base.END) for r in results])], ignore_index=True)
    summary = base.add_comparison(summary)
    original80 = summary.loc[summary.strategy == 'DCA_80_low_sweep'].iloc[0]
    for metric in ['final_nav', 'xirr_pct', 'mdd_pct', 'mean_cash']:
        summary[f'{metric}_minus_original80'] = summary[metric]-original80[metric]

    old_daily = pd.read_csv(HERE/'allocation_daily.csv', parse_dates=['date'])
    dca = old_daily[old_daily.strategy == 'DCA_100'].set_index('date')
    blend, all_cash = [r[0].set_index('date') for r in results]
    checks = []
    for col in ['cash', 'fund', 'nav', 'contributions', 'spent', 'normalized_units']:
        delta = blend[col]-(.8*dca[col]+.2*all_cash[col])
        assert np.allclose(delta, 0, atol=1e-8, rtol=0), (col, delta.abs().max())
        checks.append(dict(check='80pct_DCA_plus_20pct_all_reserve', metric=col,
                           max_absolute_difference=delta.abs().max(), tolerance=1e-8, passed=True))
    signals = trades.loc[trades.signal_date.notna()]
    assert (signals.date > signals.signal_date).all()
    for r in trades.loc[trades.reason == 'three_month_expiry'].itertuples():
        eligible = p.index[p.tradable & (p.index.to_period('M') == pd.Period(r.expiry_month))]
        assert r.date == eligible[0]
    assert batches.groupby('strategy').amount.sum().to_dict() == {'Cash_100_low_expiry3m': 440000., 'DCA_80_low_expiry3m': 88000.}
    for name, frame in [('daily', daily), ('trades', trades), ('batches', batches), ('summary', summary), ('validation', pd.DataFrame(checks))]:
        frame.to_csv(HERE/f'allocation_expiry_{name}.csv', index=False)
    paths = [Path(__file__), HERE/'allocation_study.py', base.BACKTEST/'prices_reviewed.csv', base.BACKTEST/'margin_daily.csv']
    metadata = dict(rules=__doc__, only_expiry_horizon_months=3, optimization=False,
                    source_sha256={str(x.relative_to(base.BACKTEST)): hashlib.sha256(x.read_bytes()).hexdigest() for x in paths})
    (HERE/'allocation_expiry_rules.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(summary[['strategy', 'contributions', 'final_nav', 'xirr_pct', 'mdd_pct', 'cash', 'mean_cash', 'higher_return_and_no_worse_drawdown']].round(6).to_string(index=False))
    print('Linear validation max error:', max(x['max_absolute_difference'] for x in checks))


if __name__ == '__main__':
    main()

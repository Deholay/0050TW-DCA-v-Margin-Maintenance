"""Daily state-gated exponential cash deployment. See run_study.py for rules.
Execution/accounting derived from review_20261015/window_roundtrip.py;
original files and old results are untouched. Buy campaigns here are one-day
orders, not comparable in count to the old multi-day entry episodes.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'review_20261015'))
import window_roundtrip as old
base=old.base
FEE=old.FEE
SELL_TAX=old.SELL_TAX
EPS=old.EPS


def features(m, tau=20.):
    f=old.make_features(m)
    # Normalizations use only prior observations, not future minimum prices.
    f['stress']=((f.low_threshold-m)/(f.sigma_ref*np.sqrt(20))).clip(lower=0)
    f['budget_horizon']=(tau*(f.sigma/f.sigma_ref)**2).clip(5,60)
    f['hazard']=(1+f.stress**2)/f.budget_horizon
    f['spend_fraction']=-np.expm1(-f.hazard)
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
                if not f.loc[date, 'buy_state' if side == 'buy' else 'sell_event']:
                    continue
                available = cash if side == 'buy' else tactical
                status = ('disabled' if side == 'sell' and not exits else
                          'ignored_active' if active is not None else
                          'no_cash' if side == 'buy' and available < EPS else
                          'no_units' if side == 'sell' and available < EPS else 'opened')
                window = 1 if side == 'buy' else int(f.loc[date, 'window'])
                # Daily buy orders are recomputed at each qualifying close.
                # Sell campaigns retain the original frozen-W rule.
                amount = available * f.loc[date, 'spend_fraction'] if side == 'buy' else available
                signals.append(dict(strategy=name, signal_date=date, side=side,
                                    window=window, available=available, planned_amount=amount,
                                    horizon=f.loc[date,'budget_horizon'], stress=f.loc[date,'stress'],
                                    spend_fraction=f.loc[date,'spend_fraction'], status=status))
                if status == 'opened':
                    active = dict(strategy=name, campaign=len(campaigns) + 1,
                                  side=side, signal_date=date, window=window,
                                  frozen_amount=amount, per_day=amount/window,
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

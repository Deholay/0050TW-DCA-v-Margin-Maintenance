"""Diagnostic only: low-cross/all-cash entry versus matched monthly DCA cohorts.

This is NOT the daily-condition allocation rule in allocation_study.py.
No network calls. Writes only cash_wait_* files beside this script.
Run: /tmp/margin-backtest-venv/bin/python backtest/review_20261015/cash_wait_audit.py
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUT = ROOT.parent
START = pd.Timestamp("2023-03-01")
END = pd.Timestamp("2026-10-14")
MONTHLY = 10_000.0
FEE = 0.001425


def main():
    prices = pd.read_csv(INPUT / "prices_reviewed.csv", parse_dates=["date"]).set_index("date")
    margin = pd.read_csv(INPUT / "margin_daily.csv", parse_dates=["date"]).set_index("date").ratio
    assert prices.index.equals(margin.index)
    sample = prices.loc[START:END]
    tradable = sample.index[sample.tradable]
    monthly_rows = sample[sample.tradable].groupby(sample[sample.tradable].index.to_period("M")).head(1)
    low_threshold = margin.shift(1).rolling(252).quantile(0.1)
    condition = margin < low_threshold
    crosses = condition & ~condition.shift(1, fill_value=False)
    signals = margin.index[crosses & (margin.index >= START) & (margin.index <= END)]
    schedules = {}
    for signal in signals:
        later = tradable[tradable > signal]
        if len(later):
            schedules.setdefault(later[0], []).append(signal)

    previous_execution = START - pd.Timedelta(days=1)
    campaigns, contributions, signal_rows = [], [], []
    for execution, source_signals in schedules.items():
        deposits = monthly_rows[(monthly_rows.index > previous_execution) & (monthly_rows.index <= execution)]
        capital = MONTHLY * len(deposits)
        for signal in source_signals:
            signal_rows.append(dict(signal_date=signal, execution_date=execution,
                                    ratio=float(margin.loc[signal]),
                                    low_threshold=float(low_threshold.loc[signal]),
                                    capital=capital, status="bought" if capital else "no_cash"))
        if capital:
            campaign = len(campaigns) + 1
            entry = float(prices.loc[execution, "adjusted_open"])
            month_first = monthly_rows.index[monthly_rows.index.to_period("M") == execution.to_period("M")][0]
            month_price = float(prices.loc[month_first, "adjusted_open"])
            gross_signal_units = capital / entry
            gross_dca_units = float((MONTHLY / deposits.adjusted_open).sum())
            signal_units = gross_signal_units / (1 + FEE)
            dca_units = gross_dca_units / (1 + FEE)
            waits = [(execution - date).days for date in deposits.index]
            campaigns.append(dict(
                campaign=campaign, signal_date=source_signals[-1], execution_date=execution,
                first_deposit_date=deposits.index[0], last_deposit_date=deposits.index[-1],
                deposit_count=len(deposits), capital=capital,
                average_wait_calendar_days=float(np.mean(waits)),
                maximum_wait_calendar_days=max(waits), entry_adjusted_open=entry,
                month_first_date=month_first, month_first_adjusted_open=month_price,
                entry_price_vs_month_first_pct=100 * (entry / month_price - 1),
                same_capital_units_vs_month_first_pct=100 * (month_price / entry - 1),
                matched_dca_effective_adjusted_price=capital / gross_dca_units,
                signal_units_before_fee=gross_signal_units, dca_units_before_fee=gross_dca_units,
                signal_units_after_fee=signal_units, dca_units_after_fee=dca_units,
                units_difference_after_fee=signal_units - dca_units,
                units_lift_vs_matched_dca_pct=100 * (signal_units / dca_units - 1),
            ))
            for date, row in deposits.iterrows():
                contributions.append(dict(campaign=campaign, deposit_date=date, capital=MONTHLY,
                                          dca_adjusted_open=float(row.adjusted_open),
                                          execution_date=execution, signal_adjusted_open=entry,
                                          wait_calendar_days=(execution - date).days,
                                          dca_units_after_fee=MONTHLY / (1 + FEE) / row.adjusted_open,
                                          signal_units_after_fee=MONTHLY / (1 + FEE) / entry))
        previous_execution = execution

    c = pd.DataFrame(campaigns)
    deposits = pd.DataFrame(contributions)
    pending = monthly_rows[monthly_rows.index > previous_execution]
    pending_capital = MONTHLY * len(pending)
    pending_dca_units = float((MONTHLY / (1 + FEE) / pending.adjusted_open).sum())
    final_price = float(sample.adjusted_close.iloc[-1])
    all_dca_units = float((MONTHLY / (1 + FEE) / monthly_rows.adjusted_open).sum())
    assert not deposits.deposit_date.duplicated().any()
    assert len(deposits) + len(pending) == len(monthly_rows) == 44
    assert np.isclose(c.capital.sum() + pending_capital, MONTHLY * len(monthly_rows))
    assert np.isclose(c.dca_units_after_fee.sum() + pending_dca_units, all_dca_units)
    assert np.isclose(deposits.signal_units_after_fee.sum(), c.signal_units_after_fee.sum())
    assert (deposits.execution_date >= deposits.deposit_date).all()
    assert (c.execution_date > c.signal_date).all()
    total_signal_units = float(c.signal_units_after_fee.sum())
    total_matched_dca_units = float(c.dca_units_after_fee.sum())
    summary = dict(
        diagnostic="low-cross all-cash; distinct from daily-condition allocation_study",
        start=str(START.date()), end=str(END.date()), lookback=252, low_quantile=0.1,
        signal_rule="At close, ratio strictly below the previous 252 observations' 10th percentile; false-to-true crossing only.",
        execution_rule="Invest all cash at first tradable open after crossing; no purchase on continuing low-condition days.",
        capital_rule="TWD 10,000 deposited on each month's first sample tradable day; opening-day deposit precedes a scheduled purchase.",
        comparator="Same contribution cohorts buy at each deposit date's adjusted open. Cohorts never overlap.",
        price_rule="Adjusted total-return equivalent units; same 0.1425% proportional buy fee, no minimum fee, cash interest zero.",
        interpretation="Lower entry versus current month does not imply lower entry versus all prior months whose cash waited. Diagnostic, not optimized strategy or independent out-of-sample validation.",
        signals=len(signals), funded_campaigns=len(c), positive_unit_campaigns=int((c.units_difference_after_fee > 0).sum()),
        total_deposits=len(monthly_rows), invested_deposits=len(deposits), pending_deposits=len(pending),
        contributed=MONTHLY * len(monthly_rows), invested_capital=float(c.capital.sum()), pending_cash=pending_capital,
        average_wait_calendar_days=float(deposits.wait_calendar_days.mean()),
        signal_units_after_fee=total_signal_units, matched_dca_units_after_fee=total_matched_dca_units,
        matched_unit_difference_pct=100 * (total_signal_units / total_matched_dca_units - 1),
        median_campaign_unit_difference_pct=float(c.units_lift_vs_matched_dca_pct.median()),
        final_adjusted_close=final_price,
        matched_cohort_final_value_difference=(total_signal_units - total_matched_dca_units) * final_price,
        pending_cohort_dca_final_value=pending_dca_units * final_price,
        low_cross_all_cash_final_nav=total_signal_units * final_price + pending_capital,
        dca_final_nav=all_dca_units * final_price,
        final_nav_difference=total_signal_units * final_price + pending_capital - all_dca_units * final_price,
        input_sha256={name: hashlib.sha256((INPUT / name).read_bytes()).hexdigest()
                      for name in ["prices_reviewed.csv", "margin_daily.csv"]},
    )
    c.to_csv(ROOT / "cash_wait_campaigns.csv", index=False)
    deposits.to_csv(ROOT / "cash_wait_contributions.csv", index=False)
    pd.DataFrame(signal_rows).to_csv(ROOT / "cash_wait_signals.csv", index=False)
    (ROOT / "cash_wait_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    display = c[["execution_date", "deposit_count", "capital", "average_wait_calendar_days",
                 "same_capital_units_vs_month_first_pct", "units_lift_vs_matched_dca_pct"]].copy()
    numeric_columns = display.select_dtypes(include="number").columns
    display[numeric_columns] = display[numeric_columns].round(4)
    print(display.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

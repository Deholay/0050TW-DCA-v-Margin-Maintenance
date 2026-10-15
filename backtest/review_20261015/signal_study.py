"""Read local snapshots; write new event-study artifacts without changing prior outputs.

Signal dates: 2023-03-01 through 2026-10-14. Entry: next tradable
0050 open. A horizon of H uses the close of the Hth tradable day,
counting the entry day as day 1. Only complete horizons are retained.
"""
from pathlib import Path
import json
import sys

import pandas as pd

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent / "dynamic"))
import run_dynamic as bt


def main():
    p, m = bt.load()
    f = bt.features(m)
    tradable = p.loc[p.tradable].copy()
    low = (f.ratio < f.low_threshold) & f.ready
    selections = {
        "all_tradable_entries": list(f.index[f.index >= bt.START]),
        "low_daily": list(f.index[low & (f.index >= bt.START)]),
        "low_cross": [d for d in bt.event_dates(f, "low_only") if d >= bt.START],
        "dynamic_cross": [d for d in bt.event_dates(f, "dynamic") if d >= bt.START],
    }
    rows = []
    signal_rows = []
    for group, dates in selections.items():
        used_entries = set()
        for signal_date in dates:
            signal_rows.append({"group": group, "signal_date": signal_date})
            entry_idx = tradable.index.searchsorted(signal_date, side="right")
            if entry_idx >= len(tradable):
                continue
            entry_date = tradable.index[entry_idx]
            # Suspended days can all map to one next tradable entry. The benchmark
            # counts each executable entry only once; none of these signal groups
            # has duplicate execution dates in this saved sample.
            if entry_date in used_entries:
                assert group == "all_tradable_entries"
                continue
            used_entries.add(entry_date)
            entry = tradable.iloc[entry_idx].adjusted_open
            for horizon in (20, 60, 120):
                exit_idx = entry_idx + horizon - 1
                if exit_idx >= len(tradable):
                    continue
                exit_date = tradable.index[exit_idx]
                exit_price = tradable.iloc[exit_idx].adjusted_close
                rows.append({
                    "group": group,
                    "signal_date": signal_date,
                    "entry_date": entry_date,
                    "horizon": horizon,
                    "exit_date": exit_date,
                    "adjusted_open": entry,
                    "exit_adjusted_close": exit_price,
                    "gross_return_pct": (exit_price / entry - 1) * 100,
                    "net_buy_fee_return_pct": (exit_price / entry / (1 + bt.FEE) - 1) * 100,
                })
    events = pd.DataFrame(rows)
    summary = events.groupby(["group", "horizon"], sort=False).agg(
        n=("gross_return_pct", "size"),
        mean_gross_pct=("gross_return_pct", "mean"),
        median_gross_pct=("gross_return_pct", "median"),
        positive_share_pct=("gross_return_pct", lambda s: (s > 0).mean() * 100),
        mean_net_buy_fee_pct=("net_buy_fee_return_pct", "mean"),
        min_gross_pct=("gross_return_pct", "min"),
        max_gross_pct=("gross_return_pct", "max"),
        first_entry=("entry_date", "min"),
        last_entry=("entry_date", "max"),
    ).reset_index()
    baseline = summary[summary.group == "all_tradable_entries"].set_index("horizon")
    summary["mean_vs_all_entries_pp"] = [
        r.mean_gross_pct - baseline.loc[r.horizon, "mean_gross_pct"]
        for r in summary.itertuples()
    ]
    metadata = {
        "period": [str(bt.START.date()), str(bt.END.date())],
        "input_rows": len(m),
        "inputs": ["../margin_daily.csv", "../prices_reviewed.csv"],
        "low_definition": "M[t] below 10th percentile of previous 252 observations; rolling inputs exclude t",
        "dynamic_definition": "low condition AND M[t]-M[t-W[t]] < -10 percentage points, false-to-true crossings",
        "entry": "next tradable adjusted open after signal close",
        "horizon": "20/60/120 tradable closes, counting entry date as day 1; complete horizons only",
        "returns": "adjusted-price total-return equivalents; gross and after 0.1425% entry fee; no exit fee",
        "signal_counts": {k: len(v) for k, v in selections.items()},
        "signals": {k: [str(d.date()) for d in v] for k, v in selections.items() if k != "all_tradable_entries"},
        "caveats": [
            "Every-day low signals overlap heavily. False-to-true episodes can also overlap forward horizons and are not independent observations.",
            "The all-entry benchmark measures an equal-weight entry opportunity, not the wealth or XIRR of a monthly DCA portfolio.",
            "This is a short, already-examined sample; signals cluster in 2024-2025 and may reflect regime composition. No causality, statistical significance, or out-of-sample edge is established.",
            "Financing rules, cash availability, installments and opportunity cost are excluded from this signal-quality study.",
        ],
    }
    targets = [OUT / f"signal_study.{suffix}" for suffix in ("csv", "events.csv", "signals.csv", "json")]
    for target in targets:
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite {target}")
    summary.to_csv(targets[0], index=False)
    events.to_csv(targets[1], index=False)
    pd.DataFrame(signal_rows).to_csv(targets[2], index=False)
    targets[3].write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(summary.round(3).to_string(index=False))
    print("Signal counts:", metadata["signal_counts"])


if __name__ == "__main__":
    main()

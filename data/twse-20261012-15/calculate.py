"""Replay TWSE daily totals and an incomplete same-day-close valuation.
Scope: TWSE-listed credit-trading securities (includes eligible ETFs), not TPEx.
Missing closes remain missing: displayed ratio is a covered-value lower bound,
not an official or complete maintenance ratio. Trading unit assumed 1,000 shares.
Sources: exchangeReport/MI_MARGN?response=json&date=YYYYMMDD&selectType=ALL
         exchangeReport/MI_INDEX?response=json&date=YYYYMMDD&type=ALLBUT0999
"""
import json
from decimal import Decimal as D
from pathlib import Path
ROOT = Path(__file__).resolve().parent
num = lambda s: D(s.replace(',', ''))
results = []
for day in ['20261012', '20261013', '20261014', '20261015']:
    margin = json.loads((ROOT / f'margin-{day}.json').read_text())
    prices = json.loads((ROOT / f'prices-{day}.json').read_text())
    if margin.get('stat') != 'OK' or prices.get('stat') != 'OK':
        results.append({'date': day, 'status': 'unavailable', 'source_status': margin.get('stat')})
        continue
    assert margin['date'] == prices['date'] == day
    summary, detail = margin['tables'][:2]
    total_row = next(r for r in summary['data'] if r[0] == '融資金額(仟元)')
    debt = num(total_row[5]) * 1000
    prior_debt = num(total_row[4]) * 1000
    price_table = next(t for t in prices['tables'] if '收盤價' in t.get('fields', []))
    price_map = {r[0]: r[8] for r in price_table['data']}
    assert len(price_map) == len(price_table['data'])
    assert len({r[0] for r in detail['data']}) == len(detail['data'])
    lots = sum(num(r[6]) for r in detail['data'])
    assert lots == num(summary['data'][0][5])
    value = D(0)
    missing = []
    for r in detail['data']:
        quantity = num(r[6])
        if quantity == 0:
            continue
        raw = price_map.get(r[0])
        if raw is None or raw == '--':
            missing.append({'code': r[0], 'name': r[1], 'lots': int(quantity), 'close': None})
            continue
        value += quantity * 1000 * num(raw)
    missing_lots = sum(x['lots'] for x in missing)
    results.append({'date': day, 'status': 'partial_valuation',
        'official_debt_twd': str(debt), 'debt_change_twd': str(debt-prior_debt),
        'official_debt_100m_twd': str(debt/D(100000000)),
        'covered_market_value_twd': str(value),
        'covered_value_ratio_pct_lower_bound': str(value/debt*100),
        'total_lots': int(lots), 'covered_lots_pct': str((lots-missing_lots)/lots*100),
        'missing_closes': missing})
(ROOT/'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(results, ensure_ascii=False, indent=2))

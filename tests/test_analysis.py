import numpy as np
import pandas as pd

from analysis import (order_sign_acf, fit_power_law, fit_exponent, static_book_impact, lag1_impact, order_frame,
                      spread_summary, tape_frame, BookRecorder, impact_vs_volume, average_depth_profile, return_statistics, execution_costs)
from orderbook import OrderBook


def test_flat_profile_gives_linear_impact_and_linear_profile_square_root():
    q = [10, 20, 40, 80, 160, 320, 640, 1280]
    flat = static_book_impact('flat', q)
    linear = static_book_impact('linear', q)
    assert abs(fit_exponent(flat['quantity'], flat['impact']) - 1.0) < 0.02
    assert abs(fit_exponent(linear['quantity'], linear['impact']) - 0.5) < 0.1


def test_independent_signs_have_no_significant_autocorrelation():
    rng = np.random.default_rng(0)
    signs = [rng.choice([-1, 1], size=400) for _ in range(30)]
    acf = order_sign_acf(signs, 20)
    assert (acf['acf'].abs() < 4 * acf['se']).all()
    assert fit_power_law(acf['lag'], acf['acf'], acf['se']) is None


def test_persistent_signs_give_a_power_law_exponent():
    rng = np.random.default_rng(1)
    # sign flips with probability 0.05: positive, slowly decaying correlations
    signs = []
    for _ in range(40):
        flips = rng.random(2000) < 0.05
        signs.append(np.where(np.cumsum(flips) % 2 == 0, 1, -1))
    acf = order_sign_acf(signs, 20)
    assert (acf['acf'] > 0).all()
    assert fit_power_law(acf['lag'], acf['acf'], acf['se']) is not None


def small_book_with_orders():
    book = OrderBook()
    rec = BookRecorder()
    def go(q):
        book.process_order(q)
        rec.record(book)
    for p, v in ((101, 3), (103, 4)):
        go({'type': 'limit', 'side': 'ask', 'quantity': v, 'price': p, 'trade_id': 1})
    for p, v in ((99, 3), (97, 4)):
        go({'type': 'limit', 'side': 'bid', 'quantity': v, 'price': p, 'trade_id': 2})
    go({'type': 'market', 'side': 'bid', 'quantity': 5, 'trade_id': 3})   # buys 3 @101 and 2 @103
    go({'type': 'market', 'side': 'ask', 'quantity': 1, 'trade_id': 4})   # sells 1 @99
    return book, rec


def test_order_frame_signs_mid_prices_and_aggregation():
    book, rec = small_book_with_orders()
    trades = tape_frame(book.tape)
    assert list(trades['sign']) == [1, 1, -1]
    orders = order_frame(trades, rec.frame())
    assert list(orders['sign']) == [1, -1]
    assert orders.loc[0, 'volume'] == 5
    assert orders.loc[0, 'vwap'] == (3 * 101 + 2 * 103) / 5
    assert orders.loc[0, 'mid_before'] == 100.0          # (99 + 101) / 2
    assert orders.loc[0, 'mid_after'] == (99 + 103) / 2  # the best ask moved to 103
    assert orders.loc[0, 'vbest'] == 3
    r1, se, n = lag1_impact([orders])
    assert n == 1 and r1 == 1 * (orders.loc[1, 'mid_before'] - orders.loc[0, 'mid_before'])


def test_spread_depth_and_return_statistics():
    book, rec = small_book_with_orders()
    snaps = rec.frame()
    s = spread_summary(snaps)
    assert s['observations'] > 0 and s['p95'] >= s['p5']
    profile = average_depth_profile([snaps])
    assert list(profile['level']) == [1, 2, 3, 4, 5]
    assert 'excess kurtosis' in return_statistics([snaps]).index


def test_execution_costs():
    sheet = [{'filled_quantity': 2, 'spread_before': 4.0, 'slippage': 3.0}, {'filled_quantity': 0}, {}]
    assert execution_costs(sheet) == (4.0, 6.0, 2.0)


def test_impact_vs_volume_returns_monotone_bins():
    rng = np.random.default_rng(0)
    n = 400
    o = pd.DataFrame({'sign': rng.choice([-1, 1], n), 'volume': rng.uniform(1, 30, n), 'vbest': 10.0,
                      'mid_before': np.cumsum(rng.normal(size=n)), 'mid_after': np.cumsum(rng.normal(size=n))})
    out = impact_vs_volume([o], nbins=5)
    assert len(out) == 5 and out['v'].is_monotonic_increasing

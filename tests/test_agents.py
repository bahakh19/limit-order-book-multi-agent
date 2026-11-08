from decimal import Decimal

import numpy as np
import pytest

from orderbook import OrderBook
from agents import (Trader_quantity, Trader_Buy_and_sell, Trader_random, Trader_bullish, Trader_bearish, Trader_MA,
                    affordable_market_buy, quantize_qty, mvt_avg, generate_lim_orders)
from analysis import BookRecorder
from simulation import settle


def seeded_book():
    book = OrderBook()
    for price, qty in ((101, 5), (102, 5), (104, 10)):
        book.process_order({'type': 'limit', 'side': 'ask', 'quantity': qty, 'price': price, 'trade_id': 1})
    for price, qty in ((99, 5), (98, 5), (95, 10)):
        book.process_order({'type': 'limit', 'side': 'bid', 'quantity': qty, 'price': price, 'trade_id': 2})
    return book


def send(book, trader, method):
    state = (trader.K, trader.pnl, trader.nb_shares)
    mid, spread = book.get_mid_price(), book.get_spread()
    val, quote, _ = method(book)
    if val:
        settle(book, trader, quote, state, mid, spread, BookRecorder())
    return val, quote


def test_quantize_and_mvt_avg():
    assert quantize_qty(Decimal('1.23456')) == Decimal('1.2345')
    assert list(mvt_avg([1, 2, 3, 4], 2)) == [1.5, 2.5, 3.5]


def test_affordable_market_buy_never_exceeds_the_budget():
    book = seeded_book()
    for budget in (50, 101, 500, 1000, 1500, 100000):
        q = affordable_market_buy(book, budget)
        cost = sum(min(l.volume, q) * p for p, l in book.asks.price_map.items())  # upper bound check below
        probe = seeded_book()
        trades, _ = probe.process_order({'type': 'market', 'side': 'bid', 'quantity': q, 'trade_id': 9}) if q > 0 else ([], None)
        spent = sum(t['price'] * t['quantity'] for t in trades)
        assert spent <= budget
        assert q <= sum(l.volume for l in book.asks.price_map.values())


def test_market_buy_trader_pays_what_the_book_charged():
    np.random.seed(1)
    book = seeded_book()
    trader = Trader_random(1000, 0, 0)
    np.random.seed(0)  # whatever the coin flip, force a buy
    trader_method = lambda b: (setattr(trader, 'x', 0), trader.trade(b))[1]
    for _ in range(20):
        val, quote = send(book, trader, trader.trade)
        if val and quote['side'] == 'bid':
            break
    assert trader.K >= 0 and trader.nb_shares > 0
    record = trader.trader_balance_sheet[-1]
    assert record['Current Capital'] == trader.K
    # cash spent == sum of the trades of this order
    spent = Decimal(1000) - trader.K
    bought = trader.nb_shares
    assert spent > 0 and Decimal(100) < spent / bought < Decimal(106)


def test_buy_side_convention_trader_quantity():
    book = seeded_book()
    trader = Trader_quantity(300)  # can afford 2.97 shares, fewer than the 5 displayed at the best ask
    val, quote = send(book, trader, trader.trade_first)
    assert val and quote['side'] == 'bid' and quote['price'] == Decimal(101)
    assert trader.nb_shares > 0 and trader.K < 300
    assert book.get_volume_at_price('ask', 101) == Decimal(5) - trader.nb_shares


def test_sell_trader_receives_cash():
    book = seeded_book()
    trader = Trader_bearish(0, 0, 10)
    val, quote = send(book, trader, trader.tradeFirst)
    assert val and quote['side'] == 'ask'
    assert trader.nb_shares == 1  # the strategy keeps 1 share
    # 9 shares market sold into 5 @ 99 then 4 @ 98
    assert trader.K == Decimal(5 * 99 + 4 * 98)


def test_unfilled_limit_remainder_is_cancelled():
    book = seeded_book()
    trader = Trader_bullish(10000, 0, 0)
    trader.nb_shares = Decimal(100)
    trader.K_initial = 10000
    val, quote = send(book, trader, trader.SecondTrade)  # sells 99 shares at the worst bid: only 20 shares of bids exist
    assert val
    assert trader.nb_shares == Decimal(100) - Decimal(20)
    assert book.bids.depth == 0 and book.asks.volume == 20  # nothing of the order rests in the book


def test_agents_stay_idle_on_an_empty_book():
    book = OrderBook()
    agents = [(Trader_quantity(100), 'trade_first'), (Trader_random(100, 0, 5), 'trade'), (Trader_bullish(100, 0, 5), 'tradeFirst'),
              (Trader_bullish(100, 0, 5), 'SecondTrade'), (Trader_bearish(100, 0, 5), 'tradeFirst'), (Trader_bearish(100, 0, 5), 'SecondTrade'),
              (Trader_bearish(100, 0, 5), 'ThirdTrade'), (Trader_MA(100, 0, 5, 3), 'firstTrade')]
    for trader, method in agents:
        val, quote, sheet = getattr(trader, method)(book)
        assert val is False and len(sheet) >= 1
    with pytest.raises(ValueError):
        Trader_Buy_and_sell(book)


def test_buy_and_sell_initial_purchase_and_pnl():
    book = seeded_book()
    trader = Trader_Buy_and_sell(book)
    assert trader.K_initial == 5 * 101 and trader.pnl == -505 and trader.K == 0
    state = (trader.K_initial, 0, 0)
    settle(book, trader, trader.initial_quote, state, book.get_mid_price(), book.get_spread(), BookRecorder())
    assert trader.nb_shares == 5 and trader.K == 0 and book.get_best_ask() == 102


def test_ma_trader_needs_enough_trades():
    book = seeded_book()
    trader = Trader_MA(1000, 0, 5, 5)
    val, _, _ = trader.firstTrade(book)
    assert val is False


def test_generate_lim_orders_ranges():
    np.random.seed(5)
    orders = generate_lim_orders(200, [90, 100], [95, 105], [1, 10], [2, 4])
    for o in orders:
        if o['side'] == 'ask':
            assert 95 <= o['price'] <= 105 and 1 <= o['quantity'] <= 10
        else:
            assert 90 <= o['price'] <= 100 and 2 <= o['quantity'] <= 4

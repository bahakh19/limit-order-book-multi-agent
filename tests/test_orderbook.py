import random
from decimal import Decimal

import pytest

from orderbook import OrderBook, to_decimal


def limit(side, quantity, price, trade_id=0):
    return {'type': 'limit', 'side': side, 'quantity': quantity, 'price': price, 'trade_id': trade_id}


def market(side, quantity, trade_id=0):
    return {'type': 'market', 'side': side, 'quantity': quantity, 'trade_id': trade_id}


def check_invariants(book):
    for tree in (book.bids, book.asks):
        assert tree.depth == len(tree.price_map)
        assert tree.num_orders == len(tree.order_map)
        assert tree.volume == sum(o.quantity for o in tree.order_map.values())
        for price, order_list in tree.price_map.items():
            orders = list(order_list)
            assert len(order_list) == len(orders) > 0
            assert order_list.volume == sum(o.quantity for o in orders)
            assert all(o.price == price for o in orders)
            assert order_list.head_order is orders[0] and order_list.tail_order is orders[-1]
    if book.bids.depth and book.asks.depth:
        assert book.get_best_bid() < book.get_best_ask(), 'the book is crossed'


def test_to_decimal_removes_float_noise():
    assert to_decimal(0.1) == Decimal('0.1')
    assert to_decimal(3) == Decimal(3)
    assert to_decimal('2.5') == Decimal('2.5')


def test_price_time_priority():
    book = OrderBook()
    book.process_order(limit('ask', 5, 101, 1))
    book.process_order(limit('ask', 5, 101, 2))
    book.process_order(limit('ask', 5, 100, 3))
    trades, _ = book.process_order(market('bid', 12, 9))
    # best price first (100), then the oldest order at 101
    assert [(t['price'], t['quantity'], t['party1'][0]) for t in trades] == [
        (Decimal(100), Decimal(5), 3), (Decimal(101), Decimal(5), 1), (Decimal(101), Decimal(2), 2)]
    assert book.get_volume_at_price('ask', 101) == 3
    check_invariants(book)


def test_aggressor_side_in_the_tape():
    book = OrderBook()
    book.process_order(limit('bid', 5, 99, 1))
    trades, _ = book.process_order(market('ask', 2, 7))
    assert trades[0]['party1'][1] == 'bid' and trades[0]['party2'][1] == 'ask'


def test_partial_fill_updates_every_volume():
    book = OrderBook()
    book.process_order(limit('ask', 10, 101, 1))
    book.process_order(market('bid', 4, 2))
    assert book.asks.volume == 6
    assert book.get_volume_at_price('ask', 101) == 6
    check_invariants(book)


def test_crossing_limit_order_rests_the_remainder():
    book = OrderBook()
    book.process_order(limit('ask', 3, 101, 1))
    trades, resting = book.process_order(limit('bid', 10, 102, 2))
    assert sum(t['quantity'] for t in trades) == 3
    assert resting['quantity'] == 7 and book.get_best_bid() == 102 and book.get_best_ask() is None
    check_invariants(book)


def test_market_order_on_empty_side_does_nothing():
    book = OrderBook()
    trades, resting = book.process_order(market('bid', 5, 1))
    assert trades == [] and resting is None


def test_cancel_order():
    book = OrderBook()
    _, o1 = book.process_order(limit('bid', 5, 99, 1))
    book.process_order(limit('bid', 5, 99, 2))
    book.cancel_order('bid', o1['order_id'])
    assert book.get_volume_at_price('bid', 99) == 5 and len(book.bids) == 1
    check_invariants(book)
    book.cancel_order('bid', 12345)  # unknown id: ignored


def test_modify_quantity_keeps_or_loses_priority():
    book = OrderBook()
    _, a = book.process_order(limit('ask', 5, 101, 1))
    book.process_order(limit('ask', 5, 101, 2))
    book.modify_order(a['order_id'], {'side': 'ask', 'quantity': 3, 'price': 101})  # smaller: keeps its place
    assert book.asks.get_price_list(Decimal(101)).get_head_order().trade_id == 1
    book.modify_order(a['order_id'], {'side': 'ask', 'quantity': 8, 'price': 101})  # larger: goes to the back
    assert book.asks.get_price_list(Decimal(101)).get_head_order().trade_id == 2
    check_invariants(book)
    assert book.asks.volume == 13


def test_modify_price_moves_the_order_between_levels():
    book = OrderBook()
    _, a = book.process_order(limit('ask', 5, 101, 1))
    book.process_order(limit('ask', 4, 101, 2))
    book.modify_order(a['order_id'], {'side': 'ask', 'quantity': 6, 'price': 105})
    assert book.get_volume_at_price('ask', 101) == 4 and book.get_volume_at_price('ask', 105) == 6
    assert book.asks.depth == 2 and book.asks.volume == 10 and book.asks.num_orders == 2
    check_invariants(book)
    # the last order of a level moves: the old level disappears
    book.modify_order(a['order_id'], {'side': 'ask', 'quantity': 6, 'price': 106.5})
    assert not book.asks.price_exists(Decimal(105)) and book.get_volume_at_price('ask', 106.5) == 6
    check_invariants(book)


def test_empty_book_accessors():
    book = OrderBook()
    bids, asks, trades = book.get_all()
    assert bids == [] and asks == [] and trades == []
    assert book.get_mid_price() is None and book.get_spread() is None
    assert 'Bids' in str(book)


def test_mid_and_spread():
    book = OrderBook()
    book.process_order(limit('bid', 1, 99))
    book.process_order(limit('ask', 1, 102))
    assert book.get_spread() == 3 and book.get_mid_price() == Decimal('100.5')


@pytest.mark.parametrize('quote', [limit('bid', 0, 100), limit('bid', -1, 100), {'type': 'stop', 'side': 'bid', 'quantity': 1, 'trade_id': 0}])
def test_invalid_orders_raise(quote):
    with pytest.raises(ValueError):
        OrderBook().process_order(quote)


def test_invalid_side_raises():
    book = OrderBook()
    with pytest.raises(ValueError):
        book.process_order(limit('buy', 1, 100))
    with pytest.raises(ValueError):
        book.cancel_order('buy', 1)
    with pytest.raises(ValueError):
        book.get_volume_at_price('buy', 1)


def test_tape_dump(tmp_path):
    book = OrderBook()
    book.process_order(limit('ask', 5, 101, 1))
    book.process_order(market('bid', 2, 2))
    path = tmp_path / 'tape.txt'
    book.tape_dump(str(path), 'w', 'wipe')
    assert 'Price: 101' in path.read_text() and len(book.tape) == 0
    book.process_order(market('bid', 1, 3))  # the tape still works after a wipe
    assert len(book.tape) == 1


def test_random_stress_keeps_every_invariant():
    rng = random.Random(7)
    book = OrderBook()
    live = []
    for _ in range(3000):
        action = rng.random()
        if action < 0.55:
            side = rng.choice(['bid', 'ask'])
            _, resting = book.process_order(limit(side, rng.randint(1, 9), rng.randint(90, 110), rng.randint(1, 50)))
            if resting is not None:
                live.append((side, resting['order_id']))
        elif action < 0.7:
            book.process_order(market(rng.choice(['bid', 'ask']), rng.randint(1, 15), 1))
        elif action < 0.85 and live:
            side, oid = live.pop(rng.randrange(len(live)))
            book.cancel_order(side, oid)
        elif live:
            side, oid = live[rng.randrange(len(live))]
            tree = book.bids if side == 'bid' else book.asks
            if tree.order_exists(oid):
                price = rng.randint(90, 110)
                # a modification must not cross the book, as in the real exchange
                other = book.get_best_ask() if side == 'bid' else book.get_best_bid()
                if other is None or (side == 'bid' and price < other) or (side == 'ask' and price > other):
                    book.modify_order(oid, {'side': side, 'quantity': rng.randint(1, 9), 'price': price})
        check_invariants(book)
    assert len(book.tape) > 0


def test_volume_is_conserved():
    book = OrderBook()
    total_in = 0
    rng = random.Random(3)
    for _ in range(500):
        q = rng.randint(1, 9)
        total_in += q
        book.process_order(limit(rng.choice(['bid', 'ask']), q, rng.randint(95, 105)))
    traded = sum(t['quantity'] for t in book.tape)
    # every trade removes its quantity from both sides
    assert book.bids.volume + book.asks.volume + 2 * traded == total_in

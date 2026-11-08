from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from orderbook import OrderBook
from ecology import (EcologyConfig, run_ecology, Exchange, Trader_noise_maker, Trader_market_maker, Trader_informed,
                     Trader_metaorder, Trader_latent_liquidity, fundamental_path)
from analysis import pnl_decomposition, markouts, flow_matrix, response_function, standardized_changes
from experiments import metaorder_experiment, agent_path, news_event_study, binned_acf, impact_table
from analysis import order_sign_acf

TICK = Decimal('0.01')


@pytest.fixture(scope='module')
def full():
    return run_ecology(EcologyConfig(seed=5, nb_events=6000, warmup=2000, use_strategies=True))


def test_tick_grid_is_enforced():
    book = OrderBook(tick_size=0.01)
    book.process_order({'type': 'limit', 'side': 'bid', 'quantity': 1, 'price': Decimal('100.01'), 'trade_id': 1})
    with pytest.raises(ValueError):
        book.process_order({'type': 'limit', 'side': 'bid', 'quantity': 1, 'price': Decimal('100.015'), 'trade_id': 1})
    with pytest.raises(ValueError):
        book.process_order({'type': 'limit', 'side': 'ask', 'quantity': 1, 'price': 0, 'trade_id': 1})


def test_exchange_routes_passive_fills_to_both_sides():
    book = OrderBook(tick_size=0.01)
    ex = Exchange(book)
    mm = ex.register(Trader_market_maker(1000, 0, 0, 1, 0.0, 5, 50, TICK), 'market maker')
    other = ex.register(Trader_informed(1000, 0, 0, 1, 3, 100, TICK), 'informed')
    mm.post(ex, 'ask', 10010, 5, TICK)
    ex.send(other, {'type': 'market', 'side': 'bid', 'quantity': 3})
    assert mm.nb_shares == -3 and other.nb_shares == 3
    assert mm.K == Decimal(1000) + 3*Decimal('100.10') and other.K == Decimal(1000) - 3*Decimal('100.10')
    assert len(mm.live) == 1  # partially filled, still resting
    ex.send(other, {'type': 'market', 'side': 'bid', 'quantity': 2})
    assert mm.live == {}      # fully filled: forgotten
    assert [f[4:] for f in ex.fills] == [(other.trade_id, mm.trade_id)]*2


def test_market_maker_quotes_never_cross_and_respect_inventory():
    book = OrderBook(tick_size=0.01)
    ex = Exchange(book)
    noise = ex.register(Trader_noise_maker(10**6, 0, 10**4, 5, 5, TICK, 10000), 'noise maker')
    for u in np.linspace(0.01, 0.99, 40):
        noise.limit_order(ex, u, (u*7) % 1, 0.5)
    mm = ex.register(Trader_market_maker(10**6, 0, 0, 0.5, 0.0, 5, 10, TICK), 'market maker')
    for _ in range(5):
        mm.requote(ex)
        assert book.get_best_bid() < book.get_best_ask()
        assert len(mm.live) == 2
    mm.nb_shares = Decimal(10)  # at the limit: he only sells
    mm.requote(ex)
    assert list(mm.live.values()) == ['ask']


def test_inventory_skew_moves_the_quotes():
    def quotes(inventory):
        book = OrderBook(tick_size=0.01)
        ex = Exchange(book)
        ex.send(ex.register(Trader_informed(0, 0, 0, 0, 0, 0, TICK), 'x'), {'type': 'limit', 'side': 'bid', 'quantity': 1, 'price': Decimal('99.00')})
        ex.send(ex.agents[1], {'type': 'limit', 'side': 'ask', 'quantity': 1, 'price': Decimal('101.00')})
        mm = ex.register(Trader_market_maker(10**6, 0, inventory, 2, 0.1, 5, 1000, TICK), 'market maker')
        mm.requote(ex)
        prices = {side: book.get_best_bid() if side == 'bid' else book.get_best_ask() for side in mm.live.values()}
        return prices
    flat, long = quotes(0), quotes(100)
    assert long['ask'] < flat['ask'] and long['bid'] < flat['bid']


def test_metaorder_sizes_follow_a_pareto_law():
    m = Trader_metaorder(0, 0, 0, 1.5, 10, 10**9, 5)
    u = np.random.default_rng(0).random(20000)
    sizes = np.array([int(10*(1 - x)**(-1/1.5)) for x in u])
    assert abs((sizes > 100).mean() - (10/100)**1.5) < 0.01


def test_same_seed_same_market():
    a = run_ecology(EcologyConfig(seed=9, nb_events=3000, warmup=1000))
    b = run_ecology(EcologyConfig(seed=9, nb_events=3000, warmup=1000))
    np.testing.assert_array_equal(a.mids, b.mids)
    pd.testing.assert_frame_equal(a.fills, b.fills)


def test_fundamental_path_has_its_own_generator():
    cfg = EcologyConfig(seed=3, nb_events=1000, warmup=10)
    np.testing.assert_array_equal(fundamental_path(cfg), fundamental_path(cfg))


def test_pnl_decomposition_equals_the_accounts(full):
    d = pnl_decomposition(full).set_index('agent')
    m_end, m0 = full.snapshots['mid'].iloc[-1], full.snapshots['mid'].iloc[0]
    agents = list(full.strategies.items()) + [('market maker', full.mm), ('informed', full.informed)] + [('metaorder', m) for m in full.meta]
    checked = 0
    for name, t in agents:
        cash0, shares0 = full.strategy_start.get(name, (float(t.K_initial), 0.0))
        account = float(t.K)/0.01 + float(t.nb_shares)*m_end - (cash0/0.01 + shares0*m0)
        total = d.loc[t.trade_id, 'total'] if t.trade_id in d.index else 0.0
        assert account == pytest.approx(total, abs=1e-6)
        checked += 1
    assert checked == 6 + 2 + 3


def test_book_never_crossed_and_traders_solvent(full):
    assert (full.snapshots['spread'].dropna() > 0).all()
    for t in full.strategies.values():
        assert t.K >= 0 and t.nb_shares >= 0
    assert abs(float(full.mm.nb_shares)) <= full.mm.max_inventory


def test_flow_matrix_and_markouts(full):
    fm = flow_matrix(full.fills)
    assert fm.values.sum() == pytest.approx(1.0)
    assert 'market maker' not in fm.index  # the market maker never takes liquidity
    mk = markouts(full.fills, full.snapshots, [0, 10], 'market maker')
    at_fill = mk[mk['horizon'] == 0]
    assert (at_fill['markout'] > 0).all()  # a passive fill always earns at least half a tick against the mid


def test_metaorders_create_persistent_order_signs():
    r = run_ecology(EcologyConfig(seed=2, nb_events=15000, use_market_maker=False, use_informed=False))
    z = run_ecology(EcologyConfig(seed=2, nb_events=15000, use_market_maker=False, use_informed=False, use_metaorders=False))
    a = binned_acf(order_sign_acf([r.orders['sign'].values], 30))
    b = binned_acf(order_sign_acf([z.orders['sign'].values], 30))
    assert a['acf'].iloc[:5].mean() > 0.02
    assert abs(b['acf'].iloc[:5].mean()) < a['acf'].iloc[:5].mean()


def test_common_random_numbers_isolate_the_metaorder():
    paths, peaks = metaorder_experiment(sizes=(25, 50, 100, 200), seeds=(1, 2, 3), warmup_events=500, use_market_maker=False, use_informed=False, use_metaorders=False)
    before = paths[paths['t'] == 0]['impact']
    assert (before == 0).all()  # identical markets until the metaorder starts
    assert (peaks.groupby('Q')['peak'].mean() > 0).all()
    assert peaks.groupby('Q')['peak'].mean().loc[200] > peaks.groupby('Q')['peak'].mean().loc[50]
    table = impact_table({'x': (paths, peaks)})
    assert table.loc['x', 'exponent (all Q)'] > 0


def test_latent_liquidity_profile_grows_with_distance():
    r = run_ecology(EcologyConfig(seed=4, nb_events=4000, use_market_maker=False, use_informed=False, use_metaorders=False,
                                  use_latent=True, latent_tau=1e9, latent_rate=40, latent_window=60))
    book, tick = r.order_book, Decimal('0.01')
    centre = round(r.latent.consensus)
    near = sum(l.volume for p, l in book.asks.price_map.items() if centre < p/tick <= centre + 20)
    far = sum(l.volume for p, l in book.asks.price_map.items() if centre + 40 < p/tick <= centre + 60)
    assert far > near


def test_agent_path_and_news_study(full):
    p = agent_path(full, 'market maker')
    assert p['inventory'].iloc[-1] == pytest.approx(float(full.mm.nb_shares))
    event, n = news_event_study([full])
    assert n >= 0

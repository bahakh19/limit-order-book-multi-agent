import numpy as np
import pandas as pd
import pytest

from simulation import (SimulationConfig, run_simulation, STRATEGIES, GetStats, GetList, end_of_period_tables, export_tables)


@pytest.fixture(scope='module')
def result():
    return run_simulation(SimulationConfig(nb_trader_per_strat=12, nb_transac_times=25))


def test_same_seed_same_results(result):
    again = run_simulation(SimulationConfig(nb_trader_per_strat=12, nb_transac_times=25))
    pd.testing.assert_frame_equal(result.metrics, again.metrics)
    assert result.trajecs == again.trajecs


def test_different_seed_different_results(result):
    other = run_simulation(SimulationConfig(nb_trader_per_strat=12, nb_transac_times=25, seed=7))
    assert not result.metrics['mtm_pnl'].equals(other.metrics['mtm_pnl'])


def test_traders_are_solvent_and_never_short(result):
    # the accounting follows the fills of the book, so nobody can spend money or sell shares he does not have
    assert (result.metrics['final_cash'] >= -1e-9).all()
    assert (result.metrics['final_shares'] >= -1e-9).all()


def test_shape_of_the_results(result):
    assert len(result.metrics) == 12 * len(STRATEGIES)
    assert set(result.metrics['strategy']) == set(STRATEGIES)
    for name in STRATEGIES:
        assert len(result.sheets[name]) == len(result.final_dist[name]) == 12
    assert len(result.tapes) == len(result.snapshots) == len(result.orders) == 12


def test_costs_are_consistent(result):
    m = result.metrics
    assert (m['execution_cost'] >= m['half_spread_cost'] - 1e-9).all()
    assert (m['half_spread_cost'] >= 0).all()


def test_mark_to_market_identity(result):
    # excess pnl = total mark-to-market pnl - the pnl of just holding the initial shares
    m = result.metrics
    assert np.allclose(m['excess_pnl'], m['mtm_pnl'] - m['passive_pnl'])


def test_book_is_never_crossed_in_the_snapshots(result):
    for snap in result.snapshots:
        spread = snap['spread'].dropna()
        assert (spread > 0).all()


def test_independent_simulations_restart_from_a_new_book():
    r = run_simulation(SimulationConfig(nb_trader_per_strat=3, nb_transac_times=5))
    # every simulation starts from a fresh book seeded with the same number of starting orders, so the clock starts at the same value
    assert len({s['time'].iloc[0] for s in r.snapshots}) == 1


def test_config_validation():
    with pytest.raises(ValueError):
        SimulationConfig(price_range_bid_min=101, price_range_bid_max=100)
    with pytest.raises(ValueError):
        SimulationConfig(qty_range_ask_min=0)
    with pytest.raises(ValueError):
        SimulationConfig(price_range_ask_min=95, price_range_ask_max=96)


def test_getstats_and_tables(result, tmp_path):
    out = GetStats(result.final_dist['Trader_random'], 'Trader_random', result.order_book, result.trajecs)
    assert out[0].loc['count'].iloc[0] == 12
    tables = end_of_period_tables(result)
    assert list(tables['Trader_MA'].columns) == ['Current Capital', 'Cumulative Pnl', 'Gross Return']
    export_tables(tables, 12, str(tmp_path))
    assert len(list(tmp_path.glob('*.xlsx'))) == len(STRATEGIES) == len(list(tmp_path.glob('*.csv')))


def test_getlist_draws_and_saves(result, tmp_path):
    path = tmp_path / 'fig.png'
    GetList(result.sheets['Trader_random'], 'Trader_random', result.all_trajecs, save_path=str(path))
    assert path.stat().st_size > 1000

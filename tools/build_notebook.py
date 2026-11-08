"""Builds notebooks/limit_order_book.ipynb from the modules in src/.

The notebook stays self contained (one cell per module: the engine, the demo, the strategies, the measurements, the
baseline market, the ecology and the studies) and the modules in src/ are the same code, so that the tests exercise exactly what the notebook runs.
tests/test_notebook.py checks that both stay in sync. Run from the repository root:  python tools/build_notebook.py
"""
import ast
import hashlib
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL = {'orderbook', 'agents', 'analysis', 'simulation', 'ecology', 'experiments'}


def module_source(name):
    """Source of a module of src/ without its imports of the other local modules (they are defined in earlier cells)."""
    text = open(os.path.join(ROOT, 'src', name), encoding='utf-8').read()
    lines = text.split('\n')
    drop = set()
    for node in ast.parse(text).body:
        if isinstance(node, ast.ImportFrom) and node.module in LOCAL:
            drop.update(range(node.lineno - 1, node.end_lineno))
    kept = [l for i, l in enumerate(lines) if i not in drop]
    return '\n'.join(kept).strip('\n') + '\n'


def cell(kind, text, tag):
    lines = text.rstrip('\n').split('\n')
    source = [l + '\n' for l in lines[:-1]] + [lines[-1]]
    c = {'cell_type': kind, 'id': hashlib.sha1(tag.encode()).hexdigest()[:8], 'metadata': {}, 'source': source}
    if kind == 'code':
        c.update({'execution_count': None, 'outputs': []})
    return c


MD = {}
CODE = {}

MD['title'] = """# Limit order book and multi-agent market simulation

A limit order book (LOB) exchange built from scratch, a population of trading agents that interact through it, and a set of measurements taken from *Trades, Quotes and Prices* (Bouchaud, Bonart, Donier, Gould, Cambridge University Press, 2018). The study asks two questions: **what does a trading strategy really earn once it pays the spread and moves the price**, and **which behaviours of the market participants produce the statistical regularities of real markets**.

1. **The matching engine**: price-time priority, market and limit orders, cancel and modify, a tick grid.
2. **The rules of the exchange**, shown on a small book.
3. **Six directional strategies** (quantity, buy and sell, random, bullish, bearish, moving average) and a **baseline market** where they trade against a uniform random flow.
4. to 6. **The baseline market** measured with the tools of the book, and what the strategies earn in it.
7. **The market ecology**: zero-intelligence liquidity, a market maker, an informed trader, institutional metaorders and latent liquidity, all trading through an exchange that settles every agent on his real fills, passive ones included.
8. to 14. **Studies**: one run of the ecology, which ingredient creates which stylized fact, who trades with whom and who earns what, how news gets into the price, the six strategies in both markets, the optimal spread of a market maker, and the impact of a metaorder.
15. **Insights.**

Convention used everywhere: `'bid'` is a **buy** order and `'ask'` is a **sell** order. The sign of a trade is the sign of the aggressor: +1 if the aggressor buys, -1 if it sells.

The code lives in `src/` and is embedded cell by cell below, so the tests run exactly what this notebook runs. Every random generator is seeded, so every run gives the same numbers.
"""

MD['engine'] = """## 1. The matching engine

`Order`, `OrderList` (a doubly linked list of the orders at one price, oldest first), `OrderTree` (one side of the book, prices kept sorted) and `OrderBook`. The structure and the quote format (`type`, `side`, `quantity`, `price`, `trade_id`) follow the open source project [dyn4mik3/OrderBook](https://github.com/dyn4mik3/OrderBook). Prices and quantities are `Decimal`, limit prices must lie on the tick grid, and invalid orders raise `ValueError`.
"""

MD['demo'] = """## 2. The rules of the exchange

Eight non-crossing limit orders give the book some depth, then a crossing limit order, a big crossing limit order and a market order show how time priority, partial fills and liquidity consumption work.
"""

MD['agents'] = """## 3. Six directional strategies and the baseline market

### 3.1 The strategies

Each trader keeps a balance sheet (`Current Capital`, `Cumulative Pnl`, `Gross Return`, number of shares) and decides from the state of the book:

| Strategy | Idea |
|---|---|
| `Trader_quantity` | buys at the best ask when his capital buys fewer shares than the best ask displays, sells at the best bid when the best bid is higher than what he paid |
| `Trader_Buy_and_sell` | buys the whole best ask queue at the start, sells everything once the best bid is above his purchase price |
| `Trader_random` | flips a coin every time: market buy with all his cash or market sell of his shares |
| `Trader_bullish` | market buy with all his cash, then limit sell at the worst bid |
| `Trader_bearish` | market sell of his shares, then limit buy at the worst ask, then limit sell at the worst bid |
| `Trader_MA` | buys when the last trade price is above its moving average, sells when it is below |

A trader updates his balance sheet when he sends his order, as if it was fully executed at the quoted price. `settle` (section 3.3) then replaces this guess by what the book really executed. Orders are immediate-or-cancel: the part of a limit order that cannot be executed is cancelled.
"""

MD['analysis'] = """### 3.2 Measurement toolbox

Functions that measure a simulated market with the definitions of the book:

* spread distribution (chapter 4) and average volume at the best price levels (chapter 4.7),
* autocorrelation of the order signs $C(\\ell)$ (chapter 10),
* lag-1 impact $R(1) = \\langle \\varepsilon_t (m_{t+1} - m_t) \\rangle$, the response function $R(\\ell)$ and the dependence of impact on the order volume (chapter 11, Eq. 11.5 and 11.7),
* execution cost against the mid-price (half-spread and price impact), markouts, the flow matrix between kinds of agents and an exact decomposition of every agent's PnL,
* a static book experiment on the impact of a market order (chapters 12 and 18).
"""

MD['simulation'] = """### 3.3 The baseline market

`run_simulation` runs independent simulations of a deliberately simple market: a random book of limit orders with uniform prices and quantities, one trader of each strategy, and after some of their orders the rest of the market reacts with new uniform random limit orders (`REACTION_ORDERS`). It is the benchmark against which the market ecology of section 7 is compared.
"""

MD['params'] = """### 3.4 Parameters and run

All parameters are set in one place, and the random seed makes the run reproducible.
"""

CODE['params'] = """cfg = SimulationConfig(
    seed=42,
    nb_start_orders=200,
    price_range_bid_min=90, price_range_bid_max=100,
    price_range_ask_min=95, price_range_ask_max=105,
    qty_range_ask_min=1, qty_range_ask_max=10,
    qty_range_bid_min=1, qty_range_bid_max=10,
    nb_transac_times=50,          # number of times our traders can trade
    nb_trader_per_strat=100,      # number of independent simulations
    K_trader_qty=1000, pnl_trader_qty=0, nb_shares_trader_qty=0,
    K_trader_random=10000, pnl_trader_random=0, nb_shares_trader_random=50,
    K_trader_bullish=10000, pnl_trader_bullish=0, nb_shares_trader_bullish=0,
    K_trader_bearish=5000, pnl_trader_bearish=0, nb_shares_trader_bearish=100,
    K_trader_MA=10000, pnl_trader_MA=0, nb_shares_trader_MA=50, mvg_trader_MA=5,
)
result = run_simulation(cfg)
print("simulations:", cfg.nb_trader_per_strat, "| aggressive orders per simulation:", round(sum(len(o) for o in result.orders) / len(result.orders), 1))
"""

MD['paths'] = """### 3.5 Paths of each strategy

Median, 25%-75% and 5%-95% bands over the simulations of each strategy's balance sheet and of the trade price (a few individual simulations drawn thin).
"""

CODE['paths'] = """for name in STRATEGIES:
    GetList(result.sheets[name][:30], name, result.all_trajecs[:30])
"""

MD['tables'] = """### 3.6 End of period statistics

Descriptive statistics of the last balance sheet of each strategy over all the simulations. The tables are also written to `results/` (Excel and csv).
"""

CODE['tables'] = """tables = end_of_period_tables(result)
for name, table in tables.items():
    print("End-of-period descriptive statistics over {} simulations made with the strategy {} with the specified parameters : ".format(cfg.nb_trader_per_strat, name))
    print(table)
    print("###################")
export_tables(tables, cfg.nb_trader_per_strat, '../results')
"""

MD['facts'] = """## 4. What does the baseline market look like?

A simulator is only useful if the market it produces behaves like a market. We measure the simulated tape and book with the tools of the book and compare with what is known about real markets:

| Quantity | Real markets (the book) | Where |
|---|---|---|
| Order signs | positively autocorrelated, $C(\\ell)$ decays as a power law $\\ell^{-\\gamma}$ with $\\gamma < 1$ (long memory) | chapter 10.1 |
| Lag-1 impact of a market order | $R(1) > 0$, and concave in the order volume: $R(v,1) \\sim (v / V_{best})^{\\zeta}$ with $\\zeta$ between 0 and 0.3 | chapter 11.3, Eq. 11.5 and 11.7 |
| Spread | peaked at one tick for large-tick stocks, wide with an exponential tail for small-tick stocks | chapter 4.3 |
| Price changes | fat tails | chapter 2 |
"""

CODE['facts'] = """snaps = pd.concat(result.snapshots, ignore_index=True)
print("Spread (ticks):")
print(spread_summary(snaps).round(3).to_string())
print()
acf = order_sign_acf([o['sign'].values for o in result.orders], 30)
gamma = fit_power_law(acf['lag'], acf['acf'], acf['se'])
print("Order sign autocorrelation: C(1) = %.3f (standard error under independence %.3f)" % (acf.loc[0, 'acf'], acf.loc[0, 'se']))
print("Lags with a significantly positive C(l): %d of %d" % ((acf['acf'] > 2 * acf['se']).sum(), len(acf)))
print("Power-law exponent gamma:", "not estimable (no long memory detected)" if gamma is None else round(gamma, 2))
print()
r1, se, n = lag1_impact(result.orders)
print("Lag-1 impact R(1) = %.3f ticks (standard error %.3f, %d orders)" % (r1, se, n))
iv = impact_vs_volume(result.orders)
print("Exponent of the lag-1 impact against v / Vbest: zeta = %.2f" % fit_exponent(iv['v'], iv['lag1']))
print(iv.round(3).to_string())
print()
print("Mid-price changes:")
print(return_statistics(result.snapshots).round(3).to_string())
fig = plot_microstructure(result)
"""

MD['strategies'] = """## 5. What do the strategies earn in the baseline market?

The `Cumulative Pnl` of the balance sheets is a cash flow (it only counts what was spent and received), so a strategy that bought shares looks like it lost money. Here every trader is valued **mark to market**: final cash plus shares valued at the final mid-price, minus the same at the start. `excess_pnl` removes what the initial shares would have earned just by being held.

The execution cost of an order is measured against the mid-price just before it was sent. The half-spread is the part every aggressive order pays, the rest is the price impact of walking through the book (chapters 11 and 16).
"""

CODE['strategies'] = """m = result.metrics
summary = m.groupby('strategy').agg(
    mtm_pnl_mean=('mtm_pnl', 'mean'), mtm_pnl_std=('mtm_pnl', 'std'),
    excess_pnl_mean=('excess_pnl', 'mean'), share_profitable=('mtm_pnl', lambda x: (x > 0).mean()),
    orders_sent=('orders_sent', 'mean'), filled_volume=('filled_volume', 'mean'),
    half_spread_cost=('half_spread_cost', 'mean'), execution_cost=('execution_cost', 'mean'))
summary['impact_cost'] = summary['execution_cost'] - summary['half_spread_cost']
summary['t_stat'] = summary['mtm_pnl_mean'] / (summary['mtm_pnl_std'] / np.sqrt(cfg.nb_trader_per_strat))
print(summary.loc[STRATEGIES].round(2).to_string())
fig = plot_strategies(m)
"""

MD['sqrt'] = """### Why the impact is what it is: a static book experiment

The price impact of a market order of size $Q$ walking through a static book is the inverse of the cumulative volume available. With the same volume at every price level the impact is linear in $Q$ (the Kyle-like case, exponent 1). If the volume grows linearly with the distance to the best price, the impact grows like $\\sqrt{Q}$, which is the shape of Eq. 18.20 of the book for a linear volume profile and the order of magnitude of the empirical square-root law of Eq. 12.6 (exponent between 0.4 and 0.7). The experiment below uses the matching engine itself.
"""

CODE['sqrt'] = """fig = plot_square_root()
"""

MD['ecology'] = """## 7. The market ecology

The baseline market has a spread of 6 ticks with a tick worth 1% of the price, no persistence in its order flow, and no passive liquidity provider with a PnL. The ecology replaces the uniform flow by agents that interact, each one modelled on a chapter of the book. Prices are on a grid of 0.01 (1 basis point of a price of 100).

| Agent | Behaviour | Book |
|---|---|---|
| `Trader_noise_maker` | zero-intelligence limit orders 1 to 20 ticks from the opposite best quote, each cancelled at a constant rate | chapter 8 (Santa Fe) |
| `Trader_noise_taker` | zero-intelligence market orders, buy or sell with equal probability | chapter 8 |
| `Trader_market_maker` | quotes both sides around a reservation price `mid - skew x inventory`, cancels and requotes | chapters 16 and 17 |
| `Trader_informed` | sees the fundamental value (a random walk with news jumps), buys when the ask is too low, sells when the bid is too high | chapters 15 and 16 |
| `Trader_metaorder` | 3 institutions with Pareto metaorder sizes (alpha = 1.5), executed as child market orders of 5 shares | chapters 10.5 and 12 |
| `Trader_latent_liquidity` | slow investors whose orders sit around a slow consensus price, depth growing linearly with the distance (used in section 14) | chapters 18 and 19 |
| the six directional strategies | the classes of section 3, settled on their real fills | section 3 |

The `Exchange` gives every agent a unique id and, for every trade, tells **both** counterparties about their fill (`on_fill`), so passive liquidity providers have a real PnL. Every event draws the same four uniform numbers whatever happens, so two runs with the same seed see the same randomness: section 14 uses this to measure the impact of a metaorder by running the market twice, with and without it (common random numbers).
"""

MD['experiments'] = """The studies and their figures (same code as `src/experiments.py`):
"""

MD['overview'] = """## 8. One run of the ecology

20 runs of the full ecology with the six directional strategies (seeds 1 to 20, 20,000 events after a warm-up of 3,000 events of noise flow). The figure shows the first run.
"""

CODE['overview'] = """runs, eco_table = ecology_study()
print("runs:", len(runs), "| aggressive orders per run:", round(sum(len(r.orders) for r in runs)/len(runs)))
fig = plot_ecology_overview(runs[0])
"""

MD['ablation'] = """## 9. Which ingredient creates which stylized fact?

The same statistics on four markets: the baseline market, the zero-intelligence flow alone, zero intelligence plus metaorders, and the full ecology (6 runs each).
"""

CODE['ablation'] = """ab = ablation(baseline=result)
print(stylized_facts_table(ab).round(3).to_string())
fig = plot_stylized_facts(ab)
"""

MD['flows'] = """## 10. Who trades with whom, and who earns what

Left: share of the traded volume between each kind of aggressor and each kind of passive counterparty. Right: the **markout** of the market maker's passive fills, `s x (m(t + h) - price)` per share, split by who took his liquidity. At h = 0 it is the half-spread he earned, its evolution with h is the adverse selection.

Below: the PnL of every agent split exactly into the spread earned or paid against the mid, the price move in the 100 events after the fill, and the holding of the position until the end. The three parts add up to the mark-to-market PnL of the agent's own accounts (checked in `tests/test_ecology.py`).
"""

CODE['flows'] = """fig, fm, mk = plot_flows(runs)
print((100*fm).round(1).to_string())
fig, by_kind = plot_pnl_decomposition(eco_table)
print(by_kind.round(2).to_string())
"""

MD['news'] = """## 11. How news gets into the price

Event study over every jump of the fundamental value larger than 5 ticks, in the direction of the news.
"""

CODE['news'] = """event, n_news = news_event_study(runs)
print(event.loc[[0, 250, 500, 1000, 1500], ['mispricing', 'mm_inventory', 'mm_pnl', 'informed', 'metaorder', 'noise taker']].round(2).to_string())
fig = plot_news(event, n_news)
"""

MD['comparison'] = """## 12. The six strategies in both markets

The same six classes, with the same capital, in the baseline market and in the ecology. Costs and PnL are expressed in basis points so that the two markets (tick of 1 in the baseline market, 0.01 in the ecology) can be compared.
"""

CODE['comparison'] = """comparison = market_comparison(result, eco_table)
print(comparison.round(2).to_string())
fig = plot_market_comparison(comparison)
"""

MD['sweep'] = """## 13. How wide should a market maker quote?

The market maker's PnL for five half-spreads and three intensities of informed trading (5 runs of 12,000 events each).
"""

CODE['sweep'] = """sweep = market_maker_sweep()
print(sweep.groupby(['informed_rate', 'half_spread'])[['total', 'volume']].mean().round(1).unstack(0).to_string())
fig = plot_market_maker_sweep(sweep)
"""

MD['impact'] = """## 14. The impact of a metaorder: linear or square root?

A buy metaorder of Q shares (25 to 1600) is executed with child market orders of 5 shares every 4 events, in three markets: Santa Fe liquidity only, the full ecology, and a market whose liquidity is mostly latent (`Trader_latent_liquidity` around a consensus price that does not move, a thin layer of noise liquidity at the touch). For every seed the market is run twice with the same random numbers, with and without the metaorder: the difference of the two mid-price paths is the impact of the metaorder alone. 16 seeds per market and size.
"""

CODE['impact'] = """study = impact_study()
print(impact_table(study).round(2).to_string())
fig = plot_metaorder(study)
"""

MD['conclusion'] = "@@CONCLUSION@@"


def build():
    cells = [
        cell('markdown', MD['title'], 'title'),
        cell('markdown', MD['engine'], 'engine-md'),
        cell('code', module_source('orderbook.py'), 'engine'),
        cell('markdown', MD['demo'], 'demo-md'),
        cell('code', module_source('demo.py'), 'demo'),
        cell('markdown', MD['agents'], 'agents-md'),
        cell('code', module_source('agents.py'), 'agents'),
        cell('markdown', MD['analysis'], 'analysis-md'),
        cell('code', module_source('analysis.py'), 'analysis'),
        cell('markdown', MD['simulation'], 'simulation-md'),
        cell('code', module_source('simulation.py'), 'simulation'),
        cell('markdown', MD['params'], 'params-md'),
        cell('code', CODE['params'], 'params'),
        cell('markdown', MD['paths'], 'paths-md'),
        cell('code', CODE['paths'], 'paths'),
        cell('markdown', MD['tables'], 'tables-md'),
        cell('code', CODE['tables'], 'tables'),
        cell('markdown', MD['facts'], 'facts-md'),
        cell('code', CODE['facts'], 'facts'),
        cell('markdown', MD['strategies'], 'strategies-md'),
        cell('code', CODE['strategies'], 'strategies'),
        cell('markdown', MD['sqrt'], 'sqrt-md'),
        cell('code', CODE['sqrt'], 'sqrt'),
        cell('markdown', open(os.path.join(ROOT, 'tools', 'conclusion.md'), encoding='utf-8').read(), 'conclusion'),
        cell('markdown', MD['ecology'], 'ecology-md'),
        cell('code', module_source('ecology.py'), 'ecology'),
        cell('markdown', MD['experiments'], 'experiments-md'),
        cell('code', module_source('experiments.py'), 'experiments'),
    ]
    for key in ('overview', 'ablation', 'flows', 'news', 'comparison', 'sweep', 'impact'):
        cells.append(cell('markdown', MD[key], key + '-md'))
        cells.append(cell('code', CODE[key], key))
    cells.append(cell('markdown', open(os.path.join(ROOT, 'tools', 'insights.md'), encoding='utf-8').read(), 'insights'))
    nb = {'cells': cells,
          'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                       'language_info': {'name': 'python'}},
          'nbformat': 4, 'nbformat_minor': 5}
    out = os.path.join(ROOT, 'notebooks', 'limit_order_book.ipynb')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write('\n')
    return out


if __name__ == '__main__':
    print(build())

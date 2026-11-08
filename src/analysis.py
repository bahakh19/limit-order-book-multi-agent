# Microstructure diagnostics of a simulated limit order book.
# The definitions follow Bouchaud, Bonart, Donier and Gould, "Trades, Quotes and Prices"
# (Cambridge University Press, 2018): chapter 4 (spread and depth), chapter 10 (persistence of the order flow),
# chapter 11 (lag-1 impact of market orders, Eq. 11.5 and 11.7), chapter 12 (impact of metaorders, Eq. 12.6).
# Convention used all along: the sign of an order is +1 when the aggressor BUYS (it hits the asks) and -1 when it SELLS.
from itertools import islice
from decimal import Decimal

import numpy as np
import pandas as pd

from orderbook import OrderBook


class BookRecorder:
    '''
    Records a snapshot of the top of the book after every event (order processed or cancelled).
    A snapshot holds the best quotes and the price/volume of the `levels` best price levels of each side.
    '''
    def __init__(self, levels=5):
        self.levels = levels
        self.rows = []

    def record(self, order_book):
        row = {'time': order_book.time}
        bids = list(islice(reversed(order_book.bids.price_map.items()), self.levels))
        asks = list(islice(iter(order_book.asks.price_map.items()), self.levels))
        for k in range(self.levels):
            if k < len(bids):
                row['bid_p%d' % (k + 1)] = float(bids[k][0])
                row['bid_q%d' % (k + 1)] = float(bids[k][1].volume)
            else:
                row['bid_p%d' % (k + 1)] = np.nan
                row['bid_q%d' % (k + 1)] = 0.0
            if k < len(asks):
                row['ask_p%d' % (k + 1)] = float(asks[k][0])
                row['ask_q%d' % (k + 1)] = float(asks[k][1].volume)
            else:
                row['ask_p%d' % (k + 1)] = np.nan
                row['ask_q%d' % (k + 1)] = 0.0
        row['bid_volume'] = float(order_book.bids.volume)
        row['ask_volume'] = float(order_book.asks.volume)
        self.rows.append(row)

    def frame(self):
        df = pd.DataFrame(self.rows)
        df['mid'] = (df['bid_p1'] + df['ask_p1']) / 2
        df['spread'] = df['ask_p1'] - df['bid_p1']
        return df


def tape_frame(tape):
    '''Trade tape -> DataFrame (time, price, quantity, sign). The sign is the one of the aggressor.'''
    rows = [(t['time'], float(t['price']), float(t['quantity']), 1 if t['party2'][1] == 'bid' else -1) for t in tape]
    return pd.DataFrame(rows, columns=['time', 'price', 'quantity', 'sign'])


def order_frame(trades, snapshots):
    '''
    Aggregates the fills of every aggressive order (market order or marketable limit order), the
    "market order time" of the book. mid_before is the mid-price just before the order arrives, mid_after
    the one just after it was processed and vbest the volume displayed at the opposite best quote before it arrived.
    '''
    if len(trades) == 0:
        return pd.DataFrame(columns=['time', 'sign', 'volume', 'vwap', 'mid_before', 'mid_after', 'vbest'])
    trades = trades.assign(notional=trades['price'] * trades['quantity'])
    g = trades.groupby('time')
    orders = pd.DataFrame({'sign': g['sign'].first(), 'volume': g['quantity'].sum(), 'notional': g['notional'].sum()}).reset_index()
    orders['vwap'] = orders['notional'] / orders['volume']
    snaps = snapshots[['time', 'mid', 'spread', 'bid_q1', 'ask_q1']].sort_values('time')
    before = pd.merge_asof(orders[['time']], snaps, on='time', allow_exact_matches=False)
    after = pd.merge_asof(orders[['time']], snaps, on='time', allow_exact_matches=True)
    orders['mid_before'] = before['mid'].values
    orders['spread_before'] = before['spread'].values
    orders['mid_after'] = after['mid'].values
    orders['vbest'] = np.where(orders['sign'] > 0, before['ask_q1'].values, before['bid_q1'].values)
    return orders.drop(columns='notional')


def spread_summary(snapshots, tick=1.0):
    '''Distribution of the bid-ask spread expressed in ticks (chapter 4.3)'''
    s = (snapshots['spread'].dropna() / tick)
    return pd.Series({'mean': s.mean(), 'median': s.median(), 'p5': s.quantile(0.05), 'p95': s.quantile(0.95),
                      'share of 1 tick': (s <= 1).mean(), 'observations': len(s)})


def order_sign_acf(sign_series, max_lag=50):
    '''
    Autocorrelation C(l) = E[e_t e_(t+l)] - E[e]^2 of the order signs (chapter 10, Eq. 10.1),
    pooled over a list of independent simulations. se = 1/sqrt(number of pairs) is the standard error
    under the hypothesis of independent signs.
    '''
    allsigns = np.concatenate([np.asarray(s, dtype=float) for s in sign_series])
    mean = allsigns.mean()
    rows = []
    for lag in range(1, max_lag + 1):
        num, den = 0.0, 0
        for s in sign_series:
            s = np.asarray(s, dtype=float)
            if len(s) > lag:
                num += np.sum((s[:-lag] - mean) * (s[lag:] - mean))
                den += len(s) - lag
        rows.append((lag, num / den if den else np.nan, 1 / np.sqrt(den) if den else np.nan, den))
    return pd.DataFrame(rows, columns=['lag', 'acf', 'se', 'pairs'])


def fit_power_law(lags, values, se, min_points=5):
    '''
    Slope -gamma of log C(l) against log l, on the lags where C(l) is positive and larger than twice its standard error.
    Returns None when fewer than min_points lags qualify: no long memory can be claimed then.
    '''
    lags, values, se = np.asarray(lags, dtype=float), np.asarray(values, dtype=float), np.asarray(se, dtype=float)
    ok = values > 2 * se
    if ok.sum() < min_points:
        return None
    slope = np.polyfit(np.log(lags[ok]), np.log(values[ok]), 1)[0]
    return -slope


def lag1_impact(orders_by_simulation):
    '''
    Lag-1 unconditional impact R(1) = < e_t (m_(t+1) - m_t) > in market order time (chapter 11, Eq. 11.5), where m_t is
    the mid-price just before the t-th aggressive order. Returns (R(1), standard error, number of orders).
    '''
    values = []
    for o in orders_by_simulation:
        m = o['mid_before'].values
        e = o['sign'].values
        values.append(e[:-1] * (m[1:] - m[:-1]))
    values = np.concatenate(values) if values else np.array([])
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan, 0
    se = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.nan
    return values.mean(), se, len(values)


def impact_vs_volume(orders_by_simulation, nbins=8):
    '''
    Mean signed mid-price change as a function of the order volume normalised by the volume at the best opposite quote.
    immediate = m_after - m_before (mechanical impact of the order itself)
    lag1 = m_(t+1) - m_t, the quantity of Eq. 11.7 whose volume dependence is R(v, 1) ~ (v / Vbest)^zeta, zeta in 0-0.3 in the book.
    '''
    frames = []
    for o in orders_by_simulation:
        d = o.copy()
        d['lag1'] = d['sign'] * (d['mid_before'].shift(-1) - d['mid_before'])
        d['immediate'] = d['sign'] * (d['mid_after'] - d['mid_before'])
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d = d[(d['vbest'] > 0)].dropna(subset=['immediate'])
    d['v'] = d['volume'] / d['vbest']
    d['bin'] = pd.qcut(d['v'], nbins, duplicates='drop')
    out = d.groupby('bin', observed=True).agg(v=('v', 'mean'), immediate=('immediate', 'mean'), lag1=('lag1', 'mean'), orders=('v', 'size')).reset_index(drop=True)
    return out


def fit_exponent(x, y, min_points=4):
    '''Exponent of y ~ x^zeta by least squares on log-log values, on the positive points'''
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = (x > 0) & (y > 0)
    if ok.sum() < min_points:
        return np.nan
    return np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)[0]


def average_depth_profile(snapshots_list, levels=5):
    '''Average volume at the k-th best level of the bid side and of the ask side (chapter 4.7)'''
    cat = pd.concat(snapshots_list, ignore_index=True)
    bid = [cat['bid_q%d' % k].mean() for k in range(1, levels + 1)]
    ask = [cat['ask_q%d' % k].mean() for k in range(1, levels + 1)]
    return pd.DataFrame({'level': range(1, levels + 1), 'bid': bid, 'ask': ask})


def return_statistics(snapshots_list):
    '''Mid-price changes per event: standard deviation and excess kurtosis (fat tails, chapter 2)'''
    r = np.concatenate([s['mid'].diff().dropna().values for s in snapshots_list])
    r = r[np.isfinite(r)]
    r = r[r != 0] if (r != 0).any() else r
    return pd.Series({'non-zero mid changes': len(r), 'std': r.std(), 'excess kurtosis': pd.Series(r).kurt()})


def execution_costs(sheets):
    '''
    Execution cost of the orders of a balance sheet versus the mid-price just before each order was sent.
    Returns (half-spread cost, total cost, filled volume); total cost - half-spread cost is the price impact paid on top of the spread.
    '''
    spread_cost, total_cost, volume = 0.0, 0.0, 0.0
    for record in sheets:
        f = record.get('filled_quantity')
        if f is None or float(f) == 0:
            continue
        f = float(f)
        half_spread, slippage = 0.5 * float(record['spread_before']), float(record['slippage'])
        if np.isnan(half_spread) or np.isnan(slippage):
            continue
        spread_cost += half_spread * f
        total_cost += slippage * f
        volume += f
    return spread_cost, total_cost, volume


def static_book_impact(profile, quantities, nb_levels=3000, step=0.01, best_ask=100.0):
    '''
    Mechanical impact of a market buy of size Q on a static ask side, measured with the matching engine itself: I(Q) = ask after - ask before.
    profile = 'flat'   : the same volume at every price level            -> I(Q) ~ Q       (linear, Kyle-like impact)
    profile = 'linear' : volume proportional to the distance to the best  -> I(Q) ~ sqrt(Q) (the form of Eq. 18.20 for a linear volume profile)
    '''
    rows = []
    for q in quantities:
        book = OrderBook()
        for k in range(nb_levels):
            volume = 1 if profile == 'flat' else k + 1
            price = Decimal(str(best_ask)) + Decimal(str(step)) * k
            book.process_order({'type': 'limit', 'side': 'ask', 'quantity': volume, 'price': price, 'trade_id': 0}, False, False)
        before = book.get_best_ask()
        book.process_order({'type': 'market', 'side': 'bid', 'quantity': q, 'trade_id': 1}, False, False)
        rows.append((q, float(book.get_best_ask() - before)))
    return pd.DataFrame(rows, columns=['quantity', 'impact'])


# Colours of the six strategies (fixed order, validated categorical palette) and of the chart ink
COLORS = {'Trader_quantity': '#2a78d6', 'Trader_Buy_and_sell': '#eb6834', 'Trader_random': '#1baf7a',
          'Trader_bullish': '#eda100', 'Trader_bearish': '#e87ba4', 'Trader_MA': '#008300'}
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def _style(ax, title, xlabel='', ylabel=''):
    ax.set_title(title, fontsize=11, color=INK, loc='left')
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)


def plot_microstructure(result, max_lag=30, save_path=None):
    '''Stylized facts of the simulated market (spread, depth, order flow, impact), one panel each'''
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    snaps = pd.concat(result.snapshots, ignore_index=True)

    ax = axes[0, 0]
    ax.hist(snaps['spread'].dropna(), bins=range(0, 30), color='#2a78d6', edgecolor='#fcfcfb', linewidth=1)
    _style(ax, 'Bid-ask spread distribution', 'spread (ticks)', 'number of snapshots')

    ax = axes[0, 1]
    depth = average_depth_profile(result.snapshots)
    width = 0.38
    ax.bar(depth['level'] - width / 2, depth['bid'], width, color='#2a78d6', label='bid side', edgecolor='#fcfcfb')
    ax.bar(depth['level'] + width / 2, depth['ask'], width, color='#eb6834', label='ask side', edgecolor='#fcfcfb')
    ax.legend(frameon=False, fontsize=9)
    _style(ax, 'Average volume at the k-th best price level', 'level k (1 = best quote)', 'volume')

    ax = axes[1, 0]
    acf = order_sign_acf([o['sign'].values for o in result.orders], max_lag)
    ax.bar(acf['lag'], acf['acf'], color='#2a78d6', width=0.7)
    ax.plot(acf['lag'], 2 * acf['se'], color=MUTED, linewidth=1.2, linestyle='--', label='+/- 2 standard errors (independent signs)')
    ax.plot(acf['lag'], -2 * acf['se'], color=MUTED, linewidth=1.2, linestyle='--')
    ax.legend(frameon=False, fontsize=9)
    _style(ax, 'Autocorrelation of the order signs C(l)', 'lag l (in aggressive orders)', 'C(l)')

    ax = axes[1, 1]
    iv = impact_vs_volume(result.orders)
    ax.plot(iv['v'], iv['lag1'], marker='o', markersize=7, color='#2a78d6', linewidth=2, label='lag-1 impact R(v, 1)')
    ax.plot(iv['v'], iv['immediate'].where(iv['immediate'] > 0), marker='s', markersize=7, color='#eb6834', linewidth=2, label='immediate impact')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(frameon=False, fontsize=9)
    _style(ax, 'Impact of an aggressive order vs its size', 'order volume / volume at the best opposite quote', 'signed mid-price change')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches='tight')
    return fig


def plot_strategies(metrics, save_path=None):
    '''Mark-to-market PnL of each strategy and breakdown of its execution costs'''
    import matplotlib.pyplot as plt
    names = list(COLORS)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    data = [metrics.loc[metrics['strategy'] == n, 'mtm_pnl'].values for n in names]
    try:
        bp = ax.boxplot(data, orientation='horizontal', patch_artist=True, widths=0.6, medianprops=dict(color=INK, linewidth=1.5), showfliers=True)
    except TypeError: # matplotlib < 3.10 does not know `orientation`
        bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.6, medianprops=dict(color=INK, linewidth=1.5), showfliers=True)
    for patch, n in zip(bp['boxes'], names):
        patch.set_facecolor(COLORS[n])
        patch.set_edgecolor('#fcfcfb')
    ax.set_yticks(range(1, len(names) + 1))
    ax.set_yticklabels(names)
    ax.axvline(0, color=MUTED, linewidth=1)
    _style(ax, 'Mark-to-market PnL per simulation', 'PnL (money units)')
    ax = axes[1]
    g = metrics.groupby('strategy')[['half_spread_cost', 'execution_cost']].mean().reindex(names)
    impact = g['execution_cost'] - g['half_spread_cost']
    pos = np.arange(len(names))
    ax.barh(pos, g['half_spread_cost'], color='#2a78d6', edgecolor='#fcfcfb', linewidth=2, label='half-spread cost')
    ax.barh(pos, impact, left=g['half_spread_cost'], color='#eb6834', edgecolor='#fcfcfb', linewidth=2, label='price impact')
    ax.set_yticks(pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=9)
    _style(ax, 'Average execution cost per simulation', 'cost vs the mid-price before each order (money units)')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches='tight')
    return fig


def plot_square_root(save_path=None):
    '''Impact of a market order on two static books: a flat depth profile (linear impact) and a linearly growing one (square-root impact)'''
    import matplotlib.pyplot as plt
    q = [10, 20, 40, 80, 160, 320, 640, 1280, 2560]
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for profile, color, label in (('flat', '#2a78d6', 'flat depth profile'), ('linear', '#eb6834', 'depth growing linearly with the distance')):
        df = static_book_impact(profile, q)
        delta = fit_exponent(df['quantity'], df['impact'])
        ax.plot(df['quantity'], df['impact'], marker='o', markersize=7, linewidth=2, color=color, label='%s, fitted exponent %.2f' % (label, delta))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(frameon=False, fontsize=9)
    _style(ax, 'Impact I(Q) of a market order on a static book', 'order size Q', 'price move of the best ask')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches='tight')
    return fig


def response_function(orders_by_simulation, max_lag=50):
    '''
    Response function R(l) = < e_t (m_(t+l) - m_t) > in market order time (chapter 11.3.1, Figure 11.2): how much the mid-price
    has moved in the direction of an aggressive order l aggressive orders later. Returns lag, R, standard error.
    '''
    rows = []
    for lag in range(1, max_lag + 1):
        values = []
        for o in orders_by_simulation:
            m, e = o['mid_before'].values, o['sign'].values
            if len(m) > lag:
                values.append(e[:-lag]*(m[lag:] - m[:-lag]))
        v = np.concatenate(values) if values else np.array([])
        v = v[~np.isnan(v)]
        rows.append((lag, v.mean() if len(v) else np.nan, v.std(ddof=1)/np.sqrt(len(v)) if len(v) > 1 else np.nan))
    return pd.DataFrame(rows, columns=['lag', 'R', 'se'])


def standardized_changes(mid_series, horizon):
    '''Mid-price changes over `horizon` events, each simulation standardised by its own standard deviation'''
    out = []
    for m in mid_series:
        m = np.asarray(m, dtype=float)
        m = m[~np.isnan(m)]
        d = m[horizon::horizon] - m[:-horizon:horizon]
        if len(d) > 2 and d.std() > 0:
            out.append((d - d.mean())/d.std())
    return np.concatenate(out) if out else np.array([])


def mid_at(snapshots, times):
    '''Mid-price of the last snapshot at or before each time'''
    t, m = snapshots['time'].values, snapshots['mid'].values
    k = np.searchsorted(t, np.asarray(times), side='right') - 1
    return m[np.clip(k, 0, len(m) - 1)]


def fill_sides(fills):
    '''One row per (fill, counterparty): the agent, his kind, the sign of HIS trade (+1 bought) and whether he was the aggressor'''
    a = pd.DataFrame({'time': fills['time'], 'price': fills['price'], 'quantity': fills['quantity'], 'agent': fills['aggressor'],
                      'kind': fills['aggressor_kind'], 'other': fills['passive_kind'], 's': fills['sign'], 'aggressive': True})
    p = pd.DataFrame({'time': fills['time'], 'price': fills['price'], 'quantity': fills['quantity'], 'agent': fills['passive'],
                      'kind': fills['passive_kind'], 'other': fills['aggressor_kind'], 's': -fills['sign'], 'aggressive': False})
    return pd.concat([a, p], ignore_index=True)


def markouts(fills, snapshots, horizons, agent_kind='market maker', aggressive=False):
    '''
    Markout of the fills of one kind of agent: s * (m(t + h) - price) per share, in ticks, where s = +1 when he bought.
    At h = 0 it is the spread he earned (or paid), its decay with h is the adverse selection. Split by the kind of the counterparty.
    '''
    sides = fill_sides(fills)
    sides = sides[(sides['kind'] == agent_kind) & (sides['aggressive'] == aggressive)]
    mid_before = mid_at(snapshots, sides['time'].values - 1)
    rows = []
    for h in horizons:
        m = mid_before if h == 0 else mid_at(snapshots, sides['time'].values + h)
        value = sides['s'].values*(m - sides['price'].values)
        df = pd.DataFrame({'other': sides['other'].values, 'v': value*sides['quantity'].values, 'q': sides['quantity'].values})
        g = df.groupby('other')[['v', 'q']].sum()
        for other, r in g.iterrows():
            rows.append((h, other, r['v']/r['q'], r['q']))
    return pd.DataFrame(rows, columns=['horizon', 'counterparty', 'markout', 'volume'])


def flow_matrix(fills):
    '''Share of the traded volume between each kind of aggressor (rows) and each kind of passive counterparty (columns)'''
    m = fills.pivot_table(index='aggressor_kind', columns='passive_kind', values='quantity', aggfunc='sum', fill_value=0)
    return m/m.values.sum()


def pnl_decomposition(res, horizon=100):
    '''
    Mark-to-market PnL of every agent (in ticks), split exactly into three parts, summed over his fills (s = +1 when he bought):
      spread    = s q (m_before - price)          what he earned (positive) or paid (negative) against the mid-price
      adverse   = s q (m(t + horizon) - m_before) how the price moved against (negative) or with (positive) him just after the fill
      inventory = s q (m_end - m(t + horizon)) + initial shares * (m_end - m_start)   the rest: holding the position
    spread + adverse + inventory = final cash + final shares * m_end - (initial cash + initial shares * m_start).
    '''
    snaps = res.snapshots
    sides = fill_sides(res.fills)
    t = sides['time'].values
    m_before = mid_at(snaps, t - 1)
    m_h = mid_at(snaps, t + horizon)
    m_end, m_start = snaps['mid'].iloc[-1], snaps['mid'].iloc[0]
    q, s, p = sides['quantity'].values, sides['s'].values, sides['price'].values
    sides = sides.assign(spread=s*q*(m_before - p), adverse=s*q*(m_h - m_before), inventory=s*q*(m_end - m_h))
    g = sides.groupby('agent').agg(kind=('kind', 'first'), volume=('quantity', 'sum'), spread=('spread', 'sum'),
                                   adverse=('adverse', 'sum'), inventory=('inventory', 'sum'))
    for agent_id, agent in res.exchange.agents.items():
        if agent_id in g.index and res.exchange.kinds[agent_id] in res.strategy_start:
            g.loc[agent_id, 'inventory'] += res.strategy_start[res.exchange.kinds[agent_id]][1]*(m_end - m_start)
    g['total'] = g['spread'] + g['adverse'] + g['inventory']
    return g.reset_index()

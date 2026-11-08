# The studies run on the market ecology, and their figures.
#   ablation()            which ingredient of the market produces which stylized fact (chapters 2, 4, 10, 11)
#   ecology_study()       who trades with whom, who earns what, and what the six directional strategies earn in a realistic market
#   market_maker_sweep()  the spread a market maker should quote, with and without informed traders (chapters 16 and 17)
#   metaorder_experiment() the impact of a metaorder measured with common random numbers: is it a square-root? (chapter 12)
import numpy as np
import pandas as pd

from ecology import EcologyConfig, run_ecology, Exchange, Agent
from analysis import (order_sign_acf, response_function, standardized_changes, flow_matrix, markouts, pnl_decomposition,
                      fit_exponent, fit_power_law, COLORS, INK, MUTED, GRID, _style)
from simulation import SimulationConfig, run_simulation, STRATEGIES

PALETTE = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']

DESIGNS = {
    'Zero intelligence': dict(use_market_maker=False, use_informed=False, use_metaorders=False),
    'Zero intelligence + metaorders': dict(use_market_maker=False, use_informed=False),
    'Full ecology': dict(),
}
BASELINE = 'Baseline market (uniform flow)'
KIND_LABELS = {'noise taker': 'noise takers', 'informed': 'informed', 'metaorder': 'metaorders', 'strategies': 'directional strategies',
               'market maker': 'market maker', 'noise maker': 'noise makers'}


def ablation(seeds=(1, 2, 3, 4, 5, 6), nb_events=20000, baseline=None):
    '''
    Runs the market with more and more ingredients and measures the same statistics on each one. Ticks are the unit of price
    everywhere (1 money unit in the baseline market, 0.01 in the ecology).
    '''
    out = {}
    if baseline is None:
        baseline = run_simulation(SimulationConfig(seed=42, nb_trader_per_strat=100))
    out[BASELINE] = {'orders': baseline.orders, 'snapshots': baseline.snapshots,
                          'mids': [s['mid'].values for s in baseline.snapshots]}
    for name, kw in DESIGNS.items():
        runs = [run_ecology(EcologyConfig(seed=seed, nb_events=nb_events, **kw)) for seed in seeds]
        out[name] = {'orders': [r.orders for r in runs], 'snapshots': [r.snapshots for r in runs],
                     'mids': [r.mids[r.cfg.warmup:] for r in runs]}
    return out


def stylized_facts_table(ablation_result, horizon=20):
    rows = []
    for name, d in ablation_result.items():
        acf = order_sign_acf([o['sign'].values for o in d['orders']], 100)
        gamma = fit_power_law(acf['lag'], acf['acf'], acf['se'])
        resp = response_function(d['orders'], 20)
        z = standardized_changes(d['mids'], horizon)
        spread = pd.concat(d['snapshots'])['spread'].dropna()
        rows.append({'market': name, 'C(1)': acf['acf'].iloc[0], 'C(10)': acf['acf'].iloc[9],
                     'C(50)': acf['acf'].iloc[49] if len(acf) >= 50 else np.nan,
                     'gamma': np.nan if gamma is None else gamma, 'R(1) ticks': resp['R'].iloc[0], 'R(20)/R(1)': resp['R'].iloc[19]/resp['R'].iloc[0],
                     'mean spread (ticks)': spread.mean(), 'P(spread = 1 tick)': (spread <= 1).mean(),
                     'excess kurtosis (%d events)' % horizon: pd.Series(z).kurt(),
                     'aggressive orders': sum(len(o) for o in d['orders'])})
    return pd.DataFrame(rows).set_index('market')


def binned_acf(acf, edges=(1, 2, 3, 5, 8, 13, 21, 34, 55, 101)):
    '''Average of C(l) over logarithmic bins of lags (the standard error shrinks with the number of lags in the bin)'''
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = acf[(acf['lag'] >= lo) & (acf['lag'] < hi)]
        if len(b):
            rows.append((np.exp(np.log(b['lag']).mean()), b['acf'].mean(), b['se'].mean()/np.sqrt(len(b))))
    return pd.DataFrame(rows, columns=['lag', 'acf', 'se'])


def plot_stylized_facts(ablation_result, horizon=20, save_path=None):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for (name, d), color in zip(ablation_result.items(), PALETTE):
        acf = order_sign_acf([o['sign'].values for o in d['orders']], 100)
        b = binned_acf(acf)
        ok = b['acf'] > 2*b['se']
        if ok.any():
            axes[0, 0].plot(b['lag'][ok], b['acf'][ok], marker='o', markersize=7, linewidth=2, color=color, label=name)
        else:
            axes[0, 0].plot([], [], marker='o', color=color, label=name + ': no significant memory')
        resp = response_function(d['orders'], 30)
        axes[0, 1].plot(resp['lag'], resp['R'], linewidth=2, color=color, label=name)
        axes[0, 1].fill_between(resp['lag'], resp['R'] - 2*resp['se'], resp['R'] + 2*resp['se'], color=color, alpha=0.15, linewidth=0)
        spread = np.round(pd.concat(d['snapshots'])['spread'].dropna()).astype(int)
        p = spread.value_counts(normalize=True).sort_index()
        p = p[p.index <= 15]
        axes[1, 0].plot(p.index, p.values, marker='o', markersize=7, linewidth=2, color=color, label=name)
        z = np.sort(np.abs(standardized_changes(d['mids'], horizon)))
        if len(z):
            ccdf = 1 - np.arange(len(z))/len(z)
            axes[1, 1].plot(z, ccdf, linewidth=2, color=color, label=name)
    x = np.array([1, 100])
    axes[0, 0].plot(x, 0.08*x**-0.5, color=MUTED, linestyle='--', linewidth=1.2, label='slope -0.5 (Lillo-Mike-Farmer, alpha = 1.5)')
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_yscale('log')
    _style(axes[0, 0], 'Order signs: only metaorder splitting creates long memory', 'lag l (aggressive orders, log bins)', 'C(l), values above 2 standard errors')
    axes[0, 1].axhline(0, color=MUTED, linewidth=1)
    _style(axes[0, 1], 'Response function R(l): impact grows when the flow is persistent', 'lag l (aggressive orders)', 'R(l) (ticks)')
    axes[1, 0].set_yscale('log')
    _style(axes[1, 0], 'Spread distribution: the baseline market is far too wide', 'spread (ticks)', 'probability')
    from math import erfc, sqrt
    g = np.linspace(0, 8, 100)
    axes[1, 1].plot(g, [erfc(v/sqrt(2)) for v in g], color=MUTED, linestyle='--', linewidth=1.2, label='Gaussian')
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_xlim(0, 8)
    axes[1, 1].set_ylim(1e-4, 1.1)
    _style(axes[1, 1], 'Tails of the mid-price changes over %d events' % horizon, '|change| / standard deviation', 'P(|change| > x)')
    for ax in axes.flat:
        ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches='tight')
    return fig


def agent_path(res, kind):
    '''Inventory and mark-to-market PnL (ticks x shares) of the agents of one kind, after every snapshot'''
    f = res.fills
    rows = []
    for col, s in (('aggressor_kind', 1), ('passive_kind', -1)):
        g = f[f[col] == kind]
        rows.append(pd.DataFrame({'time': g['time'], 'q': s*g['sign']*g['quantity'], 'cash': -s*g['sign']*g['quantity']*g['price']}))
    g = pd.concat(rows).groupby('time').sum().sort_index()
    snaps = res.snapshots[['time', 'mid']].copy()
    idx = np.searchsorted(g.index.values, snaps['time'].values, side='right') - 1
    inv = np.where(idx >= 0, np.cumsum(g['q'].values)[np.clip(idx, 0, None)], 0.0) if len(g) else np.zeros(len(snaps))
    cash = np.where(idx >= 0, np.cumsum(g['cash'].values)[np.clip(idx, 0, None)], 0.0) if len(g) else np.zeros(len(snaps))
    snaps['inventory'] = inv
    snaps['pnl'] = cash + inv*snaps['mid'].values
    return snaps


def plot_ecology_overview(res, save_path=None):
    '''One run of the full ecology: price discovery, inventories, liquidity'''
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, gridspec_kw={'height_ratios': [2.2, 1.3, 1.3]})
    s, f = res.snapshots, res.fundamental
    t0 = s['time'].iloc[0]
    ax = axes[0]
    ax.plot(s['time'] - t0, s['mid'], color='#2a78d6', linewidth=1.2, label='mid-price')
    ax.plot(f['time'] - t0, f['value'], color='#eb6834', linewidth=1.6, label='fundamental value (seen by the informed trader only)')
    jumps = np.where(np.abs(np.diff(f['value'].values)) > 5)[0]
    for k in jumps:
        ax.axvline(f['time'].iloc[k + 1] - t0, color=MUTED, linewidth=0.8, linestyle=':')
    ax.legend(frameon=False, fontsize=9, loc='best')
    _style(ax, 'Price discovery: the informed trader pulls the mid-price towards the fundamental value (dotted lines: news)', '', 'price (ticks)')
    ax = axes[1]
    if res.mm is not None:
        inv = pd.DataFrame(res.mm.inventory_path, columns=['time', 'inventory'])
        ax.step(inv['time'] - t0, inv['inventory'], where='post', color='#1baf7a', linewidth=1.4, label='market maker inventory')
        ax.axhline(res.mm.max_inventory, color=MUTED, linewidth=0.8, linestyle='--')
        ax.axhline(-res.mm.max_inventory, color=MUTED, linewidth=0.8, linestyle='--')
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.legend(frameon=False, fontsize=9, loc='upper left')
    _style(ax, 'The market maker keeps his inventory close to zero by skewing his quotes (dashed: his limits)', '', 'shares')
    ax = axes[2]
    p = agent_path(res, 'market maker')
    ax.plot(p['time'] - t0, p['pnl'], color='#1baf7a', linewidth=1.4, label='market maker')
    p = agent_path(res, 'informed')
    ax.plot(p['time'] - t0, p['pnl'], color='#e34948', linewidth=1.4, label='informed trader')
    for k in jumps:
        ax.axvline(f['time'].iloc[k + 1] - t0, color=MUTED, linewidth=0.8, linestyle=':')
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.legend(frameon=False, fontsize=9, loc='upper left')
    _style(ax, 'Mark-to-market PnL: the market maker earns steadily on a small inventory, the informed trader carries large and volatile positions', 'book time (events)', 'PnL (ticks x shares)')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches='tight')
    return fig


def ecology_study(seeds=tuple(range(1, 21)), nb_events=20000, horizon=100):
    '''Full ecology with the six directional strategies, over several seeds'''
    runs, tables = [], []
    for seed in seeds:
        r = run_ecology(EcologyConfig(seed=seed, nb_events=nb_events, use_strategies=True))
        runs.append(r)
        d = pnl_decomposition(r, horizon)
        d['seed'] = seed
        m_start = r.snapshots['mid'].iloc[0]
        d['start_wealth'] = np.nan
        for name, (cash, shares) in r.strategy_start.items():
            d.loc[d['kind'] == name, 'start_wealth'] = cash/r.cfg.tick + shares*m_start # in ticks, as the PnL
        d['mid_start'] = m_start
        tables.append(d)
    return runs, pd.concat(tables, ignore_index=True)


def group_kind(kind):
    return 'strategies' if kind in STRATEGIES else kind


def plot_flows(runs, horizons=(0, 1, 2, 5, 10, 20, 50, 100, 200, 400), save_path=None):
    import matplotlib.pyplot as plt
    fills = pd.concat([r.fills for r in runs], ignore_index=True)
    fills = fills.assign(aggressor_kind=fills['aggressor_kind'].map(group_kind), passive_kind=fills['passive_kind'].map(group_kind))
    fm = flow_matrix(fills)
    rows = [k for k in ['noise taker', 'informed', 'metaorder', 'strategies', 'market maker', 'noise maker'] if k in fm.index]
    cols = [k for k in ['noise maker', 'market maker', 'strategies'] if k in fm.columns]
    fm = fm.reindex(index=rows, columns=cols).fillna(0)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={'width_ratios': [1, 1.4]})
    ax = axes[0]
    ax.imshow(fm.values, cmap='Blues', aspect='auto', vmin=0, vmax=fm.values.max())
    for i in range(fm.shape[0]):
        for j in range(fm.shape[1]):
            v = fm.values[i, j]
            ax.text(j, i, '%.1f%%' % (100*v), ha='center', va='center', fontsize=10, color='#fcfcfb' if v > 0.6*fm.values.max() else INK)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([KIND_LABELS[c] for c in cols])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([KIND_LABELS[r] for r in rows])
    ax.set_title('Who takes liquidity from whom (share of the volume)', fontsize=11, color=INK, loc='left')
    ax.set_xlabel('passive side (liquidity provider)', color=MUTED, fontsize=9)
    ax.set_ylabel('aggressor (liquidity taker)', color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax = axes[1]
    snaps = None
    curves = []
    for r in runs:
        f = r.fills.assign(aggressor_kind=r.fills['aggressor_kind'].map(group_kind), passive_kind=r.fills['passive_kind'].map(group_kind))
        mk = markouts(f, r.snapshots, horizons, 'market maker', aggressive=False)
        curves.append(mk)
    mk = pd.concat(curves).assign(v=lambda d: d['markout']*d['volume']).groupby(['counterparty', 'horizon'])[['v', 'volume']].sum()
    mk = (mk['v']/mk['volume']).rename('markout').reset_index()
    for other, color in zip(['noise taker', 'strategies', 'metaorder', 'informed'], ['#2a78d6', '#eda100', '#eb6834', '#e34948']):
        c = mk[mk['counterparty'] == other]
        if len(c):
            ax.plot(c['horizon'] + 0.5, c['markout'], marker='o', markersize=7, linewidth=2, color=color, label='against ' + KIND_LABELS[other])
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xscale('log')
    ax.legend(frameon=False, fontsize=9)
    _style(ax, 'Market maker: what a passive fill is worth h events later', 'horizon h (events, log scale; leftmost point = at the fill)', 'markout per share (ticks)')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches='tight')
    return fig, fm, mk


def kind_summary(table):
    '''PnL decomposition per kind of agent, in ticks per share traded, mean over the seeds with a standard error'''
    t = table.copy()
    per_seed = t.groupby(['kind', 'seed'])[['volume', 'spread', 'adverse', 'inventory', 'total']].sum()
    out = per_seed.groupby('kind').sum()
    for c in ['spread', 'adverse', 'inventory', 'total']:
        out[c] = out[c]/out['volume']
    ratio = per_seed['total']/per_seed['volume']
    out['se_total'] = ratio.groupby('kind').std()/np.sqrt(ratio.groupby('kind').size())
    out['volume per run'] = per_seed['volume'].groupby('kind').mean()
    return out.drop(columns='volume')


def plot_pnl_decomposition(table, save_path=None):
    import matplotlib.pyplot as plt
    s = kind_summary(table)
    groups = [('The market: who earns, who pays', [k for k in ['market maker', 'informed', 'metaorder', 'noise taker', 'noise maker'] if k in s.index]),
              ('The six directional strategies', [k for k in STRATEGIES if k in s.index])]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    for ax, (title, order) in zip(axes, groups):
        t = s.loc[order]
        pos = np.arange(len(order))
        h = 0.26
        for k, (c, color, label) in enumerate([('spread', '#2a78d6', 'spread earned (+) or paid (-) against the mid'),
                                               ('adverse', '#eb6834', 'price move in the 100 events after the fill'),
                                               ('inventory', '#1baf7a', 'holding the position until the end')]):
            ax.barh(pos + (k - 1)*h, t[c], h, color=color, edgecolor='#fcfcfb', linewidth=1, label=label)
        ax.errorbar(t['total'], pos, xerr=2*t['se_total'], fmt='D', color=INK, markersize=6, capsize=3, label='total (+/- 2 standard errors)')
        ax.axvline(0, color=MUTED, linewidth=1)
        ax.set_yticks(pos)
        ax.set_yticklabels(['%s\n%.0f shares per run' % (KIND_LABELS.get(k, k), t.loc[k, 'volume per run']) for k in order], fontsize=8.5)
        ax.invert_yaxis()
        _style(ax, title, 'ticks per share traded')
    axes[0].legend(frameon=False, fontsize=8.5, loc='lower right')
    fig.suptitle('Where the PnL comes from, per share traded (full ecology, %d runs)' % table['seed'].nunique(), x=0.01, ha='left', fontsize=12, color=INK)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches='tight')
    return fig, s


def market_comparison(baseline, table):
    '''The six strategies in the first-version market and in the ecology, in units that do not depend on the tick: basis points'''
    rows = []
    fv = baseline.metrics
    mid0 = np.array([s['mid'].iloc[0] for s in baseline.snapshots])
    for name in STRATEGIES:
        m = fv[fv['strategy'] == name]
        start = np.array([baseline.start_equity[i][name] for i in m['simulation']])
        cost = m['execution_cost'].sum()/max(m['filled_volume'].sum(), 1e-12)/mid0.mean()*1e4
        r = 1e4*m['mtm_pnl'].values/start
        rows.append({'strategy': name, 'market': BASELINE, 'cost (bps of price)': cost if m['filled_volume'].sum() > 0 else np.nan,
                     'PnL (bps of wealth)': r.mean(), 'se': r.std(ddof=1)/np.sqrt(len(r))})
        e = table[table['kind'] == name]
        cost = -e['spread'].sum()/max(e['volume'].sum(), 1e-12)/e['mid_start'].mean()*1e4
        r = 1e4*(e['total']/e['start_wealth']).values
        rows.append({'strategy': name, 'market': 'Full ecology', 'cost (bps of price)': cost if e['volume'].sum() > 0 else np.nan,
                     'PnL (bps of wealth)': r.mean(), 'se': r.std(ddof=1)/np.sqrt(len(r))})
    return pd.DataFrame(rows)


def plot_market_comparison(comparison, save_path=None):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    pos = np.arange(len(STRATEGIES))
    for k, (market, color) in enumerate([(BASELINE, '#eb6834'), ('Full ecology', '#2a78d6')]):
        c = comparison[comparison['market'] == market].set_index('strategy').loc[STRATEGIES]
        axes[0].scatter(c['cost (bps of price)'], pos + (k - 0.5)*0.25, s=60, color=color, label=market, zorder=3)
        axes[1].errorbar(c['PnL (bps of wealth)'], pos + (k - 0.5)*0.25, xerr=2*c['se'], fmt='o', markersize=7, color=color, capsize=3, label=market)
    axes[0].set_xscale('log')
    for ax in axes:
        ax.set_yticks(pos)
        ax.set_yticklabels(STRATEGIES)
        ax.invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, fontsize=10)
    axes[1].axvline(0, color=MUTED, linewidth=1)
    axes[1].set_xscale('symlog', linthresh=1)
    _style(axes[0], 'Execution cost per share traded', 'basis points of the price (log scale)')
    _style(axes[1], 'Mark-to-market PnL (+/- 2 standard errors)', 'basis points of initial wealth (symmetric log scale)')
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches='tight')
    return fig


def market_maker_sweep(half_spreads=(0.5, 1, 2, 3, 5), informed_rates=(0.0, 3.0, 9.0), seeds=(1, 2, 3, 4, 5), nb_events=12000):
    '''PnL of the market maker for several half-spreads and several intensities of informed trading'''
    rows = []
    for ir in informed_rates:
        for hs in half_spreads:
            for seed in seeds:
                r = run_ecology(EcologyConfig(seed=seed, nb_events=nb_events, mm_half_spread=hs, informed_rate=ir, use_informed=ir > 0))
                d = pnl_decomposition(r, 100)
                mm = d[d['kind'] == 'market maker']
                if len(mm) == 0:
                    mm = pd.DataFrame([{'volume': 0, 'spread': 0, 'adverse': 0, 'inventory': 0, 'total': 0}])
                mm = mm.iloc[0]
                rows.append({'informed_rate': ir, 'half_spread': hs, 'seed': seed, 'volume': mm['volume'], 'spread': mm['spread'],
                             'adverse': mm['adverse'], 'inventory': mm['inventory'], 'total': mm['total'],
                             'mean_spread': r.snapshots['spread'].mean()})
    return pd.DataFrame(rows)


def plot_market_maker_sweep(sweep, save_path=None):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    labels = {0.0: 'no informed trader', 3.0: 'informed trader (baseline)', 9.0: 'three times more informed flow'}
    for (ir, g), color in zip(sweep.groupby('informed_rate'), ['#2a78d6', '#eda100', '#e34948']):
        a = g.groupby('half_spread')['total'].agg(['mean', 'std', 'size'])
        se = a['std']/np.sqrt(a['size'])
        label = labels.get(ir, 'informed rate %.1f' % ir)
        axes[0].errorbar(a.index, a['mean'], yerr=2*se, marker='o', markersize=7, linewidth=2, capsize=3, color=color, label=label)
        best = a['mean'].idxmax()
        axes[0].scatter([best], [a['mean'].max()], s=180, facecolors='none', edgecolors=color, linewidths=2, zorder=4)
        b = g.groupby('half_spread')[['spread', 'adverse', 'volume']].sum()
        axes[1].plot(b.index, -b['adverse']/b['volume'], marker='o', markersize=7, linewidth=2, color=color, label=label)
        axes[1].plot(b.index, b['spread']/b['volume'], marker='s', markersize=6, linewidth=1.4, linestyle='--', color=color)
    axes[0].axhline(0, color=MUTED, linewidth=1)
    axes[0].legend(frameon=False, fontsize=9)
    _style(axes[0], 'Market maker PnL per run (circled: best half-spread)', 'quoted half-spread (ticks)', 'mark-to-market PnL (ticks x shares)')
    _style(axes[1], 'Per share: spread earned (dashed) and adverse selection cost (solid)', 'quoted half-spread (ticks)', 'ticks per share')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches='tight')
    return fig


class Trader_test_metaorder(Agent):
  # The metaorder whose impact we measure: buys Q shares with child market orders of `child` shares, one every `pace` events
  def __init__(self, Q, child):
    self.init_account(10**9, 0, 0)
    self.Q, self.child, self.done = Q, child, 0


def metaorder_experiment(sizes=(25, 50, 100, 200, 400, 800, 1600), seeds=tuple(range(1, 25)), child=5, pace=4, warmup_events=2000, **kw):
    '''
    Impact of a buy metaorder of size Q executed with child market orders of `child` shares every `pace` events. For every
    seed the market is run twice with the same random numbers, with and without the metaorder (common random numbers): the
    difference of the two mid-price paths is the impact of the metaorder alone, most of the market noise cancels out.
    The impact is followed during the execution (duration T) and as long again after it.
    '''
    t_max = 2*int(np.ceil(max(sizes)/child))*pace
    rows, peaks = [], []
    for seed in seeds:
        cfg = EcologyConfig(seed=seed, nb_events=warmup_events + t_max + 1, record=False, **kw)
        start = cfg.warmup + warmup_events
        base = run_ecology(cfg).mids
        for Q in sizes:
            n_child = int(np.ceil(Q/child))
            T = n_child*pace
            agent = Trader_test_metaorder(Q, child)
            state = {'registered': False}

            def hook(i, exchange, agent=agent, n_child=n_child, state=state):
                if i >= start and (i - start) % pace == 0 and agent.done < n_child:
                    if not state['registered']:
                        exchange.register(agent, 'test metaorder')
                        state['registered'] = True
                    exchange.send(agent, {'type': 'market', 'side': 'bid', 'quantity': child, 'trade_id': agent.trade_id})
                    agent.done += 1
            treated = run_ecology(cfg, hook=hook).mids
            m0 = base[start - 1]
            impact = treated[start - 1:start + 2*T] - base[start - 1:start + 2*T]
            cost = float(agent.K_initial - agent.K)/float(agent.nb_shares)/cfg.tick - m0 if agent.nb_shares > 0 else np.nan
            for t, v in enumerate(impact):
                rows.append((seed, Q, T, t, v))
            peaks.append({'seed': seed, 'Q': Q, 'T': T, 'peak': impact[T], 'end': impact[2*T], 'shortfall': cost,
                          'background_move': base[start - 1 + T] - m0})
    return pd.DataFrame(rows, columns=['seed', 'Q', 'T', 't', 'impact']), pd.DataFrame(peaks)


# The three markets of the impact study. Only the liquidity changes from one to the other.
IMPACT_MARKETS = {
    'Santa Fe liquidity only': dict(use_market_maker=False, use_informed=False, use_metaorders=False),
    'Full ecology': dict(),
    'Latent liquidity': dict(use_market_maker=False, use_informed=False, use_metaorders=False, use_latent=True, window=3, limit_rate=1.5,
                             latent_rate=40, latent_window=100, latent_tau=1e9, latent_cancel_rate=0.01),
}


def impact_study(sizes=(25, 50, 100, 200, 400, 800, 1600), seeds=tuple(range(1, 17)), markets=None):
    out = {}
    for name, kw in (markets or IMPACT_MARKETS).items():
        out[name] = metaorder_experiment(sizes=sizes, seeds=seeds, **kw)
    return out


def impact_table(study):
    rows = []
    for name, (paths, peaks) in study.items():
        a = peaks.groupby('Q').agg(peak=('peak', 'mean'), end=('end', 'mean'), shortfall=('shortfall', 'mean'))
        small, large = a[a.index <= 100], a[a.index >= 200]
        rows.append({'market': name, 'exponent (all Q)': fit_exponent(a.index, a['peak']),
                     'exponent (Q <= 100)': fit_exponent(small.index, small['peak'], 3),
                     'exponent (Q >= 200)': fit_exponent(large.index, large['peak'], 3),
                     'impact kept after a further T (Q >= 200)': (large['end']/large['peak']).mean(),
                     'peak impact Q=1600 (ticks)': a['peak'].iloc[-1]})
    return pd.DataFrame(rows).set_index('market')


def plot_metaorder(study, save_path=None):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    colors = {'Santa Fe liquidity only': '#eb6834', 'Full ecology': '#2a78d6', 'Latent liquidity': '#1baf7a'}
    for name, (paths, peaks) in study.items():
        color = colors.get(name, MUTED)
        a = peaks.groupby('Q').agg(peak=('peak', 'mean'), se=('peak', lambda x: x.std(ddof=1)/np.sqrt(len(x))))
        small, large = a[a.index <= 100], a[a.index >= 200]
        axes[0].errorbar(a.index, a['peak'], yerr=2*a['se'], fmt='o-', markersize=7, linewidth=2, capsize=3, color=color,
                         label='%s: exponent %.2f for Q <= 100, %.2f for Q >= 200' % (name, fit_exponent(small.index, small['peak'], 3),
                                                                               fit_exponent(large.index, large['peak'], 3)))
        Q = 800 if 800 in paths['Q'].values else paths['Q'].max()
        p = paths[paths['Q'] == Q].groupby('t')['impact'].mean()
        T = paths.loc[paths['Q'] == Q, 'T'].iloc[0]
        axes[1].plot(p.index/T, p.values/max(p.loc[T], 1e-9), color=color, linewidth=2, label=name)
    q = np.array([25, 1600], dtype=float)
    axes[0].plot(q, 2.0*(q/q[0])**0.5, color=MUTED, linestyle='--', linewidth=1.2, label='slope 0.5 (square-root law)')
    axes[0].plot(q, 1.2*(q/q[0]), color=MUTED, linestyle=':', linewidth=1.2, label='slope 1 (linear, Kyle)')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].legend(frameon=False, fontsize=8.5)
    _style(axes[0], 'Peak impact of a buy metaorder (+/- 2 standard errors)', 'metaorder size Q (shares)', 'impact at the end of the execution (ticks)')
    axes[1].axvline(1, color=MUTED, linewidth=1, linestyle='--')
    axes[1].axhline(0, color=MUTED, linewidth=0.8)
    axes[1].legend(frameon=False, fontsize=9)
    _style(axes[1], 'Impact path of a metaorder of 800 shares, normalised by its peak', 'time / execution duration T (dashed: end of execution)', 'impact / peak impact')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches='tight')
    return fig


def news_event_study(runs, before=300, after=1500, threshold=5):
    '''
    Average market around every news (jump of the fundamental value) in direction of the news:
    mispricing = sign * (V - mid) (how much information is not yet in the price), spread, and the cumulative signed volume
    of each kind of aggressor (who brings the information into the price).
    '''
    rows = []
    for r in runs:
        f, s = r.fundamental, r.snapshots
        jumps = np.where(np.abs(np.diff(f['value'].values)) > threshold)[0] + 1
        t_s, mid, spread = s['time'].values, s['mid'].values, s['spread'].values
        fills = r.fills.assign(kind=r.fills['aggressor_kind'].map(group_kind))
        mm = agent_path(r, 'market maker') if r.mm is not None else None
        for k in jumps:
            t0 = f['time'].iloc[k]
            sign = np.sign(f['value'].iloc[k] - f['value'].iloc[k - 1])
            if t0 - before < t_s[0] or t0 + after > t_s[-1]:
                continue
            grid = np.arange(-before, after + 1, 10)
            idx = np.clip(np.searchsorted(t_s, t0 + grid, side='right') - 1, 0, len(t_s) - 1)
            v = np.interp(t0 + grid, f['time'].values, f['value'].values)
            w = fills[(fills['time'] >= t0) & (fills['time'] <= t0 + after)]
            flow = {}
            for kind, g in w.groupby('kind'):
                cum = np.cumsum((g['sign']*g['quantity']).values)*sign
                flow[kind] = np.interp(t0 + grid, g['time'].values, cum, left=0, right=cum[-1] if len(cum) else 0)
            for j, dt in enumerate(grid):
                row = {'dt': dt, 'mispricing': sign*(v[j] - mid[idx[j]]), 'spread': spread[idx[j]],
                       'mm_inventory': np.nan if mm is None else sign*mm['inventory'].values[idx[j]],
                       'mm_pnl': np.nan if mm is None else mm['pnl'].values[idx[j]] - mm['pnl'].values[idx[np.searchsorted(grid, 0)]]}
                for kind in ('informed', 'metaorder', 'noise taker', 'strategies'):
                    row[kind] = flow.get(kind, np.zeros(len(grid)))[j] if dt >= 0 else 0.0
                rows.append(row)
    d = pd.DataFrame(rows)
    return d.groupby('dt').mean(), int(len(d)/len(d['dt'].unique())) if len(d) else 0


def plot_news(event, n_events, save_path=None):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axes[0]
    ax.plot(event.index, event['mispricing'], color='#e34948', linewidth=2)
    ax.axvline(0, color=MUTED, linewidth=1, linestyle='--')
    ax.axhline(0, color=MUTED, linewidth=0.8)
    m0 = event.loc[event.index >= 0, 'mispricing']
    half = m0[m0 <= m0.iloc[0]/2]
    if len(half):
        ax.axvline(half.index[0], color='#e34948', linewidth=1, linestyle=':')
        ax.text(half.index[0] + 30, m0.iloc[0]*0.55, 'half absorbed after %d events' % half.index[0], color=INK, fontsize=9)
    _style(ax, 'Information not yet in the price', 'events after the news', 'sign x (V - mid) (ticks)')
    ax = axes[1]
    ax.plot(event.index, event['mm_inventory'], color='#1baf7a', linewidth=2, label='inventory in the direction of the news (shares)')
    ax.plot(event.index, event['mm_pnl']/10, color='#6250d6', linewidth=2, linestyle='--', label='PnL since the news / 10 (ticks x shares)')
    ax.axvline(0, color=MUTED, linewidth=1, linestyle='--')
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8.5, loc='lower left')
    _style(ax, 'The market maker is on the wrong side of the news', 'events after the news', '')
    ax = axes[2]
    for kind, color in [('informed', '#e34948'), ('metaorder', '#eb6834'), ('noise taker', '#2a78d6'), ('strategies', '#eda100')]:
        if kind in event:
            ax.plot(event.index[event.index >= 0], event.loc[event.index >= 0, kind], color=color, linewidth=2, label=KIND_LABELS[kind])
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.legend(frameon=False, fontsize=9)
    _style(ax, 'Who trades in the direction of the news', 'events after the news', 'cumulative signed volume (shares)')
    fig.suptitle('Event study: average over %d news' % n_events, x=0.01, ha='left', fontsize=12, color=INK)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches='tight')
    return fig

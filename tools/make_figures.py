"""Runs every study with the parameters of the notebook and writes the figures of the README (docs/figures) and the result tables
(results/). Run from the repository root: python tools/make_figures.py"""
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import pandas as pd
from simulation import SimulationConfig, run_simulation
from analysis import plot_microstructure, plot_strategies, plot_square_root
from experiments import (ablation, stylized_facts_table, plot_stylized_facts, ecology_study, plot_ecology_overview, plot_flows,
                         plot_pnl_decomposition, news_event_study, plot_news, market_comparison, plot_market_comparison,
                         market_maker_sweep, plot_market_maker_sweep, impact_study, impact_table, plot_metaorder)

FIG = os.path.join(ROOT, 'docs', 'figures')
RES = os.path.join(ROOT, 'results')
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)
t = time.time()

first = run_simulation(SimulationConfig(seed=42, nb_trader_per_strat=100))
plot_microstructure(first, save_path=os.path.join(FIG, 'microstructure.png'))
plot_strategies(first.metrics, save_path=os.path.join(FIG, 'strategies.png'))
plot_square_root(save_path=os.path.join(FIG, 'square_root.png'))
first.metrics.to_csv(os.path.join(RES, 'strategy_metrics.csv'), index=False)

ab = ablation(baseline=first)
facts = stylized_facts_table(ab)
facts.to_csv(os.path.join(RES, 'stylized_facts.csv'))
plot_stylized_facts(ab, save_path=os.path.join(FIG, 'stylized_facts.png'))
print(facts.round(3))

runs, table = ecology_study()
table.to_csv(os.path.join(RES, 'ecology_pnl_decomposition.csv'), index=False)
plot_ecology_overview(runs[0], save_path=os.path.join(FIG, 'ecology_overview.png'))
fig, fm, mk = plot_flows(runs, save_path=os.path.join(FIG, 'flows_markouts.png'))
print((100*fm).round(1))
print(mk.pivot(index='horizon', columns='counterparty', values='markout').round(2))
fig, summary = plot_pnl_decomposition(table, save_path=os.path.join(FIG, 'pnl_decomposition.png'))
summary.to_csv(os.path.join(RES, 'ecology_pnl_by_kind.csv'))
print(summary.round(2))
event, n = news_event_study(runs)
plot_news(event, n, save_path=os.path.join(FIG, 'news.png'))
print('news', n, event.loc[[0, 500, 1000, 1500], ['mispricing', 'mm_inventory', 'mm_pnl', 'informed']].round(2))
cmp = market_comparison(first, table)
cmp.to_csv(os.path.join(RES, 'market_comparison.csv'), index=False)
plot_market_comparison(cmp, save_path=os.path.join(FIG, 'market_comparison.png'))
print(cmp.round(2))

sweep = market_maker_sweep()
sweep.to_csv(os.path.join(RES, 'market_maker_sweep.csv'), index=False)
plot_market_maker_sweep(sweep, save_path=os.path.join(FIG, 'market_maker_sweep.png'))
print(sweep.groupby(['informed_rate', 'half_spread'])[['total', 'volume']].mean().round(1))

study = impact_study()
it = impact_table(study)
it.to_csv(os.path.join(RES, 'impact_study.csv'))
plot_metaorder(study, save_path=os.path.join(FIG, 'metaorder_impact.png'))
print(it.round(2))
for name, (paths, peaks) in study.items():
    print(name, peaks.groupby('Q')[['peak', 'end', 'shortfall']].mean().round(2).to_dict())
print('total time %.0f s' % (time.time() - t))

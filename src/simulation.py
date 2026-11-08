import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from orderbook import OrderBook
from agents import (Trader_quantity, Trader_Buy_and_sell, Trader_random, Trader_bullish,
                    Trader_bearish, Trader_MA, generate_lim_orders)
from analysis import BookRecorder, tape_frame, order_frame, execution_costs

STRATEGIES = ['Trader_quantity', 'Trader_Buy_and_sell', 'Trader_random', 'Trader_bullish', 'Trader_bearish', 'Trader_MA']

# After some of the orders of the traders, new random limit orders are injected into the book: it is the reaction of the rest of
# the market. (number of orders, (shift of the lower, of the upper price bound of the bids), (shift of the lower, of the upper price bound of the asks))
REACTION_ORDERS = {
    'quantity_first': (5, (5, 5), (0, 0)),
    'buy_and_sell':   (5, (5, 5), (6, 5)),
    'bearish_first':  (2, (-5, -3), (-3, -5)),
    'bearish_second': (5, (5, 5), (7, 10)),
    'bullish_first':  (5, (5, 5), (4, 4)),
    'bullish_second': (3, (-4, -4), (-2, -2)),
    'MA':             (2, (1, 1), (1, 2)),
}


@dataclass
class SimulationConfig:
    '''
    Parameters of the baseline market simulation: the starting book, the trading rounds and the capital of each strategy.
    '''
    seed: int = 42
    nb_start_orders: int = 200 # number of limit orders in the book at the start
    price_range_bid_min: int = 90
    price_range_bid_max: int = 100
    price_range_ask_min: int = 95
    price_range_ask_max: int = 105
    qty_range_ask_min: int = 1
    qty_range_ask_max: int = 10
    qty_range_bid_min: int = 1
    qty_range_bid_max: int = 10
    nb_transac_times: int = 50 # number of times our traders can trade
    nb_trader_per_strat: int = 30 # number of traders per strategy = number of independent simulations
    K_trader_qty: int = 1000
    pnl_trader_qty: int = 0
    nb_shares_trader_qty: int = 0
    K_trader_random: int = 10000
    pnl_trader_random: int = 0
    nb_shares_trader_random: int = 50
    K_trader_bullish: int = 10000
    pnl_trader_bullish: int = 0
    nb_shares_trader_bullish: int = 0
    K_trader_bearish: int = 5000
    pnl_trader_bearish: int = 0
    nb_shares_trader_bearish: int = 100
    K_trader_MA: int = 10000
    pnl_trader_MA: int = 0
    nb_shares_trader_MA: int = 50
    mvg_trader_MA: int = 5 # order of the moving average
    reset_book: bool = True # every simulation starts from a new book, so that the simulations are independent

    def __post_init__(self):
        if self.price_range_bid_min > self.price_range_bid_max or self.price_range_ask_min > self.price_range_ask_max:
            raise ValueError('a minimum price is larger than the corresponding maximum price')
        if self.qty_range_ask_min < 1 or self.qty_range_bid_min < 1 or self.qty_range_ask_min > self.qty_range_ask_max or self.qty_range_bid_min > self.qty_range_bid_max:
            raise ValueError('the quantity ranges must satisfy 1 <= min <= max')
        for key, (nb, (bl, bh), (al, ah)) in REACTION_ORDERS.items():
            if self.price_range_bid_min + bl > self.price_range_bid_max + bh or self.price_range_ask_min + al > self.price_range_ask_max + ah:
                raise ValueError('the price ranges are too narrow for the reaction orders "%s"' % key)


def last_price(order_book):
    # price of the last trade, or the mid-price while nothing has been traded yet
    if len(order_book.tape) > 0:
        return order_book.tape[-1]['price']
    return order_book.get_mid_price()


def settle(order_book, trader, quote, state_before, mid_before, spread_before, recorder, exchange=None):
    '''
    Sends the order of a trader to the order book and corrects the optimistic accounting of the trader with what was really executed.
    Our traders only send marketable orders, so the part of a limit order that could not be executed is cancelled (immediate-or-cancel).
    With an exchange (see ecology.py) the order goes through it, so that the passive counterparties are told about their fills.
    '''
    K0, pnl0, shares0 = state_before
    if exchange is not None:
        trades, resting = exchange.send(trader, quote)
        if resting is not None:
            exchange.cancel(resting['side'], resting['order_id'])
    else:
        trades, resting = order_book.process_order(dict(quote), False, False)
        if recorder is not None:
            recorder.record(order_book)
        if resting is not None:
            order_book.cancel_order(resting['side'], resting['order_id'])
            if recorder is not None:
                recorder.record(order_book)
    filled = sum(t['quantity'] for t in trades)
    notional = sum(t['price'] * t['quantity'] for t in trades)
    sign = 1 if quote['side'] == 'ask' else -1 # a sell brings cash in and takes shares out
    trader.K = K0 + sign * notional
    trader.pnl = pnl0 + sign * notional
    trader.nb_shares = shares0 - sign * filled
    record = trader.trader_balance_sheet[-1]
    record['Current Capital'] = trader.K
    record['Cumulative Pnl'] = trader.pnl
    record['Gross Return'] = trader.gross_return()
    record['nb_shares'] = trader.nb_shares
    record['filled_quantity'] = filled
    vwap = notional / filled if filled > 0 else None
    record['avg_fill_price'] = vwap
    record['spread_before'] = np.nan if spread_before is None else float(spread_before)
    if filled > 0 and mid_before is not None:
        # execution cost per share versus the mid-price before the order: positive = paid by the trader
        record['slippage'] = float(vwap - mid_before) if quote['side'] == 'bid' else float(mid_before - vwap)
    else:
        record['slippage'] = np.nan
    return trades


class SimulationResult:
    '''Everything the simulation produces, grouped by strategy'''
    def __init__(self):
        self.sheets = {name: [] for name in STRATEGIES} # balance sheet of every trader (one per simulation)
        self.final_dist = {name: [] for name in STRATEGIES} # last record of every balance sheet
        self.trajecs = [] # last trade price of every simulation
        self.all_trajecs = [] # all the trade prices of every simulation
        self.metrics = [] # one row per (simulation, strategy)
        self.tapes = [] # trade tape of every simulation (DataFrame)
        self.snapshots = [] # top-of-book snapshots of every simulation (DataFrame)
        self.orders = [] # aggressive orders of every simulation (DataFrame)
        self.order_book = None # the book of the last simulation
        self.start_equity = [] # value of every trader at the start of every simulation


def run_simulation(cfg):
    '''
    Runs cfg.nb_trader_per_strat independent simulations of the baseline market.
    In every simulation there is one trader of each of the six strategies that trade on the same book.
    '''
    np.random.seed(cfg.seed)
    result = SimulationResult()
    order_book = None
    qty_range_ask = [cfg.qty_range_ask_min, cfg.qty_range_ask_max]
    qty_range_bid = [cfg.qty_range_bid_min, cfg.qty_range_bid_max]

    for n in range(cfg.nb_trader_per_strat):
        recorder = BookRecorder()
        if order_book is None or cfg.reset_book:
            order_book = OrderBook()
            start_orders = generate_lim_orders(cfg.nb_start_orders, [cfg.price_range_bid_min, cfg.price_range_bid_max],
                                               [cfg.price_range_ask_min, cfg.price_range_ask_max], qty_range_ask, qty_range_bid)
            for order in start_orders:
                order_book.process_order(order, False, False)
        if order_book.get_mid_price() is None:
            raise ValueError('the starting orders left one side of the book empty: widen the price ranges or add orders')
        tape_start = len(order_book.tape)
        recorder.record(order_book)
        mid_start = order_book.get_mid_price()

        all_trades = []
        traders = {
            'Trader_quantity': Trader_quantity(cfg.K_trader_qty, cfg.pnl_trader_qty, cfg.nb_shares_trader_qty),
            'Trader_Buy_and_sell': Trader_Buy_and_sell(order_book),
            'Trader_random': Trader_random(cfg.K_trader_random, cfg.pnl_trader_random, cfg.nb_shares_trader_random),
            'Trader_bullish': Trader_bullish(cfg.K_trader_bullish, cfg.pnl_trader_bullish, cfg.nb_shares_trader_bullish),
            'Trader_bearish': Trader_bearish(cfg.K_trader_bearish, cfg.pnl_trader_bearish, cfg.nb_shares_trader_bearish),
            'Trader_MA': Trader_MA(cfg.K_trader_MA, cfg.pnl_trader_MA, cfg.nb_shares_trader_MA, cfg.mvg_trader_MA),
        }
        # value of every trader at the start: cash + shares valued at the initial mid-price (the shares of
        # Trader_Buy_and_sell are bought at the start of the simulation, so he starts with cash only)
        start_equity = {name: float(t.K_initial) + (0 if name == 'Trader_Buy_and_sell' else float(t.nb_shares) * float(mid_start)) for name, t in traders.items()}
        result.start_equity.append(start_equity)
        start_shares = {name: (0 if name == 'Trader_Buy_and_sell' else float(t.nb_shares)) for name, t in traders.items()}
        all_trades.append(last_price(order_book))

        def send(trader, method):
            # the trader decides, then his order goes to the book
            state = (trader.K, trader.pnl, trader.nb_shares)
            mid_before, spread_before = order_book.get_mid_price(), order_book.get_spread()
            val, quote, balance_sheet = method(order_book)
            if val == True:
                settle(order_book, trader, quote, state, mid_before, spread_before, recorder)
                all_trades.append(last_price(order_book))
            return val

        def react(key):
            # the rest of the market reacts with new random limit orders
            nb, (bl, bh), (al, ah) = REACTION_ORDERS[key]
            orders = generate_lim_orders(nb, [cfg.price_range_bid_min + bl, cfg.price_range_bid_max + bh],
                                         [cfg.price_range_ask_min + al, cfg.price_range_ask_max + ah], qty_range_ask, qty_range_bid)
            for order in orders:
                order_book.process_order(order, False, False)
                recorder.record(order_book)
                all_trades.append(last_price(order_book))

        # The initial purchase of Trader_Buy_and_sell
        T = traders['Trader_Buy_and_sell']
        state = (T.K_initial, 0, 0) # the state before the purchase assumed by the constructor
        mid_before, spread_before = order_book.get_mid_price(), order_book.get_spread()
        settle(order_book, T, T.initial_quote, state, mid_before, spread_before, recorder)
        all_trades.append(last_price(order_book))
        # settle() works on the last record of the balance sheet, which is the one written by the constructor

        for nb in np.arange(cfg.nb_transac_times):
            if send(traders['Trader_quantity'], traders['Trader_quantity'].trade_first):
                react('quantity_first')
            send(traders['Trader_quantity'], traders['Trader_quantity'].tradeSecond)

            if send(traders['Trader_Buy_and_sell'], traders['Trader_Buy_and_sell'].trade):
                react('buy_and_sell')

            send(traders['Trader_random'], traders['Trader_random'].trade)

            if send(traders['Trader_bearish'], traders['Trader_bearish'].tradeFirst):
                react('bearish_first')
            if send(traders['Trader_bearish'], traders['Trader_bearish'].SecondTrade):
                react('bearish_second')
            send(traders['Trader_bearish'], traders['Trader_bearish'].ThirdTrade)

            if send(traders['Trader_bullish'], traders['Trader_bullish'].tradeFirst):
                react('bullish_first')
            if send(traders['Trader_bullish'], traders['Trader_bullish'].SecondTrade):
                react('bullish_second')

            if send(traders['Trader_MA'], traders['Trader_MA'].firstTrade):
                react('MA')

        mid_end = order_book.get_mid_price()
        mid_end = float(last_price(order_book)) if mid_end is None else float(mid_end)
        for name, t in traders.items():
            sheet = t.trader_balance_sheet
            result.sheets[name].append(sheet)
            result.final_dist[name].append(sheet[-1])
            equity = float(t.K) + float(t.nb_shares) * mid_end
            spread_cost, total_cost, volume = execution_costs(sheet)
            result.metrics.append({
                'simulation': n, 'strategy': name,
                'mtm_pnl': equity - start_equity[name],
                'passive_pnl': start_shares[name] * (mid_end - float(mid_start)),
                'orders_sent': sum(1 for r in sheet if isinstance(r['order_sent']['type'], str)),
                'filled_volume': volume, 'half_spread_cost': spread_cost, 'execution_cost': total_cost,
                'final_cash': float(t.K), 'final_shares': float(t.nb_shares)})
        result.trajecs.append(all_trades[-1])
        result.all_trajecs.append(all_trades)

        tape = [t for i, t in enumerate(order_book.tape) if i >= tape_start]
        snap = recorder.frame()
        trades_df = tape_frame(tape)
        result.tapes.append(trades_df)
        result.snapshots.append(snap)
        result.orders.append(order_frame(trades_df, snap))

    result.order_book = order_book
    result.metrics = pd.DataFrame(result.metrics)
    result.metrics['excess_pnl'] = result.metrics['mtm_pnl'] - result.metrics['passive_pnl']
    return result


def GetStats(Balancesheet,strategy_name, order_book, all_trades):
  '''Cross-sectional statistics (over the simulations) of the end of period balance sheet records of one strategy'''

  current_cap = []
  cum_pnl = []
  gross_ret = []

  for i in range(len(Balancesheet)):
    current_cap.append(float(Balancesheet[i]['Current Capital']))
    cum_pnl.append(float(Balancesheet[i]['Cumulative Pnl']))
    gross_ret.append(float(Balancesheet[i]['Gross Return']))

  all_trades = [float(x) for x in all_trades]
  trades_df = pd.DataFrame(list(zip(list(np.arange(len(all_trades))), all_trades)), columns = ['Trade order', 'Trade price'])
  mean_current_cap = np.mean(current_cap)
  std_current_cap = np.std(current_cap)
  min_current_cap = np.min(current_cap)
  max_current_cap = np.max(current_cap)
  q1_current_cap = np.quantile(current_cap, 0.25)
  q2_current_cap = np.quantile(current_cap, 0.5)
  q3_current_cap = np.quantile(current_cap, 0.75)
  q95_current_cap = np.quantile(current_cap, 0.95)
  count = len(current_cap)
  df_stats_current_cap = pd.DataFrame([count, mean_current_cap, std_current_cap, min_current_cap, max_current_cap,q1_current_cap,q2_current_cap,q3_current_cap, q95_current_cap],
                                      columns=['Descriptive statistics Current Capital for the strategy: {}'.format(strategy_name)],
                                      index=['count', 'Mean', 'Std', 'Min', 'Max', '1st quartile', 'median', '3rd quartile', '95th quantile'])

  mean_cum_pnl = np.mean(cum_pnl)
  std_cum_pnl = np.std(cum_pnl)
  min_cum_pnl = np.min(cum_pnl)
  max_cum_pnl = np.max(cum_pnl)
  q1_cum_pnl = np.quantile(cum_pnl, 0.25)
  q2_cum_pnl = np.quantile(cum_pnl, 0.5)
  q3_cum_pnl = np.quantile(cum_pnl, 0.75)
  q95_cum_pnl = np.quantile(cum_pnl, 0.95)
  count = len(cum_pnl)
  df_stats_cum_pnl = pd.DataFrame([count, mean_cum_pnl, std_cum_pnl, min_cum_pnl, max_cum_pnl,q1_cum_pnl,q2_cum_pnl,q3_cum_pnl, q95_cum_pnl],
                                      columns=['Descriptive statistics Cumulative Pnl for the strategy: {}'.format(strategy_name)],
                                      index=['count', 'Mean', 'Std', 'Min', 'Max', '1st quartile', 'median', '3rd quartile', '95th quantile'])

  mean_gross_ret = np.mean(gross_ret)
  std_gross_ret = np.std(gross_ret)
  min_gross_ret = np.min(gross_ret)
  max_gross_ret = np.max(gross_ret)
  count = len(gross_ret)
  q1_gross_ret = np.quantile(gross_ret, 0.25)
  q2_gross_ret = np.quantile(gross_ret, 0.5)
  q3_gross_ret = np.quantile(gross_ret, 0.75)
  q95_gross_ret = np.quantile(gross_ret, 0.95)
  df_stats_gross_ret = pd.DataFrame([count, mean_gross_ret, std_gross_ret, min_gross_ret, max_gross_ret,q1_gross_ret,q2_gross_ret,q3_gross_ret, q95_gross_ret],
                                      columns=['Descriptive statistics Gross Return for the strategy: {}'.format(strategy_name)],
                                      index=['count', 'Mean', 'Std', 'Min', 'Max', '1st quartile', 'median', '3rd quartile', '95th quantile'])

  mean_all_trades = np.mean(all_trades)
  std_all_trades = np.std(all_trades)
  min_all_trades = np.min(all_trades)
  max_all_trades = np.max(all_trades)
  count = len(all_trades)
  q1_all_trades = np.quantile(all_trades, 0.25)
  q2_all_trades = np.quantile(all_trades, 0.5)
  q3_all_trades = np.quantile(all_trades, 0.75)
  q95_all_trades = np.quantile(all_trades, 0.95)
  df_stats_all_trades = pd.DataFrame([count, mean_all_trades, std_all_trades, min_all_trades, max_all_trades, q1_all_trades, q2_all_trades, q3_all_trades, q95_all_trades],
                                      columns=['Descriptive statistics Trade Price for the strategy: {}'.format(strategy_name)],
                                      index=['count', 'Mean', 'Std', 'Min', 'Max', '1st quartile', 'median', '3rd quartile', '95th quantile'])

  return df_stats_current_cap, df_stats_cum_pnl, df_stats_gross_ret, trades_df, current_cap, cum_pnl, gross_ret


def GetList(traders, strategy_name, trajecs, save_path=None):

    for Balancesheet, j in zip(traders, range(len(traders))):
      current_cap = []
      cum_pnl = []
      gross_ret = []
      trajec = []

      for i in range(len(Balancesheet)):
        current_cap.append(float(Balancesheet[i]['Current Capital']))
        cum_pnl.append(float(Balancesheet[i]['Cumulative Pnl']))
        gross_ret.append(float(Balancesheet[i]['Gross Return']))

      for i in range(len(trajecs[j])):
        trajec.append(float(trajecs[j][i]))
      current_cap = np.array(current_cap)
      cum_pnl = np.array(cum_pnl)
      gross_ret = np.array(gross_ret)

      if j==0:
        df_current_cap = pd.DataFrame(current_cap, columns=["trader "+str(j)+' '+strategy_name])
        df_cum_pnl = pd.DataFrame(cum_pnl, columns=["trader "+str(j)+' '+strategy_name])
        df_gross_ret = pd.DataFrame(gross_ret, columns=["trader "+str(j)+' '+strategy_name])
        df_trajec = pd.DataFrame(trajec, columns=["trajectories "+str(j)+' '+strategy_name])

      else:
        df_current_cap["trader "+str(j)+' '+strategy_name] = current_cap
        df_cum_pnl["trader "+str(j)+' '+strategy_name] = cum_pnl
        df_gross_ret["trader "+str(j)+' '+strategy_name] = gross_ret
        # the trajectories do not all have the same length: one column per simulation, aligned on the first trade
        df_trajec = pd.concat([df_trajec, pd.DataFrame(trajec, columns=["trajectories "+str(j)+' '+strategy_name])], axis=1)
    GetVisualization2(df_current_cap, df_cum_pnl, df_gross_ret,df_trajec, strategy_name, save_path)

def GetVisualization2(df_current_cap, df_cum_pnl, df_gross_ret,df_trajec, strategy_name, save_path=None):
  # One panel per quantity. Instead of one line per simulation, the median over the simulations, the 25%-75% band and
  # the 5%-95% band (a fan chart), plus a few simulations drawn thin to show what a single path looks like
  fig=plt.figure(figsize=(16,8.5))
  panels = [(df_cum_pnl, 'Cumulative Pnl (cash flow)', 'step of the trader (record of his balance sheet)'),
            (df_gross_ret, 'Gross Return (cash / initial cash)', 'step of the trader (record of his balance sheet)'),
            (df_current_cap, 'Current Capital', 'step of the trader (record of his balance sheet)'),
            (df_trajec, 'Trade price of the market', 'trade number in the simulation')]
  for k, (df, title, xlabel) in enumerate(panels):
    ax = fig.add_subplot(2, 2, k + 1)
    q = df.quantile([0.05, 0.25, 0.5, 0.75, 0.95], axis=1).T
    q[df.notna().sum(axis=1) < max(5, 0.2*df.shape[1])] = np.nan # too few simulations reach this step: no band
    x = np.arange(len(q))
    ax.fill_between(x, q[0.05], q[0.95], color='#9ec5f4', alpha=0.45, linewidth=0, label='5%-95% of the simulations')
    ax.fill_between(x, q[0.25], q[0.75], color='#5598e7', alpha=0.55, linewidth=0, label='25%-75%')
    for c in df.columns[:3]:
      ax.plot(np.arange(len(df)), df[c].values, color='#52514e', linewidth=0.6, alpha=0.6)
    ax.plot(x, q[0.5], color='#0d366b', linewidth=2, label='median')
    ax.set_title('{} : {}'.format(title, strategy_name), fontsize=11, loc='left', color='#0b0b0b')
    ax.set_xlabel(xlabel, fontsize=9, color='#52514e')
    ax.grid(True, color='#e4e3df', linewidth=0.8)
    for side in ('top', 'right'):
      ax.spines[side].set_visible(False)
    if k == 0:
      ax.legend(frameon=False, fontsize=8.5)
  fig.tight_layout()
  if save_path is not None:
    fig.savefig(save_path, dpi=110, bbox_inches='tight')
  plt.show()

def end_of_period_tables(result, strategy_names=None):
    '''Descriptive statistics of the end of period balance sheets of each strategy over the simulations'''
    tables = {}
    for name in (strategy_names or STRATEGIES):
        df_cap, df_pnl, df_ret, trades_df, cap, pnl, ret = GetStats(result.final_dist[name], name, result.order_book, result.trajecs)
        table = pd.concat([df_cap, df_pnl, df_ret], axis=1, ignore_index=True)
        table.columns = ['Current Capital', 'Cumulative Pnl', 'Gross Return']
        tables[name] = table
    return tables


def export_tables(tables, nb_simulations, directory='results'):
    '''One Excel file per strategy (and the same table as csv)'''
    os.makedirs(directory, exist_ok=True)
    for name, table in tables.items():
        path = os.path.join(directory, 'End-of-period statistics with the strategy {} over {} simulations'.format(name, nb_simulations))
        with pd.ExcelWriter(path + '.xlsx') as writer:
            table.to_excel(writer)
        table.to_csv(path + '.csv')

# A market ecology for our limit order book.
# In the baseline market (simulation.py) the rest of the market is a uniform random flow of limit orders plus fixed reaction
# orders, and nobody is ever filled passively. Here the rest of the market is a set of interacting agents, each of them modelled
# on a chapter of Trades, Quotes and Prices (Bouchaud, Bonart, Donier, Gould, 2018):
#   Trader_noise_maker  : the zero-intelligence liquidity of the Santa Fe model, limit orders and cancellations (chapter 8)
#   Trader_noise_taker  : the zero-intelligence market orders of the Santa Fe model (chapter 8)
#   Trader_latent_liquidity : slow investors whose orders sit around a slowly moving consensus price, with a depth that grows
#                         linearly with the distance (latent liquidity, chapters 18 and 19, Mastromatteo, Toth and Bouchaud 2014)
#   Trader_market_maker : quotes on both sides around a reservation price skewed by his inventory (chapters 16 and 17)
#   Trader_informed     : trades towards a fundamental value that only he sees, the source of adverse selection (chapters 15 and 16)
#   Trader_metaorder    : institutional investors who split large orders with power-law sizes into small child orders (chapters 10.5 and 12)
# The six directional strategies of agents.py can trade in this market too (use_strategies).
# All the prices are handled in ticks (integers) by the agents and converted to Decimal on the tick grid of the book.
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from orderbook import OrderBook, to_decimal
from agents import Trader_quantity, Trader_Buy_and_sell, Trader_random, Trader_bullish, Trader_bearish, Trader_MA
from analysis import BookRecorder, order_frame
from simulation import SimulationConfig, settle, STRATEGIES

KINDS = ['noise maker', 'noise taker', 'latent liquidity', 'market maker', 'informed', 'metaorder'] + STRATEGIES

# Order in which the six directional strategies act, the same order as in the baseline market
STRATEGY_STEPS = [('Trader_quantity', 'trade_first'), ('Trader_quantity', 'tradeSecond'), ('Trader_Buy_and_sell', 'trade'),
                ('Trader_random', 'trade'), ('Trader_bearish', 'tradeFirst'), ('Trader_bearish', 'SecondTrade'),
                ('Trader_bearish', 'ThirdTrade'), ('Trader_bullish', 'tradeFirst'), ('Trader_bullish', 'SecondTrade'),
                ('Trader_MA', 'firstTrade')]


class Exchange:
  '''
  Sits between the agents and the OrderBook. It gives every agent a unique trade_id, sends the orders, and tells the two
  counterparties of every trade that they were filled (on_fill), the passive one included. Every fill is also written in
  self.fills: (time, price, quantity, sign of the aggressor, trade_id of the aggressor, trade_id of the passive side).
  '''
  def __init__(self, order_book, recorder=None):
    self.order_book = order_book
    self.recorder = recorder
    self.agents = {}
    self.kinds = {}
    self.fills = []

  def register(self, agent, kind):
    agent.trade_id = len(self.agents) + 1
    agent.kind = kind
    self.agents[agent.trade_id] = agent
    self.kinds[agent.trade_id] = kind
    return agent

  def send(self, agent, quote):
    quote = dict(quote)
    quote['trade_id'] = agent.trade_id
    trades, resting = self.order_book.process_order(quote, False, False)
    sign = 1 if quote['side'] == 'bid' else -1
    for t in trades:
      passive = self.agents.get(t['party1'][0])
      if passive is not None and hasattr(passive, 'on_fill'):
        passive.on_fill(-sign, t['price'], t['quantity'], t['party1'][2], t['party1'][3])
      if hasattr(agent, 'on_fill'):
        agent.on_fill(sign, t['price'], t['quantity'], None, None)
      self.fills.append((t['time'], t['price'], t['quantity'], sign, agent.trade_id, t['party1'][0]))
    if self.recorder is not None:
      self.recorder.record(self.order_book)
    return trades, resting

  def cancel(self, side, order_id):
    self.order_book.cancel_order(side, order_id)
    if self.recorder is not None:
      self.recorder.record(self.order_book)


def best_ticks(order_book, tick):
  # best bid and best ask in ticks (None when the side is empty)
  bb, ba = order_book.get_best_bid(), order_book.get_best_ask()
  return (None if bb is None else int(bb / tick)), (None if ba is None else int(ba / tick))


class Agent:
  '''Accounting shared by the agents of the ecology: they are told about every fill, aggressive or passive'''
  def init_account(self, K, pnl, nb_shares):
    self.K = to_decimal(K)
    self.K_initial = self.K
    self.pnl = to_decimal(pnl)
    self.nb_shares = to_decimal(nb_shares)
    self.live = {} # order_id -> side of the orders of the agent resting in the book
    self.trader_balance_sheet = []

  def gross_return(self):
    return self.K/self.K_initial if self.K_initial else 0

  def on_fill(self, sign, price, quantity, order_id, remaining):
    # sign = +1 when the agent bought, -1 when he sold. remaining is None when his resting order was fully executed
    self.K -= sign*price*quantity
    self.pnl -= sign*price*quantity
    self.nb_shares += sign*quantity
    if order_id is not None and remaining is None:
      self.live.pop(order_id, None)

  def write(self, quote):
    record = {'Current Capital' : self.K, 'Cumulative Pnl' : self.pnl,'Gross Return' : self.gross_return(), 'nb_shares' : self.nb_shares,'order_sent' : quote}
    self.trader_balance_sheet.append(record)

  def post(self, exchange, side, price_ticks, quantity, tick):
    quote = {'type' : 'limit', 'side' : side, 'quantity' : quantity, 'price' : tick*price_ticks, 'trade_id' : self.trade_id}
    trades, resting = exchange.send(self, quote)
    if resting is not None:
      self.live[resting['order_id']] = side
    return quote

  def cancel_all(self, exchange):
    for order_id, side in list(self.live.items()):
      exchange.cancel(side, order_id)
    self.live = {}


def geometric_size(u, mean):
  # order sizes: 1 + an exponential variable, rounded down (mean about `mean` shares)
  return 1 + int(-math.log(1 - u)*(mean - 0.5))


class Trader_noise_maker(Agent):
  # The zero-intelligence liquidity of the Santa Fe model: many anonymous traders who place limit orders at a uniform distance
  # (1 to `window` ticks) from the opposite best quote, and cancel each of their orders at a constant rate
  def __init__(self, K, pnl, nb_shares, window, size_mean, tick, anchor_ticks):
    self.init_account(K, pnl, nb_shares)
    self.window, self.size_mean, self.tick, self.anchor = window, size_mean, tick, anchor_ticks
    self.ids = [] # the live order ids in a list, to pick one at random in O(1)
    self.position = {}

  def on_fill(self, sign, price, quantity, order_id, remaining):
    Agent.on_fill(self, sign, price, quantity, None, None)
    if order_id is not None and remaining is None:
      self._forget(order_id)

  def _forget(self, order_id):
    k = self.position.pop(order_id, None)
    if k is None:
      return
    last = self.ids.pop()
    if k < len(self.ids):
      self.ids[k] = last
      self.position[last] = k
    self.live.pop(order_id, None)

  def limit_order(self, exchange, u_side, u_price, u_size):
    bb, ba = best_ticks(exchange.order_book, self.tick)
    distance = 1 + int(u_price*self.window)
    if u_side < 0.5:
      side = 'bid'
      ref = ba if ba is not None else (bb + 1 if bb is not None else self.anchor)
      price = ref - distance
    else:
      side = 'ask'
      ref = bb if bb is not None else (ba - 1 if ba is not None else self.anchor)
      price = ref + distance
    if price <= 0:
      return None
    quote = {'type' : 'limit', 'side' : side, 'quantity' : geometric_size(u_size, self.size_mean), 'price' : self.tick*price, 'trade_id' : self.trade_id}
    trades, resting = exchange.send(self, quote)
    if resting is not None:
      self.live[resting['order_id']] = side
      self.position[resting['order_id']] = len(self.ids)
      self.ids.append(resting['order_id'])
    return quote

  def cancel_one(self, exchange, u):
    if not self.ids:
      return None
    order_id = self.ids[int(u*len(self.ids))]
    side = self.live[order_id]
    exchange.cancel(side, order_id)
    self._forget(order_id)
    return order_id


class Trader_latent_liquidity(Trader_noise_maker):
  # Slow investors (pension funds, asset managers) who are willing to buy below and sell above a consensus price that only moves
  # slowly (an exponential average of the mid-price over `tau` events). Their orders are placed at a distance d from this consensus
  # with a density proportional to d: almost nothing close to the price, more and more liquidity further away. This is the
  # V-shaped latent order book of chapters 18 and 19. When the price is pushed away from the consensus, it meets more and more
  # of this liquidity, and when the price is below the consensus some of their buy orders are directly executable.
  def __init__(self, K, pnl, nb_shares, window, size_mean, tick, anchor_ticks, tau):
    Trader_noise_maker.__init__(self, K, pnl, nb_shares, window, size_mean, tick, anchor_ticks)
    self.consensus = float(anchor_ticks)
    self.tau = tau

  def update(self, mid_ticks):
    if mid_ticks == mid_ticks: # not NaN
      self.consensus += (mid_ticks - self.consensus)/self.tau

  def limit_order(self, exchange, u_side, u_price, u_size):
    distance = 1 + int(self.window*math.sqrt(u_price)) # density of the distance proportional to the distance
    centre = int(round(self.consensus))
    side = 'bid' if u_side < 0.5 else 'ask'
    price = centre - distance if side == 'bid' else centre + distance
    if price <= 0:
      return None
    quote = {'type' : 'limit', 'side' : side, 'quantity' : geometric_size(u_size, self.size_mean), 'price' : self.tick*price, 'trade_id' : self.trade_id}
    trades, resting = exchange.send(self, quote)
    if resting is not None:
      self.live[resting['order_id']] = side
      self.position[resting['order_id']] = len(self.ids)
      self.ids.append(resting['order_id'])
    return quote


class Trader_noise_taker(Agent):
  # The zero-intelligence market orders of the Santa Fe model: buy or sell with equal probability, independently of everything
  def __init__(self, K, pnl, nb_shares, size_mean):
    self.init_account(K, pnl, nb_shares)
    self.size_mean = size_mean

  def trade(self, exchange, u_side, u_size):
    side = 'bid' if u_side < 0.5 else 'ask'
    quote = {'type': 'market', 'side': side, 'quantity': geometric_size(u_size, self.size_mean), 'trade_id': self.trade_id}
    exchange.send(self, quote)
    return quote


class Trader_market_maker(Agent):
  # This trader quotes on both sides of the book. He centres his quotes on a reservation price r = mid - skew*inventory
  # (the inventory control of Avellaneda and Stoikov, chapter 17): when he holds too many shares he lowers both quotes to sell
  # more and buy less. He earns the spread on the uninformed flow and loses on the informed flow (adverse selection, chapter 16).
  def __init__(self, K, pnl, nb_shares, half_spread, skew, size, max_inventory, tick):
    self.init_account(K, pnl, nb_shares)
    self.half_spread, self.skew, self.size, self.max_inventory, self.tick = half_spread, skew, size, max_inventory, tick
    self.inventory_path = []

  def requote(self, exchange):
    self.cancel_all(exchange)
    bb, ba = best_ticks(exchange.order_book, self.tick)
    if bb is None or ba is None:
      self.val = False
      return self.val
    inventory = float(self.nb_shares)
    r = (bb + ba)/2 - self.skew*inventory
    bid = min(math.floor(r - self.half_spread), ba - 1)
    ask = max(math.ceil(r + self.half_spread), bb + 1)
    if ask <= bid: # never quote against himself
      ask = bid + 1
    quotes = []
    if inventory < self.max_inventory and bid > 0:
      quotes.append(self.post(exchange, 'bid', bid, min(self.size, self.max_inventory - inventory), self.tick))
    if inventory > -self.max_inventory:
      quotes.append(self.post(exchange, 'ask', ask, min(self.size, self.max_inventory + inventory), self.tick))
    self.inventory_path.append((exchange.order_book.time, inventory))
    self.write(quotes)
    self.val = True
    return self.val


class Trader_informed(Agent):
  # This trader knows the fundamental value V of the share (a random walk with rare jumps: the news). When the best ask is
  # below V by more than a threshold he buys with a market order, when the best bid is above V by more than the threshold
  # he sells. He is the one who brings the information into the price.
  def __init__(self, K, pnl, nb_shares, threshold, size, max_position, tick):
    self.init_account(K, pnl, nb_shares)
    self.threshold, self.size, self.max_position, self.tick = threshold, size, max_position, tick

  def trade(self, exchange, value_ticks):
    bb, ba = best_ticks(exchange.order_book, self.tick)
    position = float(self.nb_shares)
    quote = None
    if ba is not None and value_ticks - ba > self.threshold and position < self.max_position:
      quote = {'type': 'market', 'side': 'bid', 'quantity': min(self.size, self.max_position - position), 'trade_id': self.trade_id}
    elif bb is not None and bb - value_ticks > self.threshold and position > -self.max_position:
      quote = {'type': 'market', 'side': 'ask', 'quantity': min(self.size, self.max_position + position), 'trade_id': self.trade_id}
    if quote is not None:
      exchange.send(self, quote)
      self.write(quote)
    return quote


class Trader_metaorder(Agent):
  # An institutional investor who must buy or sell a large quantity Q (a metaorder, chapter 12). Q has a Pareto distribution
  # P(Q > q) = (q_min / q)^alpha, and the metaorder is executed with small child market orders. Lillo, Mike and Farmer showed
  # that this splitting makes the signs of the market orders autocorrelated with C(l) ~ l^-(alpha - 1) (chapter 10.5).
  def __init__(self, K, pnl, nb_shares, alpha, q_min, q_max, child_size):
    self.init_account(K, pnl, nb_shares)
    self.alpha, self.q_min, self.q_max, self.child_size = alpha, q_min, q_max, child_size
    self.remaining, self.sign = 0, 0
    self.metaorders = [] # (start time, sign, size, end time)

  def trade(self, exchange, u_sign, u_size):
    if self.remaining <= 0:
      self.sign = 1 if u_sign < 0.5 else -1
      self.remaining = min(int(self.q_min*(1 - u_size)**(-1/self.alpha)), self.q_max)
      self.metaorders.append([exchange.order_book.time, self.sign, self.remaining, None])
    side = 'bid' if self.sign > 0 else 'ask'
    quantity = min(self.child_size, self.remaining)
    quote = {'type': 'market', 'side': side, 'quantity': quantity, 'trade_id': self.trade_id}
    trades, _ = exchange.send(self, quote)
    self.remaining -= quantity # an unfilled child (empty side) is not sent again: the investor gives up on it
    if self.remaining <= 0:
      self.metaorders[-1][3] = exchange.order_book.time
    return quote


@dataclass
class EcologyConfig:
  '''Parameters of the market ecology. Rates are relative: at every event, an agent is drawn with a probability proportional to its rate.'''
  seed: int = 11
  nb_events: int = 20000 # events after the warm-up
  warmup: int = 3000 # events of noise flow only, to build the book
  tick: float = 0.01
  start_price: float = 100.0
  # zero-intelligence (Santa Fe) flow
  window: int = 20 # limit orders are placed 1 to `window` ticks away from the opposite best quote
  limit_rate: float = 1.0 # per price level of the window and per side
  market_rate: float = 2.5 # noise market orders, per side
  cancel_rate: float = 0.15 # per resting noise order
  noise_size: float = 5.0
  # latent liquidity
  use_latent: bool = False
  latent_rate: float = 10.0
  latent_window: int = 60 # ticks
  latent_cancel_rate: float = 0.02 # per resting latent order
  latent_tau: float = 5000.0 # events, memory of the consensus price
  # market maker
  use_market_maker: bool = True
  mm_rate: float = 6.0
  mm_half_spread: float = 1.0 # ticks
  mm_skew: float = 0.05 # ticks of reservation price per share of inventory
  mm_size: int = 10
  mm_max_inventory: int = 100
  # informed trader and fundamental value
  use_informed: bool = True
  informed_rate: float = 3.0
  informed_threshold: float = 3.0 # ticks
  informed_size: int = 10
  informed_max_position: int = 3000
  fundamental_vol: float = 0.15 # ticks per event
  jump_probability: float = 0.0005 # news, per event
  jump_size: float = 15.0 # ticks
  # metaorders
  use_metaorders: bool = True
  nb_metaorder_traders: int = 3
  metaorder_rate: float = 1.5 # per institutional trader
  metaorder_alpha: float = 1.5
  metaorder_min: int = 10
  metaorder_max: int = 5000
  child_size: int = 5
  # the six directional strategies of agents.py
  use_strategies: bool = False
  strategy_rate: float = 0.5
  strategy_config: SimulationConfig = None
  record: bool = True


class EcologyResult:
  '''What a run of the ecology produces'''
  pass


def fundamental_path(cfg):
  # V_t = V_(t-1) + vol * N(0,1) + rare jumps (news). It has its own random generator, so that the market can be replayed
  rng = np.random.default_rng(cfg.seed + 1_000_003)
  n = cfg.warmup + cfg.nb_events
  steps = cfg.fundamental_vol*rng.standard_normal(n)
  jumps = (rng.random(n) < cfg.jump_probability)*cfg.jump_size*np.where(rng.random(n) < 0.5, -1, 1)
  return cfg.start_price/cfg.tick + np.cumsum(steps + jumps)


def run_ecology(cfg, hook=None):
  '''
  Runs the market ecology. Every event draws 4 uniform numbers from one generator whatever happens, so that two runs with the
  same seed see the same random numbers (common random numbers): `hook(i, exchange)` is called after the i-th event and can
  send extra orders (a test metaorder for instance) without changing the random numbers seen by the rest of the market.
  '''
  if cfg.use_strategies:
    np.random.seed(cfg.seed) # the six directional strategies use numpy's global generator
  rng = np.random.default_rng(cfg.seed)
  tick = to_decimal(cfg.tick)
  anchor = int(round(cfg.start_price/cfg.tick))
  order_book = OrderBook(tick_size=cfg.tick)
  recorder = BookRecorder(levels=5)
  exchange = Exchange(order_book, None)
  maker = exchange.register(Trader_noise_maker(10**9, 0, 10**7, cfg.window, cfg.noise_size, tick, anchor), 'noise maker')
  taker = exchange.register(Trader_noise_taker(10**9, 0, 10**7, cfg.noise_size), 'noise taker')
  latent = None
  if cfg.use_latent:
    latent = exchange.register(Trader_latent_liquidity(10**9, 0, 10**7, cfg.latent_window, cfg.noise_size, tick, anchor, cfg.latent_tau), 'latent liquidity')
  value = fundamental_path(cfg)
  mm = informed = None
  meta, strategies = [], {}
  mids = np.full(cfg.warmup + cfg.nb_events, np.nan)
  values = []
  step = 0

  for i in range(cfg.warmup + cfg.nb_events):
    u = rng.random(4)
    if i == cfg.warmup:
      # the book is built: the other agents arrive
      if cfg.record:
        exchange.recorder = recorder
        recorder.record(order_book)
      start_time = order_book.time
      if cfg.use_market_maker:
        mm = exchange.register(Trader_market_maker(100000, 0, 0, cfg.mm_half_spread, cfg.mm_skew, cfg.mm_size, cfg.mm_max_inventory, tick), 'market maker')
      if cfg.use_informed:
        informed = exchange.register(Trader_informed(100000, 0, 0, cfg.informed_threshold, cfg.informed_size, cfg.informed_max_position, tick), 'informed')
      if cfg.use_metaorders:
        meta = [exchange.register(Trader_metaorder(10**7, 0, 0, cfg.metaorder_alpha, cfg.metaorder_min, cfg.metaorder_max, cfg.child_size), 'metaorder')
                for _ in range(cfg.nb_metaorder_traders)]
      if cfg.use_strategies:
        lc = cfg.strategy_config or SimulationConfig()
        strategies = {
          'Trader_quantity': Trader_quantity(lc.K_trader_qty, lc.pnl_trader_qty, lc.nb_shares_trader_qty),
          'Trader_Buy_and_sell': Trader_Buy_and_sell(order_book),
          'Trader_random': Trader_random(lc.K_trader_random, lc.pnl_trader_random, lc.nb_shares_trader_random),
          'Trader_bullish': Trader_bullish(lc.K_trader_bullish, lc.pnl_trader_bullish, lc.nb_shares_trader_bullish),
          'Trader_bearish': Trader_bearish(lc.K_trader_bearish, lc.pnl_trader_bearish, lc.nb_shares_trader_bearish),
          'Trader_MA': Trader_MA(lc.K_trader_MA, lc.pnl_trader_MA, lc.nb_shares_trader_MA, lc.mvg_trader_MA),
        }
        for name, t in strategies.items():
          exchange.register(t, name)
        T = strategies['Trader_Buy_and_sell']
        settle(order_book, T, T.initial_quote, (T.K_initial, 0, 0), order_book.get_mid_price(), order_book.get_spread(), None, exchange)
      strategy_start = {name: (float(t.K_initial), 0.0 if name == 'Trader_Buy_and_sell' else float(t.nb_shares)) for name, t in strategies.items()}

    active = i >= cfg.warmup
    rates = [2*cfg.window*cfg.limit_rate, 2*cfg.market_rate, cfg.cancel_rate*len(maker.ids),
             cfg.mm_rate if mm is not None else 0, cfg.informed_rate if informed is not None else 0,
             cfg.metaorder_rate*len(meta), cfg.strategy_rate if strategies else 0,
             cfg.latent_rate if latent is not None else 0, cfg.latent_cancel_rate*len(latent.ids) if latent is not None else 0]
    cumulative = np.cumsum(rates)
    x = u[0]*cumulative[-1]
    k = min(int(np.searchsorted(cumulative, x, side='right')), max(j for j, r in enumerate(rates) if r > 0))
    w = min((x - (cumulative[k - 1] if k > 0 else 0))/rates[k], 0.999999) # uniform in [0, 1) again, used inside the event
    if k == 0:
      maker.limit_order(exchange, w, u[1], u[2])
    elif k == 1:
      taker.trade(exchange, w, u[1])
    elif k == 2:
      maker.cancel_one(exchange, w)
    elif k == 3:
      mm.requote(exchange)
    elif k == 4:
      informed.trade(exchange, value[i])
    elif k == 5:
      meta[min(int(w*len(meta)), len(meta) - 1)].trade(exchange, u[1], u[2])
    elif k == 7:
      latent.limit_order(exchange, w, u[1], u[2])
    elif k == 8:
      latent.cancel_one(exchange, w)
    else:
      name, method = STRATEGY_STEPS[step % len(STRATEGY_STEPS)]
      step += 1
      trader = strategies[name]
      state = (trader.K, trader.pnl, trader.nb_shares)
      mid_before, spread_before = order_book.get_mid_price(), order_book.get_spread()
      val, quote, balance_sheet = getattr(trader, method)(order_book)
      if val == True:
        settle(order_book, trader, quote, state, mid_before, spread_before, None, exchange)
    if hook is not None:
      hook(i, exchange)
    mid = order_book.get_mid_price()
    mids[i] = np.nan if mid is None else float(mid)/cfg.tick
    if latent is not None:
      latent.update(mids[i])
    if active and cfg.record:
      values.append((order_book.time, value[i]))

  res = EcologyResult()
  res.cfg, res.exchange, res.order_book = cfg, exchange, order_book
  res.mids = mids # mid-price in ticks after every event
  res.mm, res.informed, res.meta, res.strategies, res.latent = mm, informed, meta, strategies, latent
  if not cfg.record:
    return res
  res.strategy_start = strategy_start
  res.start_time = start_time
  res.snapshots = recorder.frame()
  res.snapshots[['mid', 'spread']] = res.snapshots[['mid', 'spread']]/cfg.tick # in ticks from now on
  for c in [c for c in res.snapshots.columns if c.startswith(('bid_p', 'ask_p'))]:
    res.snapshots[c] = res.snapshots[c]/cfg.tick
  res.fundamental = pd.DataFrame(values, columns=['time', 'value'])
  fills = pd.DataFrame(exchange.fills, columns=['time', 'price', 'quantity', 'sign', 'aggressor', 'passive'])
  fills = fills[fills['time'] > start_time].reset_index(drop=True)
  fills['price'] = fills['price'].astype(float)/cfg.tick
  fills['quantity'] = fills['quantity'].astype(float)
  fills['aggressor_kind'] = fills['aggressor'].map(exchange.kinds)
  fills['passive_kind'] = fills['passive'].map(exchange.kinds)
  res.fills = fills
  orders = order_frame(fills[['time', 'price', 'quantity', 'sign']], res.snapshots)
  orders['kind'] = orders['time'].map(fills.groupby('time')['aggressor_kind'].first())
  res.orders = orders
  return res

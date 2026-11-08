# Note that here we consider a change of the Capital from the moment the order is sent and not at the moment
# the trade took place. It is a simplification of course, but it can be justified by the fact that, for the sake of
# simplification, no trader will be able to change or delete his order, but note that every trader could modify or even
# delete his order, our OrderBook() Class can handle this case without problem while keeping the priority.
# The simulation loop then compares this optimistic update with what the order book really executed
# (see settle() in simulation.py) and corrects the capital, the pnl and the number of shares.
# Convention used everywhere: 'bid' = BUY order, 'ask' = SELL order (the side of the order itself, as in the order book).
import numpy as np
from decimal import Decimal, ROUND_DOWN
from orderbook import to_decimal

QTY_STEP = Decimal('0.0001') # smallest tradable quantity, avoids dust orders

def quantize_qty(x):
  return to_decimal(x).quantize(QTY_STEP, rounding=ROUND_DOWN)

def affordable_market_buy(order_book, budget):
  '''Largest quantity a market buy can sweep from the ask side without spending more than the budget'''
  remaining = to_decimal(budget)
  quantity = Decimal(0)
  for price, order_list in order_book.asks.price_map.items():
    take = min(order_list.volume, remaining/price)
    quantity += take
    remaining -= take*price
    if take < order_list.volume:
      break
  return quantize_qty(quantity)

class Trader:
  '''Helpers shared by all the strategies'''
  def gross_return(self):
    return self.K/self.K_initial if self.K_initial else 0

  def _skip(self):
    # no order is sent this time: we only record the state of the trader
    quote = {'type' : np.nan,
             'side' : np.nan,
             'quantity' : np.nan,
             'price' : np.nan,
             'trade_id' : self.trade_id}
    self.current_return = self.gross_return()
    record = {'Current Capital' : self.K, 'Cumulative Pnl' : self.pnl,'Gross Return' : self.current_return, 'nb_shares' : self.nb_shares,'order_sent' : quote}
    self.trader_balance_sheet.append(record)
    self.val=False
    return self.val, quote, self.trader_balance_sheet


def mvt_avg(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def generate_lim_orders(nb_orders, price_range_bid, price_range_ask, qty_range_ask, qty_range_bid):
  '''Draws nb_orders random limit orders (a Bid or an Ask with equal probability), uniform in price and in quantity'''
  limit_orders = []
  trade_id_list = []
  for i in range(nb_orders):
    order_type = 'limit'
    alea = np.random.randint(0,2) # coin flip: 0 -> Ask, 1 -> Bid
    if alea == 0:
      side = 'ask'
      quantity = np.random.randint(qty_range_ask[0], qty_range_ask[1]+1)
      price = np.random.randint(price_range_ask[0], price_range_ask[1]+1)
      trade_id = 1000*np.random.randint(1,10) + 100*np.random.randint(1,10) + 10*np.random.randint(1,10) + np.random.randint(1,10)
      quote = {'type' : order_type,
               'side' : side,
               'quantity' : quantity,
               'price' : price,
               'trade_id' : trade_id}
      limit_orders.append(quote)
    else:
      side = 'bid'
      quantity = np.random.randint(qty_range_bid[0], qty_range_bid[1]+1)
      price = np.random.randint(price_range_bid[0], price_range_bid[1]+1)
      trade_id = 100*np.random.randint(1,10) + 10*np.random.randint(1,10) + np.random.randint(1,10)
      quote = {'type' : order_type,
               'side' : side,
               'quantity' : quantity,
               'price' : price,
               'trade_id' : trade_id}
      limit_orders.append(quote)
  return limit_orders



class Trader_quantity(Trader):

  def __init__(self, K, pnl=0, nb_shares=0):
    self.K = K
    self.K_initial = self.K
    self.nb_shares = nb_shares
    self.pnl = pnl
    self.trade_id = 1000*np.random.randint(1,10) + 100*np.random.randint(1,10) + 10*np.random.randint(1,10) + np.random.randint(1,10)
    self.trader_balance_sheet = []
    quote = {'type' : np.nan,
               'side' : np.nan,
               'quantity' : np.nan,
               'price' : np.nan,
               'trade_id' : self.trade_id}
    self.current_return = self.gross_return()
    record = {'Current Capital' : self.K, 'Cumulative Pnl' : self.pnl,'Gross Return' : self.current_return, 'nb_shares' : self.nb_shares, 'order_sent' : quote}
    self.val=False

  def trade_first(self, order_book):
    best_ask = order_book.get_best_ask()
    best_bid = order_book.get_best_bid()
    if best_ask is None or best_bid is None: # one side of the book is empty: nothing to do
      return self._skip()
    qty_best_ask = order_book.get_volume_at_price('ask', best_ask)
    qty_best_bid = order_book.get_volume_at_price('bid', best_bid)
    vol_possible_to_buy = quantize_qty(self.K/best_ask)
    vol_possible_to_sell = self.nb_shares
    if self.K>0 and vol_possible_to_buy>0 and vol_possible_to_buy<qty_best_ask:
      self.order_type = 'limit'
      self.side = 'bid' # buy at the best ask: a marketable limit order
      self.quantity = vol_possible_to_buy
      self.price = best_ask
      self.last_price = best_ask
      quote = {'type' : self.order_type,
               'side' : self.side,
               'quantity' : self.quantity,
               'price' : self.price,
               'trade_id' : self.trade_id}
      self.K -= vol_possible_to_buy*best_ask
      self.pnl -= vol_possible_to_buy*best_ask
      self.nb_shares += self.quantity
      self.current_return = self.gross_return()
      record = {'Current Capital' : self.K, 'Cumulative Pnl' : self.pnl,'Gross Return' : self.current_return, 'nb_shares' : self.nb_shares, 'order_sent' : quote}
      self.trader_balance_sheet.append(record)
      self.val=True
    elif self.nb_shares<qty_best_bid and self.nb_shares>0:
        self.order_type = 'limit'
        self.side = 'ask' # sell at the best bid: a marketable limit order
        self.quantity = max(self.nb_shares-1,0)
        if self.quantity<=0:
          quote = {'type' : np.nan,
                  'side' : np.nan,
                  'quantity' : np.nan,
                  'price' : np.nan,
                  'trade_id' : self.trade_id}
          self.current_return = self.gross_return()
          record = {'Current Capital' : self.K, 'Cumulative Pnl' : self.pnl,'Gross Return' : self.current_return, 'nb_shares' : self.nb_shares,'order_sent' : quote}
          self.trader_balance_sheet.append(record)
          self.val=False
        else:
          self.price = best_bid
          self.last_price = self.price
          quote = {'type' : self.order_type,
                  'side' : self.side,
                  'quantity' : self.quantity,
                  'price' : self.price,
                  'trade_id' : self.trade_id}
          self.K += self.quantity*best_bid
          self.pnl += self.quantity*best_bid
          self.nb_shares -= self.quantity
          self.current_return = self.gross_return()
          record = {'Current Capital' : self.K, 'Cumulative Pnl' : self.pnl,'Gross Return' : self.current_return, 'nb_shares' : self.nb_shares,'order_sent' : quote}
          self.trader_balance_sheet.append(record)
          self.val=True
    else:
      quote = {'type' : np.nan,
                'side' : np.nan,
                'quantity' : np.nan,
                'price' : np.nan,
                'trade_id' : self.trade_id}
      self.current_return = self.gross_return()
      record = {'Current Capital' : self.K, 'Cumulative Pnl' : self.pnl,'Gross Return' : self.current_return, 'order_sent' : quote}
      self.trader_balance_sheet.append(record)
      self.val=False
    return self.val, quote, self.trader_balance_sheet

  def tradeSecond(self, order_book):
    if self.nb_shares>0 and self.val==True:
      best_bid = order_book.get_best_bid()
      if best_bid is None:
        return self._skip()
      if best_bid>self.last_price:
        self.order_type = 'limit'
        self.side = 'ask'
        self.quantity = max(self.nb_shares-1,0)
        if self.quantity<=0:
          quote = {'type' : np.nan,
                  'side' : np.nan,
                  'quantity' : np.nan,
                  'price' : np.nan,
                  'trade_id' : self.trade_id}
          self.current_return = self.gross_return()
          record = {'Current Capital' : self.K, 'Cumulative Pnl' : self.pnl,'Gross Return' : self.current_return, 'nb_shares' : self.nb_shares,'order_sent' : quote}
          self.trader_balance_sheet.append(record)
          self.val=False
        else:
          self.price = order_book.get_best_bid()
          quote = {'type' : self.order_type,
                  'side' : self.side,
                  'quantity' : self.quantity,
                  'price' : self.price,
                  'trade_id' : self.trade_id}
          self.K += self.quantity*best_bid
          self.pnl += self.quantity*best_bid
          self.nb_shares -= self.quantity
          self.current_return = self.gross_return()
          record = {'Current Capital' : self.K, 'Cumulative Pnl' : self.pnl,'Gross Return' : self.current_return, 'nb_shares' : self.nb_shares,'order_sent' : quote}
          self.trader_balance_sheet.append(record)
          self.val=True
      else:
        quote = {'type' : np.nan,
                'side' : np.nan,
                'quantity' : np.nan,
                'price' : np.nan,
                'trade_id' : self.trade_id}
        self.current_return = self.gross_return()
        record = {'Current Capital' : self.K, 'Cumulative Pnl' : self.pnl,'Gross Return' : self.current_return, 'order_sent' : quote}
        self.trader_balance_sheet.append(record)
        self.val=False
    else:
      quote = {'type' : np.nan,
                'side' : np.nan,
                'quantity' : np.nan,
                'price' : np.nan,
                'trade_id' : self.trade_id}
      self.current_return = self.gross_return()
      record = {'Current Capital' : self.K, 'Cumulative Pnl' : self.pnl,'Gross Return' : self.current_return, 'nb_shares' : self.nb_shares,'order_sent' : quote}
      self.trader_balance_sheet.append(record)
      self.val=False
    return self.val, quote, self.trader_balance_sheet

class Trader_Buy_and_sell(Trader):

  def __init__(self, order_book):
    best_ask = order_book.get_best_ask()
    if best_ask is None:
      raise ValueError('Trader_Buy_and_sell needs a non-empty ask side to buy from')
    qty_best_ask = order_book.get_volume_at_price('ask', best_ask)
    self.K = qty_best_ask*best_ask
    self.K_initial = self.K
    self.pnl = 0
    self.trade_id = 1000*np.random.randint(1,10) + 100*np.random.randint(1,10) + 10*np.random.randint(1,10) + np.random.randint(1,10)
    self.trader_balance_sheet = []
    self.last_price = best_ask
    self.pnl -= self.K
    self.K -= self.K
    self.nb_shares = qty_best_ask
    type_order = 'limit'
    self.side = 'bid' # he starts by buying the whole best ask queue
    quote = {  'type' : type_order,
               'side' : self.side,
               'quantity' : qty_best_ask,
               'price' : best_ask,
               'trade_id' : self.trade_id}
    self.initial_quote = quote # the simulation sends it to the order book
    self.current_return = self.gross_return()
    record = {'Current Capital' : self.K, 'Cumulative Pnl' : self.pnl,'Gross Return' : self.current_return, 'nb_shares' : self.nb_shares,'order_sent' : quote}
    self.trader_balance_sheet.append(record)
    self.val=True

  def trade(self, order_book):
    if self.nb_shares > 0:
      best_bid = order_book.get_best_bid()
      if best_bid is None:
        return self._skip()
      qty_best_bid = max(self.nb_shares,0)
      if qty_best_bid<=0:
        quote = {'type' : np.nan,
                'side' : np.nan,
                'quantity' : np.nan,
                'price' : np.nan,
                'trade_id' : self.trade_id}
        self.current_return = self.gross_return()
        record = {'Current Capital' : self.K, 'Cumulative Pnl' : self.pnl,'Gross Return' : self.current_return, 'nb_shares' : self.nb_shares,'order_sent' : quote}
        self.trader_balance_sheet.append(record)
        self.val=False
      else:
        if best_bid>self.last_price:
          type_order = 'limit'
          self.side = 'ask' # he sells everything he holds at the best bid
          self.K += best_bid*qty_best_bid
          self.pnl += best_bid*qty_best_bid
          self.nb_shares -= qty_best_bid
          quote = {  'type' : type_order,
                'side' : self.side,
                'quantity' : qty_best_bid,
                'price' : best_bid,
                'trade_id' : self.trade_id}
          self.current_return = self.gross_return()
          record = {'Current Capital' : self.K, 'Cumulative Pnl' : self.pnl,'Gross Return' : self.current_return, 'nb_shares' : self.nb_shares,'order_sent' : quote}
          self.trader_balance_sheet.append(record)
          self.val=True
        else:
          quote = {'type' : np.nan,
                  'side' : np.nan,
                  'quantity' : np.nan,
                  'price' : np.nan,
                  'trade_id' : self.trade_id}
          self.current_return = self.gross_return()
          record = {'Current Capital' : self.K, 'Cumulative Pnl' : self.pnl,'Gross Return' : self.current_return, 'nb_shares' : self.nb_shares,'order_sent' : quote}
          self.trader_balance_sheet.append(record)
          self.val=False
    else:
      quote = {'type' : np.nan,
                'side' : np.nan,
                'quantity' : np.nan,
                'price' : np.nan,
                'trade_id' : self.trade_id}
      self.current_return = self.gross_return()
      record = {'Current Capital' : self.K, 'Cumulative Pnl' : self.pnl,'Gross Return' : self.current_return, 'nb_shares' : self.nb_shares,'order_sent' : quote}
      self.trader_balance_sheet.append(record)
      self.val=False
    return self.val, quote, self.trader_balance_sheet

class Trader_random(Trader):

  def __init__(self, K, pnl, nb_shares):
    self.K = K
    self.K_initial = self.K
    self.pnl = pnl
    self.nb_shares = nb_shares
    self.trade_id = 1000*np.random.randint(1,10) + 100*np.random.randint(1,10) + 10*np.random.randint(1,10) + np.random.randint(1,10)
    self.trader_balance_sheet = []
    self.current_return = self.gross_return()
    quote = {'type' : np.nan,
               'side' : np.nan,
               'quantity' : np.nan,
               'price' : np.nan,
               'trade_id' : self.trade_id}
    record = {'Current Capital' : self.K, 'Cumulative Pnl' : self.pnl,'Gross Return' : self.current_return, 'nb_shares' : self.nb_shares,'order_sent' : quote}
    self.trader_balance_sheet.append(record)
    self.val = False

  def trade(self, order_book):
    x = np.random.randint(0,2) # coin flip: 0 -> buy, 1 -> sell
    if order_book.get_best_ask() is None or order_book.get_best_bid() is None:
      return self._skip()
    if x == 0 and self.K >= 0:
      type_order = 'market'
      self.side = 'bid' # market buy
      quantity = max(affordable_market_buy(order_book, self.K),0)
      if quantity<=0:
        quote = {'type' : np.nan,
                'side' : np.nan,
                'quantity' : np.nan,
                'price' : np.nan,
                'trade_id' : self.trade_id}
        self.current_return = self.gross_return()
        record = {'Current Capital' : self.K, 'Cumulative Pnl' : self.pnl,'Gross Return' : self.current_return, 'nb_shares' : self.nb_shares,'order_sent' : quote}
        self.trader_balance_sheet.append(record)
        self.val=False
      else:
        quote = {'type': type_order,
                  'side': self.side,
                  'quantity': quantity,
                  'trade_id': self.trade_id}
        self.K -= quantity*order_book.get_best_ask()
        self.pnl -= quantity*order_book.get_best_ask()
        self.nb_shares += quantity
        self.current_return = self.gross_return()
        record = {'Current Capital' : self.K, 'Cumulative Pnl' : self.pnl,'Gross Return' : self.current_return, 'nb_shares' : self.nb_shares,'order_sent' : quote}
        self.trader_balance_sheet.append(record)
        self.val=True
    elif x == 1 and self.nb_shares > 0:
      type_order = 'market'
      self.side = 'ask' # market sell
      quantity = max(self.nb_shares-1,0)
      if quantity<=0:
        quote = {'type' : np.nan,
                'side' : np.nan,
                'quantity' : np.nan,
                'price' : np.nan,
                'trade_id' : self.trade_id}
        self.current_return = self.gross_return()
        record = {'Current Capital' : self.K, 'Cumulative Pnl' : self.pnl,'Gross Return' : self.current_return, 'nb_shares' : self.nb_shares,'order_sent' : quote}
        self.trader_balance_sheet.append(record)
        self.val=False
      else:
        quote = {'type': type_order,
                  'side': self.side,
                  'quantity': quantity,
                  'trade_id': self.trade_id}
        self.K += quantity*order_book.get_best_bid()
        self.pnl += quantity*order_book.get_best_bid()
        self.nb_shares -= quantity
        self.current_return = self.gross_return()
        record = {'Current Capital' : self.K, 'Cumulative Pnl' : self.pnl,'Gross Return' : self.current_return, 'nb_shares' : self.nb_shares,'order_sent' : quote}
        self.trader_balance_sheet.append(record)
        self.val=True
    else:
      quote = {'type' : np.nan,
                'side' : np.nan,
                'quantity' : np.nan,
                'price' : np.nan,
                'trade_id' : self.trade_id}
      self.current_return = self.gross_return()
      record = {'Current Capital' : self.K, 'Cumulative Pnl' : self.pnl,'Gross Return' : self.current_return, 'nb_shares' : self.nb_shares,'order_sent' : quote}
      self.trader_balance_sheet.append(record)
      self.val=False
    return self.val, quote, self.trader_balance_sheet

class Trader_bullish(Trader):
  #This trader thinks he will be able to sell very high so he buys with a market order and then sends a limit sell order at the worst of the Bids
  def __init__(self, K, pnl, nb_shares):
    self.K = K
    self.K_initial = self.K
    self.nb_shares = nb_shares
    self.pnl = pnl
    self.trade_id = 1000*np.random.randint(1,10) + 100*np.random.randint(1,10) + 10*np.random.randint(1,10) + np.random.randint(1,10)
    self.trader_balance_sheet = []
    self.current_return = self.gross_return()
    quote = {'type' : np.nan,
               'side' : np.nan,
               'quantity' : np.nan,
               'price' : np.nan,
               'trade_id' : self.trade_id}
    record = {'Current Capital' : self.K, 'Cumulative Pnl' : self.pnl,'Gross Return' : self.current_return, 'nb_shares' : self.nb_shares,'order_sent' : quote}
    self.trader_balance_sheet.append(record)
    self.val=False

  def tradeFirst(self, order_book):
     if order_book.get_best_ask() is None:
        return self._skip()
     type_order = 'market'
     self.side = 'bid' # market buy
     quantity = max(affordable_market_buy(order_book, self.K),0)
     if quantity<=0:
        quote = {'type' : np.nan,
                'side' : np.nan,
                'quantity' : np.nan,
                'price' : np.nan,
                'trade_id' : self.trade_id}
        self.current_return = self.gross_return()
        record = {'Current Capital' : self.K, 'Cumulative Pnl' : self.pnl,'Gross Return' : self.current_return, 'nb_shares' : self.nb_shares,'order_sent' : quote}
        self.trader_balance_sheet.append(record)
        self.val=False
     else:
        quote = {'type': type_order,
                  'side': self.side,
                  'quantity': quantity,
                  'trade_id': self.trade_id}
        self.K -= quantity*order_book.get_best_ask()
        self.pnl -= quantity*order_book.get_best_ask()
        self.nb_shares += quantity
        self.current_return = self.gross_return()
        record = {'Current Capital' : self.K, 'Cumulative Pnl' : self.pnl,'Gross Return' : self.current_return, 'nb_shares' : self.nb_shares,'order_sent' : quote}
        self.trader_balance_sheet.append(record)
        self.val=True
     return self.val, quote, self.trader_balance_sheet

  def SecondTrade(self, order_book):
    if self.nb_shares>0 and order_book.get_worst_bid() is not None:
      worst_bid = order_book.get_worst_bid()
      type_order = 'limit'
      self.side = 'ask' # limit sell at the worst bid
      quantity = max(self.nb_shares-1,0)
      if quantity<=0:
        quote = {'type' : np.nan,
                'side' : np.nan,
                'quantity' : np.nan,
                'price' : np.nan,
                'trade_id' : self.trade_id}
        self.current_return = self.gross_return()
        record = {'Current Capital' : self.K, 'Cumulative Pnl' : self.pnl,'Gross Return' : self.current_return, 'nb_shares' : self.nb_shares,'order_sent' : quote}
        self.trader_balance_sheet.append(record)
        self.val=False
      else:
        quote = {'type' : type_order,
                'side' : self.side,
                'quantity' : quantity,
                'price' : worst_bid,
                'trade_id' : self.trade_id}
        self.K += quantity*worst_bid
        self.pnl += quantity*worst_bid
        self.current_return = self.gross_return()
        self.nb_shares -= quantity
        record = {'Current Capital' : self.K, 'Cumulative Pnl' : self.pnl,'Gross Return' : self.current_return, 'nb_shares' : self.nb_shares,'order_sent' : quote}
        self.trader_balance_sheet.append(record)
        self.val=True
    else:
      quote = {'type' : np.nan,
                'side' : np.nan,
                'quantity' : np.nan,
                'price' : np.nan,
                'trade_id' : self.trade_id}
      self.current_return = self.gross_return()
      record = {'Current Capital' : self.K, 'Cumulative Pnl' : self.pnl,'Gross Return' : self.current_return, 'nb_shares' : self.nb_shares,'order_sent' : quote}
      self.trader_balance_sheet.append(record)
      self.val=False
    return self.val, quote, self.trader_balance_sheet

class Trader_bearish(Trader):
  #This one is the opposite of the previous one: he already holds shares and he thinks he will be able to buy them back very cheaply
  def __init__(self, K, pnl, nb_shares):
    self.K = K
    self.K_initial = self.K
    self.nb_shares = nb_shares
    self.pnl = pnl
    self.trade_id = 1000*np.random.randint(1,10) + 100*np.random.randint(1,10) + 10*np.random.randint(1,10) + np.random.randint(1,10)
    self.trader_balance_sheet = []
    self.current_return = self.gross_return()
    quote = {'type' : np.nan,
               'side' : np.nan,
               'quantity' : np.nan,
               'price' : np.nan,
               'trade_id' : self.trade_id}
    record = {'Current Capital' : self.K, 'Cumulative Pnl' : self.pnl,'Gross Return' : self.current_return, 'nb_shares' : self.nb_shares,'order_sent' : quote}
    self.trader_balance_sheet.append(record)
    self.val = False

  def tradeFirst(self, order_book):
    if self.nb_shares>0 and order_book.get_best_bid() is not None:
      type_order = 'market'
      self.side = 'ask' # market sell
      quantity = max(self.nb_shares-1,0)
      if quantity<=0:
        quote = {'type' : np.nan,
                'side' : np.nan,
                'quantity' : np.nan,
                'price' : np.nan,
                'trade_id' : self.trade_id}
        self.current_return = self.gross_return()
        record = {'Current Capital' : self.K, 'Cumulative Pnl' : self.pnl,'Gross Return' : self.current_return, 'nb_shares' : self.nb_shares,'order_sent' : quote}
        self.trader_balance_sheet.append(record)
        self.val=False
      else:
        quote = {'type': type_order,
                  'side': self.side,
                  'quantity': quantity,
                  'trade_id': self.trade_id}
        self.K += quantity*order_book.get_best_bid()
        self.pnl += quantity*order_book.get_best_bid()
        self.nb_shares -= quantity
        self.current_return = self.gross_return()
        record = {'Current Capital' : self.K, 'Cumulative Pnl' : self.pnl,'Gross Return' : self.current_return, 'nb_shares' : self.nb_shares,'order_sent' : quote}
        self.trader_balance_sheet.append(record)
        self.val=True
    else:
      quote = {'type' : np.nan,
                'side' : np.nan,
                'quantity' : np.nan,
                'price' : np.nan,
                'trade_id' : self.trade_id}
      self.current_return = self.gross_return()
      record = {'Current Capital' : self.K, 'Cumulative Pnl' : self.pnl,'Gross Return' : self.current_return, 'nb_shares' : self.nb_shares,'order_sent' : quote}
      self.trader_balance_sheet.append(record)
      self.val=False
    return self.val, quote, self.trader_balance_sheet

  def SecondTrade(self, order_book):
    if self.nb_shares<=1 and order_book.get_worst_ask() is not None: # tradeFirst always keeps 1 share
      worst_ask = order_book.get_worst_ask()
      type_order = 'limit'
      self.side = 'bid' # limit buy at the worst ask
      quantity = max(quantize_qty(self.K/order_book.get_worst_ask()-1),0)
      if quantity<=0:
        quote = {'type' : np.nan,
                'side' : np.nan,
                'quantity' : np.nan,
                'price' : np.nan,
                'trade_id' : self.trade_id}
        self.current_return = self.gross_return()
        record = {'Current Capital' : self.K, 'Cumulative Pnl' : self.pnl,'Gross Return' : self.current_return, 'nb_shares' : self.nb_shares,'order_sent' : quote}
        self.trader_balance_sheet.append(record)
        self.val=False
      else:
        quote = {'type' : type_order,
                'side' : self.side,
                'quantity' : quantity,
                'price' : worst_ask,
                'trade_id' : self.trade_id}
        self.K -= quantity*worst_ask
        self.pnl -= quantity*worst_ask
        self.current_return = self.gross_return()
        self.nb_shares += quantity
        record = {'Current Capital' : self.K, 'Cumulative Pnl' : self.pnl,'Gross Return' : self.current_return, 'nb_shares' : self.nb_shares,'order_sent' : quote}
        self.trader_balance_sheet.append(record)
        self.val=True
    else:
      quote = {'type' : np.nan,
                'side' : np.nan,
                'quantity' : np.nan,
                'price' : np.nan,
                'trade_id' : self.trade_id}
      self.current_return = self.gross_return()
      record = {'Current Capital' : self.K, 'Cumulative Pnl' : self.pnl,'Gross Return' : self.current_return, 'nb_shares' : self.nb_shares,'order_sent' : quote}
      self.trader_balance_sheet.append(record)
      self.val=False
    return self.val, quote, self.trader_balance_sheet

  def ThirdTrade(self, order_book):
    if self.nb_shares>0 and order_book.get_worst_bid() is not None:
      worst_bid = order_book.get_worst_bid()
      type_order = 'limit'
      self.side = 'ask' # limit sell at the worst bid
      quantity = max(self.nb_shares,0)
      if quantity<=0:
        quote = {'type' : np.nan,
                'side' : np.nan,
                'quantity' : np.nan,
                'price' : np.nan,
                'trade_id' : self.trade_id}
        self.current_return = self.gross_return()
        record = {'Current Capital' : self.K, 'Cumulative Pnl' : self.pnl,'Gross Return' : self.current_return, 'nb_shares' : self.nb_shares,'order_sent' : quote}
        self.trader_balance_sheet.append(record)
        self.val=False
      else:
        quote = {'type' : type_order,
                'side' : self.side,
                'quantity' : quantity,
                'price' : worst_bid,
                'trade_id' : self.trade_id}
        self.K += quantity*worst_bid
        self.pnl += quantity*worst_bid
        self.current_return = self.gross_return()
        self.nb_shares -= quantity
        record = {'Current Capital' : self.K, 'Cumulative Pnl' : self.pnl,'Gross Return' : self.current_return, 'nb_shares' : self.nb_shares,'order_sent' : quote}
        self.trader_balance_sheet.append(record)
        self.val=True
    else:
      quote = {'type' : np.nan,
                'side' : np.nan,
                'quantity' : np.nan,
                'price' : np.nan,
                'trade_id' : self.trade_id}
      self.current_return = self.gross_return()
      record = {'Current Capital' : self.K, 'Cumulative Pnl' : self.pnl,'Gross Return' : self.current_return, 'nb_shares' : self.nb_shares,'order_sent' : quote}
      self.trader_balance_sheet.append(record)
      self.val=False
    return self.val, quote, self.trader_balance_sheet

class Trader_MA(Trader):
  #This trader takes the n-period Moving Average of the trade prices and then watches for the moment when the price of the last trade goes below this MA(n): he will then sell,
  #he will send an adapted limit sell order so as to lose nothing and thus look for the best execution, and conversely if the price of the last trade goes above this MA(n) he will buy
  def __init__(self, K, pnl, nb_shares, n):
    self.val = False
    self.K = K
    self.K_initial = self.K
    self.nb_shares = nb_shares
    self.pnl = pnl
    self.n = n
    self.trade_id = 1000*np.random.randint(1,10) + 100*np.random.randint(1,10) + 10*np.random.randint(1,10) + np.random.randint(1,10)
    self.trader_balance_sheet = []
    self.current_return = self.gross_return()
    quote = {'type' : np.nan,
               'side' : np.nan,
               'quantity' : np.nan,
               'price' : np.nan,
               'trade_id' : self.trade_id}
    record = {'Current Capital' : self.K, 'Cumulative Pnl' : self.pnl,'Gross Return' : self.current_return, 'nb_shares' : self.nb_shares,'order_sent' : quote}
    self.trader_balance_sheet.append(record)

  def firstTrade(self, order_book):
    mva_order = self.n
    bids, asks, trades_done = order_book.get_all()
    prices_trades = [x['price'] for x in trades_done]
    if len(prices_trades) < mva_order or order_book.get_best_ask() is None or order_book.get_best_bid() is None:
      return self._skip() # not enough trades to compute the MA, or one side of the book is empty
    mva_avg = mvt_avg(prices_trades, mva_order)
    if prices_trades[-1] > mva_avg[-1] and self.K > 0:
      type_order = 'limit'
      self.side = 'bid' # limit buy at the best ask
      quantity = max(min(quantize_qty(self.K/order_book.get_best_ask()-1), order_book.get_volume_at_price('ask',order_book.get_best_ask())),0)
      if quantity<=0:
        quote = {'type' : np.nan,
                'side' : np.nan,
                'quantity' : np.nan,
                'price' : np.nan,
                'trade_id' : self.trade_id}
        self.current_return = self.gross_return()
        record = {'Current Capital' : self.K, 'Cumulative Pnl' : self.pnl,'Gross Return' : self.current_return, 'nb_shares' : self.nb_shares,'order_sent' : quote}
        self.trader_balance_sheet.append(record)
        self.val=False
      else:
        price = order_book.get_best_ask()
        quote = {'type' : type_order,
                'side' : self.side,
                'quantity' : quantity,
                'price' : price,
                'trade_id' : self.trade_id}
        self.K -= price*quantity
        self.pnl -= price*quantity
        self.nb_shares += quantity
        self.current_return = self.gross_return()
        record = {'Current Capital' : self.K, 'Cumulative Pnl' : self.pnl,'Gross Return' : self.current_return, 'nb_shares' : self.nb_shares,'order_sent' : quote}
        self.trader_balance_sheet.append(record)
        self.val=True
    elif prices_trades[-1] < mva_avg[-1] and self.nb_shares > 0:
      type_order = 'limit'
      self.side = 'ask' # limit sell at the best bid
      quantity = max(self.nb_shares-1,0)
      if quantity<=0:
        quote = {'type' : np.nan,
                'side' : np.nan,
                'quantity' : np.nan,
                'price' : np.nan,
                'trade_id' : self.trade_id}
        self.current_return = self.gross_return()
        record = {'Current Capital' : self.K, 'Cumulative Pnl' : self.pnl,'Gross Return' : self.current_return, 'nb_shares' : self.nb_shares,'order_sent' : quote}
        self.trader_balance_sheet.append(record)
        self.val=False
      else:
        price = order_book.get_best_bid()
        quote = {'type' : type_order,
                'side' : self.side,
                'quantity' : quantity,
                'price' : price,
                'trade_id' : self.trade_id}
        self.K += price*quantity
        self.pnl += price*quantity
        self.nb_shares -= quantity
        self.current_return = self.gross_return()
        record = {'Current Capital' : self.K, 'Cumulative Pnl' : self.pnl,'Gross Return' : self.current_return, 'nb_shares' : self.nb_shares,'order_sent' : quote}
        self.trader_balance_sheet.append(record)
        self.val=True
    else:
      quote = {'type' : np.nan,
                'side' : np.nan,
                'quantity' : np.nan,
                'price' : np.nan,
                'trade_id' : self.trade_id}
      self.current_return = self.gross_return()
      record = {'Current Capital' : self.K, 'Cumulative Pnl' : self.pnl,'Gross Return' : self.current_return, 'nb_shares' : self.nb_shares,'order_sent' : quote}
      self.trader_balance_sheet.append(record)
      self.val=False
    return self.val, quote, self.trader_balance_sheet


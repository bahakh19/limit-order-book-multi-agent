# Now we are going to drop quite a few of the simplifying assumptions we had made until now.
# We are really going to try to build an order book system that can process not only
# market orders but also limit orders.
# A limit bid order works as follows: we set a limit price, and when the best Ask (i.e. the min)
# reaches this price, our order is converted into a market order. A limit ask order works similarly:
# we set a limit price, and when the best Bid (i.e. the max) reaches this price, our order is
# converted into a market order. While coding our order book we asked ourselves, for a same price
# when we have several orders, which one should be served first, which raises the question of the
# priority rule. We chose a FIFO approach where the first to arrive is also the first to leave,
# which seems more realistic, and for reasons of simplification and of simulating reality
# we included a notion of time: each order is given a timestamp.
from decimal import Decimal
from collections import deque
from io import StringIO
from numbers import Integral, Real
from sortedcontainers import SortedDict


def to_decimal(value):
    '''
    Converts a price or a quantity to Decimal. Floats go through their shortest string
    representation (Decimal(0.1) would otherwise keep the binary noise of the float and
    two "equal" prices would end up on two different price levels).
    '''
    if isinstance(value, Decimal):
        return value
    if isinstance(value, Integral):
        return Decimal(int(value))
    if isinstance(value, Real):
        return Decimal(str(float(value)))
    return Decimal(value)


class Order(object):
    '''
    The Order class is the fundamental element of the exchange. Every bid/ask is an Order.
    Orders are doubly linked within a same bid/ask price and we define the attributes that will
    help us update the orders tied to a same price: next_order, prev_order
    '''
    def __init__(self, quote, order_list):
        self.timestamp = int(quote['timestamp']) # an integer representing the creation time of the order
        self.quantity = to_decimal(quote['quantity']) # The quantity specified in the order, it can be non-integer
        self.price = to_decimal(quote['price']) # The price specified in the order
        self.order_id = int(quote['order_id'])
        self.trade_id = quote['trade_id']
        # a doubly linked list to make it easy to rearrange the orders sharing a same price
        self.next_order = None
        self.prev_order = None
        self.order_list = order_list

    def update_quantity(self, new_quantity, new_timestamp):
        new_quantity = to_decimal(new_quantity)
        if new_quantity > self.quantity and self.order_list.tail_order != self:
            # if the order is not already the last one of the OrderList and its quantity increases, it loses its time priority
            self.order_list.move_to_tail(self) # Put it at the tail of the list
        self.order_list.volume -= (self.quantity - new_quantity) # Update the volume of the OrderList
        self.timestamp = new_timestamp
        self.quantity = new_quantity

    def __str__(self):
        return "quantity : {}, price : {}/ Trade_id : {} - Order timestamp : {} seconds".format(self.quantity, self.price,
                                      self.trade_id, self.timestamp)


class OrderList(object):
    '''
    Since a same price can have several orders tied to it, these orders are held in an OrderList,
    which is a doubly linked list. When there is a match, i.e. two or more orders at the same price,
    this list is arranged by time and of course the orders at the head of the list have priority over
    the orders at the end of the queue.
    '''

    def __init__(self):
        self.head_order = None # First Order in the list
        self.tail_order = None # Last order in the list
        self.length = 0 # Number of orders in the list
        self.volume = 0 # The sum of the quantities of the orders in the list, i.e. the Volume
        self.last = None # We use this element to iterate over the list

    def __len__(self):
        '''
        Special method returning the number of orders
        '''
        return self.length

    def __iter__(self):
        self.last = self.head_order
        return self

    def next(self):
        '''Takes the next order in the list.
        Sets self.last to the next order. If there is no next order
        in the list then an error is raised to stop the iteration.
        '''
        if self.last == None:
            raise StopIteration
        else:
            return_value = self.last
            self.last = self.last.next_order
            return return_value

    __next__ = next

    def get_head_order(self):
      """
      Returns the order that has priority
      """
      return self.head_order

    def append_order(self, order):
        """
        Adds an order to our list. If our list is empty, i.e. no order,
        then this new order is both the tail and the head and there is no order before or after it.

        """
        if len(self) == 0:
            order.next_order = None
            order.prev_order = None
            self.head_order = order
            self.tail_order = order
        else:
            order.prev_order = self.tail_order
            order.next_order = None
            self.tail_order.next_order = order
            self.tail_order = order
        self.length +=1
        self.volume += order.quantity

    def remove_order(self, order):
        self.volume -= order.quantity
        self.length -= 1

        # To remove an order from the OrderList. First we take the next_order / prev_order
        # of the order we are going to remove. Then we redo all the right re-associations. Finally
        # we remove the Order.
        next_order = order.next_order
        prev_order = order.prev_order
        if next_order != None and prev_order != None:
            next_order.prev_order = prev_order
            prev_order.next_order = next_order
        elif next_order != None:  # If there was no order before the order we remove
            next_order.prev_order = None
            self.head_order = next_order # The order following the one we remove from the OrderList becomes the head
        elif prev_order != None: # And if there was no order after the one we remove
            prev_order.next_order = None
            self.tail_order = prev_order # The prev_order becomes the last of the OrderList once the order is removed
        else: # It was the only order of the list: the list is now empty, no stale pointer must remain
            self.head_order = None
            self.tail_order = None
        order.next_order = None
        order.prev_order = None

    def move_to_tail(self, order):
        '''
        After updating the quantity of the order, we have to put it at the end of the OrderList queue
        '''
        if order.next_order is None: # already the last one
            return
        if order.prev_order != None:  # If this order is not the first in the OrderList
            order.prev_order.next_order = order.next_order # link prev_order and next_order, then move the order to the end of the queue
        else: # If the order is the first in the OrderList
            self.head_order = order.next_order # We have to make next_order the first one

        order.next_order.prev_order = order.prev_order
        order.prev_order = self.tail_order
        order.next_order = None
        # Move the Order to the last position.
        self.tail_order.next_order = order
        self.tail_order = order

    def __repr__(self):
        return 'OrderList(length={}, volume={})'.format(self.length, self.volume)

    def __str__(self):
        temp_file = StringIO()
        for order in self:
            temp_file.write("%s\n" % str(order))
        return temp_file.getvalue()


class OrderTree(object):
    '''We use a sorted dictionary (SortedDict, sortedcontainers) to keep our OrderLists in price order
    (it plays the role of a balanced search tree such as a red-black tree: ordered access to the best price in logarithmic time).
    The exchange uses the OrderTree to collect the bid/ask data (with one OrderTree per side, i.e Bid side and Ask side).
    This choice of data structure makes the detection of a match between orders simpler and faster.
    '''

    def __init__(self):
        self.price_map = SortedDict() # A dictionary containing {price : an OrderList object}
        self.prices = self.price_map.keys()# The keys of this dict are of course the prices
        self.order_map = {} # Dictionary containing {order_id : an Order object}
        self.volume = 0 # The total quantity of all the orders in a tree
        self.num_orders = 0 # Contains the number of orders in a tree
        self.depth = 0 # Number of distinct prices in a tree which gives us the depth of a market

    def __len__(self):
        #Returns the number of orders
        return len(self.order_map)

    def get_price_list(self, price):
        return self.price_map[price]

    def get_order(self, order_id):
        return self.order_map[order_id]

    def create_price(self, price):
        self.depth += 1 # Adds a price to the tree, i.e. a new level of depth for the market
        new_list = OrderList()
        self.price_map[price] = new_list

    def remove_price(self, price):
        self.depth -= 1 # We remove a price from the tree, i.e. one level of depth less for the market
        del self.price_map[price]

    def price_exists(self, price):
        #Used to check whether a price exists in our tree, it returns a boolean
        return price in self.price_map

    def order_exists(self, order):
        #Used to check whether an order exists in our tree, it returns a boolean
        return order in self.order_map

    def insert_order(self, quote):
        if self.order_exists(quote['order_id']):
            self.remove_order_by_id(quote['order_id'])
        self.num_orders += 1
        quote['price'] = to_decimal(quote['price'])
        if quote['price'] not in self.price_map:
            self.create_price(quote['price']) # if the price of the order is not in our tree, then we create a node in our tree
        order = Order(quote, self.price_map[quote['price']]) # We create a new Order
        self.price_map[order.price].append_order(order) # Adds the order to the OrderList in the Price Map
        self.order_map[order.order_id] = order
        self.volume += order.quantity

    def update_order(self, order_update):
        order = self.order_map[order_update['order_id']]
        order_update.setdefault('trade_id', order.trade_id) # the update does not have to repeat the owner of the order
        order_update['price'] = to_decimal(order_update['price'])
        order_update['quantity'] = to_decimal(order_update['quantity'])
        original_quantity = order.quantity
        if order_update['price'] != order.price:
            # If the price has changed then we remove the order and update the tree
            # (remove_order_by_id also removes the price level when it becomes empty and
            # keeps volume, depth and num_orders consistent, insert_order puts the order back at its new price).
            self.remove_order_by_id(order_update['order_id'])
            self.insert_order(order_update)
        else:
            # If the quantity has changed but the price stayed the same
            order.update_quantity(order_update['quantity'], order_update['timestamp'])
            self.volume += order.quantity - original_quantity

    def remove_order_by_id(self, order_id):
        self.num_orders -= 1
        order = self.order_map[order_id]
        self.volume -= order.quantity
        order.order_list.remove_order(order)
        if len(order.order_list) == 0:
            self.remove_price(order.price)
        del self.order_map[order_id]

    def max_price(self):
        if self.depth > 0:
            return self.prices[-1]
        else:
            return None

    def min_price(self):
        if self.depth > 0:
            return self.prices[0]
        else:
            return None

    def max_price_list(self):
        if self.depth > 0:
            return self.get_price_list(self.max_price())
        else:
            return None

    def min_price_list(self):
        if self.depth > 0:
            return self.get_price_list(self.min_price())
        else:
            return None

class OrderBook(object):
    def __init__(self, tick_size = 0.0001):
        self.tape = deque(maxlen=None) # Trades in chronological order: the last element is the most recent Trade that was executed
        self.bids = OrderTree()
        self.asks = OrderTree()
        self.last_tick = None
        self.last_timestamp = 0
        self.tick_size = tick_size
        self.tick = to_decimal(tick_size) # every limit price must be a multiple of the tick
        self.time = 0
        self.next_order_id = 0

    def update_time(self):
        self.time += 1

    def process_order(self, quote, from_data=False, verbose=False):
        order_type = quote['type']
        order_in_book = None
        if from_data:
            self.time = quote['timestamp']
        else:
            self.update_time()
            quote['timestamp'] = self.time
        quote['quantity'] = to_decimal(quote['quantity'])
        if quote['quantity'] <= 0:
            raise ValueError('process_order() was given a quantity <= 0')
        if not from_data:
            self.next_order_id += 1
        if order_type == 'market':
            trades = self.process_market_order(quote, verbose)
        elif order_type == 'limit':
            quote['price'] = to_decimal(quote['price'])
            self.check_tick(quote['price'])
            trades, order_in_book = self.process_limit_order(quote, from_data, verbose)
        else:
            raise ValueError("order_type given to process_order() is neither 'market' nor 'limit'")
        return trades, order_in_book

    def check_tick(self, price):
        # An exchange only accepts prices on its tick grid (chapter 3.1 of Trades, Quotes and Prices)
        if price <= 0 or price % self.tick != 0:
            raise ValueError('price {} is not a positive multiple of the tick size {}'.format(price, self.tick))

    def process_order_list(self, side, order_list, quantity_still_to_trade, quote, verbose):
        '''
        Takes as argument an OrderList (collection of orders at a given price) and the incoming order and matches
        the right trades given the quantity specified in the order.
        '''
        trades = []
        quantity_to_trade = quantity_still_to_trade
        while len(order_list) > 0 and quantity_to_trade > 0:
            head_order = order_list.get_head_order()
            traded_price = head_order.price
            counter_party = head_order.trade_id
            new_book_quantity = None
            if quantity_to_trade < head_order.quantity:
                traded_quantity = quantity_to_trade
                # Carry out the transaction
                new_book_quantity = head_order.quantity - quantity_to_trade
                head_order.update_quantity(new_book_quantity, head_order.timestamp)
                if side == 'bid':
                    self.bids.volume -= traded_quantity
                else:
                    self.asks.volume -= traded_quantity
                quantity_to_trade = 0
            elif quantity_to_trade == head_order.quantity:
                traded_quantity = quantity_to_trade
                if side == 'bid':
                    self.bids.remove_order_by_id(head_order.order_id)
                else:
                    self.asks.remove_order_by_id(head_order.order_id)
                quantity_to_trade = 0
            else: # If the quantity to process is larger than the order at the head of the queue
                traded_quantity = head_order.quantity
                if side == 'bid':
                    self.bids.remove_order_by_id(head_order.order_id)
                else:
                    self.asks.remove_order_by_id(head_order.order_id)
                quantity_to_trade -= traded_quantity
            if verbose:
                print(("TRADE: Time - {} seconds, Price - {}, Quantity - {}, TradeID - {}, Matching TradeID - {}".format(self.time, traded_price, traded_quantity, counter_party, quote['trade_id'])))

            transaction_record = {
                    'timestamp': self.time,
                    'price': traded_price,
                    'quantity': traded_quantity,
                    'time': self.time
                    }

            if side == 'bid':
                transaction_record['party1'] = [counter_party, 'bid', head_order.order_id, new_book_quantity]
                transaction_record['party2'] = [quote['trade_id'], 'ask', None, None]
            else:
                transaction_record['party1'] = [counter_party, 'ask', head_order.order_id, new_book_quantity]
                transaction_record['party2'] = [quote['trade_id'], 'bid', None, None]

            self.tape.append(transaction_record)
            trades.append(transaction_record)
        return quantity_to_trade, trades

    def process_market_order(self, quote, verbose):
        trades = []
        quantity_to_trade = quote['quantity']
        side = quote['side']
        if side == 'bid':
            while quantity_to_trade > 0 and self.asks:
                best_price_asks = self.asks.min_price_list()
                quantity_to_trade, new_trades = self.process_order_list('ask', best_price_asks, quantity_to_trade, quote, verbose)
                trades += new_trades
        elif side == 'ask':
            while quantity_to_trade > 0 and self.bids:
                best_price_bids = self.bids.max_price_list()
                quantity_to_trade, new_trades = self.process_order_list('bid', best_price_bids, quantity_to_trade, quote, verbose)
                trades += new_trades
        else:
            raise ValueError('process_market_order() received neither "bid" nor "ask"')
        return trades

    def process_limit_order(self, quote, from_data, verbose):
        order_in_book = None
        trades = []
        quantity_to_trade = quote['quantity']
        side = quote['side']
        price = quote['price']
        if side == 'bid':
            while (self.asks and price >= self.asks.min_price() and quantity_to_trade > 0):
                best_price_asks = self.asks.min_price_list()
                quantity_to_trade, new_trades = self.process_order_list('ask', best_price_asks, quantity_to_trade, quote, verbose)
                trades += new_trades
            if quantity_to_trade > 0:
                if not from_data:
                    quote['order_id'] = self.next_order_id
                quote['quantity'] = quantity_to_trade
                self.bids.insert_order(quote)
                order_in_book = quote
        elif side == 'ask':
            while (self.bids and price <= self.bids.max_price() and quantity_to_trade > 0):
                best_price_bids = self.bids.max_price_list()
                quantity_to_trade, new_trades = self.process_order_list('bid', best_price_bids, quantity_to_trade, quote, verbose)
                trades += new_trades
            if quantity_to_trade > 0:
                if not from_data:
                    quote['order_id'] = self.next_order_id
                quote['quantity'] = quantity_to_trade
                self.asks.insert_order(quote)
                order_in_book = quote
        else:
            raise ValueError('process_limit_order() received neither "bid" nor "ask"')
        return trades, order_in_book

    def cancel_order(self, side, order_id, time=None):
        if time:
            self.time = time
        else:
            self.update_time()
        if side == 'bid':
            if self.bids.order_exists(order_id):
                self.bids.remove_order_by_id(order_id)
        elif side == 'ask':
            if self.asks.order_exists(order_id):
                self.asks.remove_order_by_id(order_id)
        else:
            raise ValueError('cancel_order() received neither "bid" nor "ask"')

    def modify_order(self, order_id, order_update, time=None):
        if time:
            self.time = time
        else:
            self.update_time()
        side = order_update['side']
        self.check_tick(to_decimal(order_update['price']))
        order_update['order_id'] = order_id
        order_update['timestamp'] = self.time
        if side == 'bid':
            if self.bids.order_exists(order_update['order_id']):
                self.bids.update_order(order_update)
        elif side == 'ask':
            if self.asks.order_exists(order_update['order_id']):
                self.asks.update_order(order_update)
        else:
            raise ValueError('modify_order() received neither "bid" nor "ask"')

    def get_volume_at_price(self, side, price):
        price = to_decimal(price)
        if side == 'bid':
            volume = 0
            if self.bids.price_exists(price):
                volume = self.bids.get_price_list(price).volume
            return volume
        elif side == 'ask':
            volume = 0
            if self.asks.price_exists(price):
                volume = self.asks.get_price_list(price).volume
            return volume
        else:
            raise ValueError('get_volume_at_price() received neither "bid" nor "ask"')

    def get_best_bid(self):
        return self.bids.max_price()

    def get_worst_bid(self):
        return self.bids.min_price()

    def get_best_ask(self):
        return self.asks.min_price()

    def get_worst_ask(self):
        return self.asks.max_price()

    def get_mid_price(self):
        # Mid-price m = (best bid + best ask) / 2, None while one side of the book is empty
        if self.bids.depth == 0 or self.asks.depth == 0:
            return None
        return (self.get_best_bid() + self.get_best_ask()) / 2

    def get_spread(self):
        # Bid-ask spread s = best ask - best bid, None while one side of the book is empty
        if self.bids.depth == 0 or self.asks.depth == 0:
            return None
        return self.get_best_ask() - self.get_best_bid()

    def tape_dump(self, filename, filemode, tapemode):
        with open(filename, filemode) as dumpfile:
            for tapeitem in self.tape:
                dumpfile.write('Time: %s seconds, Price: %s, Quantity: %s\n' % (tapeitem['time'],
                                                                        tapeitem['price'],
                                                                        tapeitem['quantity']))
        if tapemode == 'wipe':
            self.tape.clear()

    def get_all(self):
        bid_orders = []
        ask_orders = []
        if self.bids != None and len(self.bids) > 0:
          for key, value in reversed(self.bids.price_map.items()):
                bid_orders.append(value)
        if self.asks != None and len(self.asks) > 0:
          for key, value in self.asks.price_map.items():
                ask_orders.append(value)
        trades_done = []
        if self.tape != None and len(self.tape) > 0:
          for entry in self.tape:
                dico = {
                        'quantity' : entry['quantity'],
                        'price' : entry['price'],
                        'timestamp' : entry['timestamp'],
                        'counterparty1_id' : entry['party1'],
                        'counterparty2_id' : entry['party2']
                    }
                trades_done.append(dico)
        return bid_orders, ask_orders, trades_done

    def __str__(self):
        tempfile = StringIO()
        tempfile.write("***Bids***\n")
        if self.bids != None and len(self.bids) > 0:
            for key, value in reversed(self.bids.price_map.items()):
                tempfile.write('%s' % value)
        tempfile.write("\n***Asks***\n")
        if self.asks != None and len(self.asks) > 0:
            for key, value in self.asks.price_map.items():
                tempfile.write('%s' % value)
        tempfile.write("\n***Trades***\n")
        if self.tape != None and len(self.tape) > 0:
            num = 0
            for entry in self.tape:
                if num < 10:
                    tempfile.write('quantity : ' + str(entry['quantity']) + " , Price : " + str(entry['price']) + " (Trade execution time : " + str(entry['timestamp']) + " seconds), Id_party1 : " + str(entry['party1'][0]) + "/ Id_party2 : " + str(entry['party2'][0]) + "\n")
                    num += 1
                else:
                    break
        tempfile.write("\n")
        return tempfile.getvalue()

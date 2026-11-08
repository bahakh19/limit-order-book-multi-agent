#We are going to run a test to make sure that our exchange works well
#and also to explain the rules of the latter
order_book = OrderBook()

# We create limit orders to give our market some depth

limit_orders = [{'type' : 'limit',
                   'side' : 'ask',
                    'quantity' : 5,
                    'price' : 101,
                    'trade_id' : 100},
                   {'type' : 'limit',
                    'side' : 'ask',
                    'quantity' : 5,
                    'price' : 103,
                    'trade_id' : 101},
                   {'type' : 'limit',
                    'side' : 'ask',
                    'quantity' : 5,
                    'price' : 101,
                    'trade_id' : 102},
                   {'type' : 'limit',
                    'side' : 'ask',
                    'quantity' : 5,
                    'price' : 101,
                    'trade_id' : 103},
                   {'type' : 'limit',
                    'side' : 'bid',
                    'quantity' : 5,
                    'price' : 99,
                    'trade_id' : 100},
                   {'type' : 'limit',
                    'side' : 'bid',
                    'quantity' : 5,
                    'price' : 98,
                    'trade_id' : 101},
                   {'type' : 'limit',
                    'side' : 'bid',
                    'quantity' : 5,
                    'price' : 99,
                    'trade_id' : 102},
                   {'type' : 'limit',
                    'side' : 'bid',
                    'quantity' : 5,
                    'price' : 97,
                    'trade_id' : 103},
                   ]

# We add the orders to our Orderbook
for order in limit_orders:
    trades, order_id = order_book.process_order(order, False, False)

# The current book can be displayed with a print
print(order_book)
# As we can see, at this point we have our order book with the Bids, the asks and the Trades
# There is no trade for the moment and this is on purpose in the way we specified the orders
# They are all limit orders that do not cross each other so the result is that no trade can take place
# Let us also see why we chose a sorted price structure: we have a double classification
# The bids are sorted first by decreasing price and then by the timestamp of the order within a same price in increasing order
# The asks are sorted first by increasing price and then by the timestamp of the orders within a same price in increasing order
# In other words the best Ask, the one that is going to be served first for any incoming order, is the Ask with the lowest price that arrived the earliest
# And the best Bid is the one that is going to be served first for any incoming Ask order, the Bid that has priority is the Bid with the highest price that arrived the earliest
# And this is exactly how an exchange works

print(order_book.get_best_ask())
print(order_book.get_volume_at_price('bid', order_book.get_best_bid()))
print(order_book.get_best_bid())
bids, asks, trades_done = order_book.get_all()
print(trades_done)
print(bids[0])
print(asks)
print("#########################")
# We submit a limit order Bid that has a price higher than the best Ask, i.e. the Ask with the lowest price
crossing_limit_order = {'type': 'limit',
                        'side': 'bid',
                        'quantity': 2,
                        'price': 102,
                        'trade_id': 109}

print(crossing_limit_order)
trades, order_in_book = order_book.process_order(crossing_limit_order, False, False)
print("A Trade takes place because the incoming Bid crosses the best Ask")
print(trades)
#We can see that a trade did take place at the price of the best Ask, i.e. 101, and that 2 units were exchanged
#The interesting thing to note here is that only the first seller took part in the transaction
#And this is the proof that the order of arrival of the orders is what matters: it is the best Ask that arrived the earliest that took part in the transaction
print(order_book)
bids, asks, trades_done = order_book.get_all()
print(trades_done[0]['price'])
print(trades_done[0]['quantity'])
print(trades_done)
#And here we can see how the order book was updated accordingly: indeed the quantity of the best Ask order decreased by 2 to reach 3
print("###############")
# If a limit order bid crosses the best Ask but is only partially matched, due to an insufficient volume at this price, the remaining volume will be
# placed in the book as a limit order Bid for the remaining quantity, always respecting the order of priority
big_crossing_limit_order = {'type': 'limit',
                            'side': 'bid',
                            'quantity': 50,
                            'price': 102,
                            'trade_id': 110}
print(big_crossing_limit_order)
trades, order_in_book = order_book.process_order(big_crossing_limit_order, False, False)
print("A big limit order bid crosses the best ask. The remaining volume is placed in the book")
print(trades)
print(order_book)
print("###########################")

# Market Order
#Market orders focus on the quantity whatever the price
# Market orders only need the specification of a side (bid or ask), a quantity, and their unique trade id
market_order = {'type': 'market',
                'side': 'ask',
                'quantity': 40,
                'trade_id': 111}
trades, order_id = order_book.process_order(market_order, False, False)
print("#################")
print(trades)
print("#############")
print("a market order takes the specified volume from the inside of the book, whatever the price")
#Market orders consume liquidity in the market
print("A market ask for 40 gives us:")
print(order_book)
print("###############")
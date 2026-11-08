# Methodology

This page describes how the market is built, how it is measured, and what guarantees the results.

## 1. The matching engine (`src/orderbook.py`)

* **Price-time priority.** Each side of the book is a sorted map from price to an `OrderList`, a doubly linked list of orders in order of arrival. Incoming orders match against the best price first, then the oldest order at that price.
* **Orders.** Market and limit orders. A limit order that crosses the book trades first and the remainder rests in the book. `cancel_order` removes an order. `modify_order` keeps the time priority when the quantity decreases and loses it when the quantity increases or the price changes.
* **Numbers.** Prices and quantities are `Decimal`. Floats are converted through their shortest representation, so 0.1 is exactly the price level 0.1. Limit prices must be positive multiples of the tick size.
* **Trades.** Every trade of the tape records the price, the quantity, the time and both counterparties with their side. `party2` is the aggressor.
* **Invariants.** After every operation, for each side of the book: the depth equals the number of price levels, the number of orders equals the size of the order map, the volume of every level equals the sum of its orders, the total volume equals the sum of all orders, head and tail pointers are consistent, and the best bid is below the best ask. `tests/test_orderbook.py` checks all of them after each of 3,000 random operations (limit, market, cancel, modify).

## 2. The agents

### Directional strategies (`src/agents.py`)

Six strategies that decide from the state of the book. A strategy updates its balance sheet when it sends an order, as if fully executed at the quoted price. `settle` (`src/simulation.py`) then sends the order and rewrites cash, PnL and shares from the real fills. Orders are immediate-or-cancel, and market buys are sized with `affordable_market_buy` so that no strategy can spend money or sell shares it does not have.

### The baseline market (`src/simulation.py`)

A random book of uniform limit orders, one trader of each strategy, and fixed bursts of uniform random limit orders after some of their orders (`REACTION_ORDERS`). Each simulation starts from a fresh book.

### The market ecology (`src/ecology.py`)

| Agent | Rule | Main parameters (default) |
|---|---|---|
| `Trader_noise_maker` | limit orders at 1 to `window` ticks from the opposite best quote, uniform; each resting order is cancelled at rate `cancel_rate` | window 20, rate 1 per level and side, cancel 0.15 |
| `Trader_noise_taker` | market orders, buy or sell with probability 1/2 | rate 2.5 per side |
| `Trader_market_maker` | cancels his quotes and posts a bid at `floor(r - h)` and an ask at `ceil(r + h)` with `r = mid - skew x inventory`, never crossing the book or himself, within an inventory limit | h = 1 tick, skew 0.05, size 10, limit 100 |
| `Trader_informed` | buys with a market order when `V - ask > threshold`, sells when `bid - V > threshold`, within a position limit | threshold 3 ticks, size 10, rate 3 |
| `Trader_metaorder` | draws a size `Q` with `P(Q > q) = (q_min / q)^alpha` and a random sign, then sends child market orders of 5 shares until done | 3 institutions, alpha 1.5, q_min 10 |
| `Trader_latent_liquidity` | limit orders at a distance `d` from a slow consensus price (exponential average of the mid), with density proportional to `d` | window 60, tau 5,000 events |

Sizes of noise orders are 1 plus an exponential variable (mean about 5 shares). The fundamental value `V` follows a Gaussian random walk (0.15 ticks per event) with rare jumps of 15 ticks (probability 0.0005 per event).

**Scheduling.** Time is counted in events. At each event one action is drawn with a probability proportional to its rate: noise limit order, noise market order, noise cancellation (rate proportional to the number of resting noise orders), market maker requote, informed trade, metaorder child, directional strategy step, latent order, latent cancellation. The first 3,000 events contain only noise flow, to build the book.

**Settlement.** The `Exchange` assigns every agent a unique id and, for every trade, calls `on_fill` on both counterparties. Passive liquidity providers therefore have a real cash and inventory.

**Common random numbers.** Every event draws exactly four uniform numbers from one generator, whatever happens, and the fundamental value has its own generator. Two runs with the same seed therefore see the same randomness, and an extra agent inserted through `hook` (the test metaorder of the impact study) changes the market only through its own orders.

## 3. Measurements (`src/analysis.py`, `src/experiments.py`)

All prices are in ticks. The sign `s` of a trade for an agent is +1 when he bought.

| Quantity | Definition | Book |
|---|---|---|
| Order-sign autocorrelation | `C(l) = E[e_t e_(t+l)] - E[e]^2` over aggressive orders, pooled over runs; shown in logarithmic bins where it exceeds two standard errors | Eq. 10.1 |
| Response function | `R(l) = E[e_t (m_(t+l) - m_t)]`, `m_t` the mid-price just before the t-th aggressive order | Eq. 11.5 |
| Impact against volume | lag-1 impact by bins of `v / V_best`, fitted as a power law | Eq. 11.7 |
| Markout | `s (m(t + h) - price)` per share, by kind of counterparty | |
| PnL decomposition | spread `s q (m_before - price)`, adverse selection `s q (m(t + 100) - m_before)`, inventory `s q (m_end - m(t + 100))` plus initial shares times `m_end - m_start` | |
| Metaorder impact | mid-price path with the metaorder minus mid-price path without it, same seed | chapter 12 |

The three parts of the PnL decomposition add up exactly to the mark-to-market PnL computed from each agent's own cash and shares: `tests/test_ecology.py` checks the identity for every non-noise agent.

## 4. Validation

* 63 tests: matching rules and invariants, accounting of every agent, solvency, determinism, the statistical behaviour of the agents (Pareto sizes, persistence created by metaorders, V-shaped latent liquidity, inventory skew), the measurement functions on cases with known answers, and the notebook (in sync with `src/`, valid, executed without errors).
* The notebook gives identical outputs when run twice, and identical printed numbers on Python 3.11, 3.12 and 3.13 with numpy 1.26 / pandas 2.1 / matplotlib 3.8 and with numpy 2.4 / pandas 3.1 / matplotlib 3.11.

## 5. Limits

* Parameters are set by hand to produce a realistic spread and price discovery, not calibrated to data. Runs are short (20,000 events).
* Agents have no latency, the informed trader sees the fundamental exactly, there is no self-trade prevention beyond the market maker never crossing his own quotes, and the directional strategies act in a fixed order.
* The baseline market's dynamics depend on `REACTION_ORDERS`.

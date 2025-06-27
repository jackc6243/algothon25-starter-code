import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class BaseStrategy:
    """Base class for all trading strategies"""

    def __init__(self, n_inst):
        self.curPos = np.zeros(n_inst)

    def reset_positions(self, n_inst):
        """Reset positions when number of instruments changes"""
        if self.curPos.shape[0] != n_inst:
            self.curPos = np.zeros(n_inst)


class BaseStrategyClass(BaseStrategy):
    """Base Strategy (Stateful)"""

    def __init__(self, n_inst, capital_allocation=5000):
        super().__init__(n_inst)
        self.capital_allocation = capital_allocation

    def __call__(self, prcSoFar):
        (nins, nt) = prcSoFar.shape
        self.reset_positions(nins)

        if nt < 2:
            self.curPos = np.zeros(nins)
            return self.curPos

        lastRet = np.log(prcSoFar[:, -1] / prcSoFar[:, -2])
        lNorm = np.sqrt(lastRet.dot(lastRet))
        lastRet /= lNorm
        rpos = np.array(
            [int(x) for x in self.capital_allocation * lastRet / prcSoFar[:, -1]]
        )
        self.curPos = rpos
        return self.curPos


class LinearRegressionStrategy(BaseStrategy):
    """Linear Regression Strategy (Stateful)"""

    def __init__(
        self, n_inst, look_back_period=7, min_threshold=0.001, capital_allocation=5000
    ):
        super().__init__(n_inst)
        self.look_back_period = look_back_period
        self.min_threshold = min_threshold
        self.capital_allocation = capital_allocation

    def __call__(self, prcSoFar):
        (nins, nt) = prcSoFar.shape
        self.reset_positions(nins)

        if nt < self.look_back_period + 1:
            self.curPos = np.zeros(nins)
            return self.curPos

        positions = np.zeros(nins)
        for i in range(nins):
            X = []
            y = []
            for t in range(nt - self.look_back_period):
                X.append(prcSoFar[i, t : t + self.look_back_period])
                y.append(prcSoFar[i, t + self.look_back_period])
            if len(X) == 0:
                continue
            X = np.array(X)
            y = np.array(y)
            X_b = np.c_[np.ones(X.shape[0]), X]
            try:
                w = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
                x_pred = np.r_[1, prcSoFar[i, -self.look_back_period :]]
                next_price_pred = x_pred @ w
                current_price = prcSoFar[i, -1]
                expected_return = (next_price_pred - current_price) / current_price
                if abs(expected_return) > self.min_threshold:
                    position_size = int(
                        self.capital_allocation * expected_return / current_price
                    )
                    positions[i] = position_size
            except np.linalg.LinAlgError:
                if nt >= 2:
                    momentum = (prcSoFar[i, -1] - prcSoFar[i, -2]) / prcSoFar[i, -2]
                    if abs(momentum) > self.min_threshold:
                        position_size = int(
                            self.capital_allocation * momentum / prcSoFar[i, -1]
                        )
                        positions[i] = position_size

        self.curPos = positions
        return self.curPos


class MeanReversionStrategy2(BaseStrategy):
    """Mean Reversion Strategy (Stateful)"""

    def __init__(
        self,
        n_inst,
        lookback_period=120,
        standard_deviations=2,
        max_look_back=70,
        capital_allocation=7000,
    ):
        super().__init__(n_inst)
        self.lookback_period = lookback_period
        self.standard_deviations = standard_deviations
        self.max_look_back = max_look_back
        self.capital_allocation = capital_allocation
        self.constant_position_size = 80

    def __call__(self, prcSoFar):
        (nins, nt) = prcSoFar.shape
        self.reset_positions(nins)

        MIN_DATA = max(self.lookback_period + 1, self.max_look_back + 1)
        if nt < MIN_DATA:
            self.curPos = np.zeros(nins)
            return self.curPos

        # Start with current positions and add new signals
        positions = self.curPos.copy()

        for i in range(nins):
            prices = prcSoFar[i, -self.lookback_period - 1 : -1]
            current_price = prcSoFar[i, -1]
            sma = np.mean(prices)  # Middle line (SMA)
            std_dev = np.std(prices)
            upper_band = sma + (self.standard_deviations * std_dev)
            lower_band = sma - (self.standard_deviations * std_dev)
            trend_start_price = prcSoFar[i, -self.max_look_back - 1]
            trend_end_price = current_price
            trend_direction = np.sign(trend_end_price - trend_start_price)

            signal = 0
            if positions[i] == 0:
                # Generate buy/sell signal based on mean reversion (entry signals)
                if current_price < lower_band:
                    signal = 1  # Buy signal - price below lower band and trend is up
                elif current_price > upper_band:
                    signal = (
                        -1
                    )  # Sell signal - price above upper band and trend is down
            else:
                # Take profit logic - close position when price touches the middle line (SMA)
                if positions[i] > 0:  # Long position
                    if (
                        current_price >= sma
                    ):  # Take profit when price reaches SMA (middle line)
                        signal = -1  # Take profit - sell when price reaches middle line
                elif positions[i] < 0:  # Short position
                    if (
                        current_price <= sma
                    ):  # Take profit when price reaches SMA (middle line)
                        signal = 1  # Take profit - buy when price reaches middle line

            # Add constant position size based on signal
            position_change = signal * self.constant_position_size

            # Print trade information when a signal is generated
            # if signal != 0:
            #     signal_type = "BUY" if signal == 1 else "SELL"
            #     print(
            #         f"MeanReversion TRADE: Instrument {i}, {signal_type}, Price: {current_price:.4f}, Position Change: {position_change}, Current Position: {positions[i]}"
            #     )

            positions[i] += position_change

        self.curPos = positions
        return self.curPos


class MomentumStrategy(BaseStrategy):
    """Momentum Strategy (Stateful)"""

    def __init__(
        self,
        n_inst,
        momentum_period=5,
        momentum_threshold=0.02,
        capital_allocation=5000,
    ):
        super().__init__(n_inst)
        self.momentum_period = momentum_period
        self.momentum_threshold = momentum_threshold
        self.capital_allocation = capital_allocation

    def __call__(self, prcSoFar):
        (nins, nt) = prcSoFar.shape
        self.reset_positions(nins)

        if nt < self.momentum_period + 1:
            self.curPos = np.zeros(nins)
            return self.curPos

        # Start with current positions and add new signals
        positions = self.curPos.copy()

        for i in range(nins):
            current_price = prcSoFar[i, -1]
            past_price = prcSoFar[i, -(self.momentum_period + 1)]
            if past_price != 0:
                momentum = (current_price - past_price) / past_price
            else:
                momentum = 0

            # Generate buy/sell signal
            signal = 0
            if momentum > self.momentum_threshold:
                signal = 1  # Buy signal
            elif momentum < -self.momentum_threshold:
                signal = -1  # Sell signal

            # Add to existing position based on signal
            if current_price > 0:
                position_change = int(self.capital_allocation * signal / current_price)
                positions[i] += position_change

        self.curPos = positions
        return self.curPos


class MomentumROCStrategy(BaseStrategy):
    """Momentum strategy using Rate of Change (ROC) oscillator"""

    def __init__(
        self,
        n_inst,
        roc_period=5,
        ma_period=3,
        roc_threshold=0.02,
        capital_allocation=5000,
    ):
        super().__init__(n_inst)
        self.roc_period = roc_period
        self.ma_period = ma_period
        self.roc_threshold = roc_threshold
        self.capital_allocation = capital_allocation

    def __call__(self, prcSoFar):
        n_inst, n_t = prcSoFar.shape
        self.reset_positions(n_inst)

        # Check data adequacy
        min_data_required = self.roc_period + self.ma_period
        if n_t < min_data_required:
            self.curPos = np.zeros(n_inst)
            return self.curPos

        positions = self.curPos.copy()

        for i in range(n_inst):
            # Extract price history for current instrument
            prices = prcSoFar[i, :]
            current_price = prices[-1]

            # Calculate ROC and its moving average
            roc_values = []
            for t in range(n_t - self.roc_period, n_t):
                if t >= self.roc_period:
                    past_price = prices[t - self.roc_period]
                    if past_price > 0:
                        roc = (prices[t] - past_price) / past_price
                    else:
                        roc = 0
                    roc_values.append(roc)

            # Calculate ROC MA if enough data exists
            if len(roc_values) >= self.ma_period:
                roc_ma = np.mean(roc_values[-self.ma_period :])
                current_roc = roc_values[-1]

                # Generate signal based on ROC deviation
                signal = 0
                if current_roc - roc_ma > self.roc_threshold:
                    signal = 1  # Bullish momentum
                elif current_roc - roc_ma < -self.roc_threshold:
                    signal = -1  # Bearish momentum
            else:
                signal = 0

            # Update position
            if current_price > 0:
                position_change = int(self.capital_allocation * signal / current_price)
                positions[i] += position_change

        self.curPos = positions
        return positions


class StatArbStrategy(BaseStrategy):
    """Statistical Arbitrage Strategy (Stateful)"""

    def __init__(
        self, n_inst, lookback=60, z_entry=2.0, max_pairs=5, capital_per_pair=5000
    ):
        super().__init__(n_inst)
        self.lookback = lookback
        self.z_entry = z_entry
        self.max_pairs = max_pairs
        self.capital_per_pair = capital_per_pair

    def __call__(self, prcSoFar):
        (nins, nt) = prcSoFar.shape
        self.reset_positions(nins)

        if nt < self.lookback + 10:
            self.curPos = np.zeros(nins)
            return self.curPos

        positions = np.zeros(nins)
        pairs = []
        for i in range(nins):
            for j in range(i + 1, nins):
                pairs.append((i, j))
        pair_stats = []
        for i, j in pairs:
            prices_i = prcSoFar[i, -self.lookback :]
            prices_j = prcSoFar[j, -self.lookback :]
            X = np.vstack([np.ones(self.lookback), prices_j]).T
            y = prices_i
            beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
            alpha, beta = beta_hat
            spread = prices_i - (alpha + beta * prices_j)
            spread_mean = np.mean(spread)
            spread_std = np.std(spread)
            current_spread = prcSoFar[i, -1] - (alpha + beta * prcSoFar[j, -1])
            z_score = (
                (current_spread - spread_mean) / spread_std if spread_std > 0 else 0
            )
            pair_stats.append(
                {
                    "pair": (i, j),
                    "z_score": abs(z_score),
                    "spread_std": spread_std,
                    "alpha": alpha,
                    "beta": beta,
                    "spread_mean": spread_mean,
                }
            )
        df_pairs = pd.DataFrame(pair_stats)
        df_pairs = df_pairs[df_pairs["spread_std"] > 0]
        df_pairs = df_pairs.sort_values("z_score", ascending=False)
        selected_pairs = df_pairs.head(self.max_pairs)
        for _, row in selected_pairs.iterrows():
            i, j = row["pair"]
            alpha = row["alpha"]
            beta = row["beta"]
            spread_mean = row["spread_mean"]
            spread_std = row["spread_std"]
            current_spread = prcSoFar[i, -1] - (alpha + beta * prcSoFar[j, -1])
            actual_z = (current_spread - spread_mean) / spread_std
            if actual_z > self.z_entry and prcSoFar[i, -1] > 0 and prcSoFar[j, -1] > 0:
                positions[i] -= self.capital_per_pair / prcSoFar[i, -1]
                positions[j] += (self.capital_per_pair * beta) / prcSoFar[j, -1]
            elif (
                actual_z < -self.z_entry and prcSoFar[i, -1] > 0 and prcSoFar[j, -1] > 0
            ):
                positions[i] += self.capital_per_pair / prcSoFar[i, -1]
                positions[j] -= (self.capital_per_pair * beta) / prcSoFar[j, -1]

        self.curPos = positions.astype(int)
        return self.curPos


class LeadLagStrategy(BaseStrategy):
    """Lead-Lag Strategy (Stateful)"""

    def __init__(
        self,
        n_inst,
        update_interval_days=30,
        return_threshold=0.01,
        capital_allocation=1000,
    ):
        super().__init__(n_inst)
        self.update_interval_days = update_interval_days
        self.return_threshold = return_threshold
        self.capital_allocation = capital_allocation
        self.lead_lag_pairs = None
        self.last_pairs_update = 0

    def __call__(self, prcSoFar):
        nins, nt = prcSoFar.shape
        self.reset_positions(nins)
        positions = np.zeros(nins)

        # 1. Update lead/lag pairs every update_interval_days days
        if (
            self.lead_lag_pairs is None
            or nt - self.last_pairs_update > self.update_interval_days
        ):
            self.lead_lag_pairs = discover_lead_lag_pairs(prcSoFar)
            self.last_pairs_update = nt

        # 2. Lead/lag strategy
        for leader, lagger, lag, corr in self.lead_lag_pairs:
            if nt > lag + 1 and prcSoFar[lagger, -1] > 0:
                leader_ret = np.log(
                    prcSoFar[leader, -lag - 1] / prcSoFar[leader, -lag - 2]
                )
                if abs(leader_ret) > self.return_threshold:
                    pos_size = int(
                        self.capital_allocation
                        * corr
                        * leader_ret
                        / prcSoFar[lagger, -1]
                    )
                    positions[lagger] += pos_size

        self.curPos = positions
        return self.curPos


class MixedStrategy(BaseStrategy):
    """Mixed Strategy combining multiple strategies"""

    def __init__(self, n_inst, strategy_classes, strategy_params_list, decays):
        super().__init__(n_inst)
        self.strategy_classes = strategy_classes
        self.strategy_params_list = strategy_params_list
        self.decays = decays
        self.strategy_scores = np.ones(len(strategy_classes))
        # Create strategy instances
        self.strategies = []
        for i, (strategy_class, params) in enumerate(
            zip(strategy_classes, strategy_params_list)
        ):
            self.strategies.append(strategy_class(n_inst, **params))

            # Ensure all strategies have the correct number of instruments
        for strat in self.strategies:
            strat.reset_positions(n_inst)

    def __call__(self, prcSoFar):
        n_inst, n_t = prcSoFar.shape
        self.reset_positions(n_inst)
        n_strategies = len(self.strategies)

        if n_t < 2:  # Need at least 2 time points to calculate price changes
            return np.zeros(n_inst)

        # Update scores based on previous predictions' performance
        for i in range(n_strategies):
            prev_prediction = self.strategies[i].curPos
            price_change = prcSoFar[:, -1] - prcSoFar[:, -2]
            profit = np.sum(prev_prediction * price_change)
            if profit > 0:
                self.strategy_scores[i] += 1
            elif profit < 0:
                self.strategy_scores[i] -= 1

        predictions = [strat(prcSoFar) for strat in self.strategies]

        # Select the best performing strategy
        best_strat_idx = np.argmax(self.strategy_scores)
        final_positions = predictions[best_strat_idx]

        # Apply decay to strategy scores
        self.strategy_scores = self.strategy_scores * self.decays
        self.curPos = final_positions

        return final_positions


class EmaSmaCrossStrategy(BaseStrategy):
    """Strategy using EMA and SMA crossovers with user-specified periods and capital allocation."""

    def __init__(self, n_inst, ema_period=10, sma_period=30, capital_allocation=7000):
        super().__init__(n_inst)
        self.ema_period = ema_period
        self.sma_period = sma_period
        self.capital_allocation = capital_allocation
        # State for each instrument
        self.prev_ema = [None] * n_inst
        self.sma_window = [[] for _ in range(n_inst)]
        self.prev_sma = [None] * n_inst

    def reset_positions(self, n_inst):
        super().reset_positions(n_inst)
        # Reset state if number of instruments changes
        self.prev_ema = [None] * n_inst
        self.sma_window = [[] for _ in range(n_inst)]
        self.prev_sma = [None] * n_inst

    def __call__(self, prcSoFar):
        (nins, nt) = prcSoFar.shape
        positions = np.zeros(nins)
        alpha = 2 / (self.ema_period + 1)

        for i in range(nins):
            price_data = prcSoFar[i, :]
            current_price = price_data[-1]
            # Update SMA window
            if len(self.sma_window[i]) >= self.sma_period:
                self.sma_window[i].pop(0)
            self.sma_window[i].append(current_price)
            # Calculate rolling SMA
            if len(self.sma_window[i]) == self.sma_period:
                curr_sma = np.mean(self.sma_window[i])
            else:
                curr_sma = None
            # Calculate streaming EMA
            if self.prev_ema[i] is None:
                curr_ema = current_price
            else:
                curr_ema = alpha * current_price + (1 - alpha) * self.prev_ema[i]
            # Crossover logic (need previous values)
            signal = 0
            if (
                self.prev_ema[i] is not None
                and self.prev_sma[i] is not None
                and curr_sma is not None
            ):
                if curr_ema > curr_sma:
                    signal = 1  # EMA crossed above SMA: buy
                elif curr_ema < curr_sma:
                    signal = -1  # EMA crossed below SMA: sell

                # if self.prev_ema[i] < self.prev_sma[i] and curr_ema > curr_sma:
                #     signal = 1  # EMA crossed above SMA: buy
                # elif self.prev_ema[i] > self.prev_sma[i] and curr_ema < curr_sma:
                #     signal = -1  # EMA crossed below SMA: sell
            # Position sizing
            if signal != 0 and current_price > 0:
                position_size = int(self.capital_allocation * signal / current_price)
                # position_size = int(50 * signal)
                positions[i] = position_size
            else:
                positions[i] = 0
            # Update state
            self.prev_ema[i] = curr_ema
            self.prev_sma[i] = curr_sma
        self.curPos = positions
        return self.curPos


# ----------- Lead-Lag Pair Discovery (Stateless Helper) -----------
def discover_lead_lag_pairs(prcSoFar, window=60, max_lag=10, min_corr=0.3, top_n=5):
    nins, nt = prcSoFar.shape
    if nt < window + max_lag + 2:
        return []
    returns = np.diff(np.log(prcSoFar[:, -window - max_lag :]), axis=1)
    pairs = []
    for i in range(nins):
        for j in range(nins):
            if i == j:
                continue
            best_corr = 0
            best_lag = 0
            for lag in range(1, max_lag + 1):
                x = returns[i, :-lag]
                y = returns[j, lag:]
                if x.std() == 0 or y.std() == 0:
                    continue
                corr = np.corrcoef(x, y)[0, 1]
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag
            if abs(best_corr) > min_corr:
                pairs.append((i, j, best_lag, best_corr))
    pairs = sorted(pairs, key=lambda x: abs(x[3]), reverse=True)
    return pairs[:top_n]


class MarkovStrategy(BaseStrategy):
    """Markov Chain Strategy for Price Prediction (Stateful)"""

    def __init__(
        self,
        n_inst,
        lookback_period=11,
        capital_allocation=6000,
        min_confidence=0.9,  # Minimum confidence threshold for trading
    ):
        super().__init__(n_inst)
        self.lookback_period = lookback_period
        self.capital_allocation = capital_allocation
        self.min_confidence = min_confidence
        self.transition_matrices = [None] * n_inst

    def _build_transition_matrix(self, price_series):
        """
        Build a 2-state Markov transition matrix based on price changes:
        State 0: price down or unchanged
        State 1: price up
        """
        n = len(price_series)
        if n < 3:  # Need at least 3 points for 2 transitions
            return np.array([[0.5, 0.5], [0.5, 0.5]])  # Uniform if insufficient data

        # Convert price series to states (0 = down/flat, 1 = up)
        states = np.zeros(n - 1, dtype=int)
        for i in range(1, n):
            states[i - 1] = 1 if price_series[i] > price_series[i - 1] else 0

        # Count state transitions
        transition_counts = np.zeros((2, 2))
        for i in range(len(states) - 1):
            current_state = states[i]
            next_state = states[i + 1]
            transition_counts[current_state, next_state] += 1

        # Convert counts to probabilities
        transition_matrix = np.zeros((2, 2))
        for i in range(2):
            total_transitions = np.sum(transition_counts[i])
            if total_transitions > 0:
                transition_matrix[i] = transition_counts[i] / total_transitions
            else:
                transition_matrix[i] = [0.5, 0.5]  # Default to 50/50

        return transition_matrix

    def _get_signal_strength(self, prob_up, prob_down):
        """
        Calculate signal strength based on probability confidence
        """
        max_prob = max(prob_up, prob_down)

        # Only trade if confidence exceeds minimum threshold
        if max_prob < self.min_confidence:
            return 0

        # Signal strength based on confidence level
        if prob_up > prob_down:
            return min(1.0, (prob_up - 0.5) * 2)  # Scale 0.5-1.0 to 0-1.0
        else:
            return -min(1.0, (prob_down - 0.5) * 2)  # Negative for sell signal

    def __call__(self, prcSoFar):
        (nins, nt) = prcSoFar.shape
        self.reset_positions(nins)

        if nt < self.lookback_period + 2:  # Need extra data for transitions
            self.curPos = np.zeros(nins)
            return self.curPos

        positions = self.curPos.copy()

        for i in range(nins):
            # Get recent price history
            prices = prcSoFar[i, -(self.lookback_period + 1) :]
            current_price = prices[-1]

            if current_price <= 0:
                continue

            # Build/update transition matrix for this instrument
            transition_matrix = self._build_transition_matrix(prices)
            self.transition_matrices[i] = transition_matrix

            # Determine current state (based on latest price change)
            if len(prices) >= 2:
                current_state = 1 if prices[-1] > prices[-2] else 0
            else:
                continue

            # Get probabilities for next state
            prob_down = transition_matrix[current_state, 0]  # P(next state = down)
            prob_up = transition_matrix[current_state, 1]  # P(next state = up)

            # Generate trading signal based on Markov prediction
            signal_strength = self._get_signal_strength(prob_up, prob_down)

            # Position sizing based on signal strength
            if abs(signal_strength) > 0:
                position_change = int(
                    self.capital_allocation * signal_strength / current_price
                )
                positions[i] += position_change

        self.curPos = positions
        return self.curPos

    def get_transition_probabilities(self, instrument_id):
        """
        Get the current transition matrix for a specific instrument
        Useful for debugging and analysis
        """
        if 0 <= instrument_id < len(self.transition_matrices):
            return self.transition_matrices[instrument_id]
        return None

    def print_strategy_state(self, prcSoFar, instrument_id=0):
        """
        Print current strategy state for debugging
        """
        if instrument_id < len(self.transition_matrices):
            matrix = self.transition_matrices[instrument_id]
            if matrix is not None:
                print(f"Instrument {instrument_id} Transition Matrix:")
                print(
                    f"From DOWN state: {matrix[0][0]:.3f} stay DOWN, {matrix[0][1]:.3f} go UP"
                )
                print(
                    f"From UP state:   {matrix[1][0]:.3f} go DOWN, {matrix[1][1]:.3f} stay UP"
                )

                # Current state
                prices = prcSoFar[instrument_id, -(self.lookback_period + 1) :]
                if len(prices) >= 2:
                    current_state = 1 if prices[-1] > prices[-2] else 0
                    state_name = "UP" if current_state == 1 else "DOWN"
                    next_up_prob = matrix[current_state, 1]
                    print(f"Current state: {state_name}")
                    print(f"Predicted probability of UP next: {next_up_prob:.3f}")


# ----------- Plotting Function with SMA/EMA and Bollinger Bands -----------
def plot_price_with_indicators(
    prices, instrument_idx=0, period=20, std_dev=1.6, ma_type="SMA", figsize=(12, 8)
):
    """
    Plot prices with moving average and Bollinger bands.

    Parameters:
    - prices: numpy array of shape (n_instruments, n_timepoints)
    - instrument_idx: index of the instrument to plot (default 0)
    - period: period for moving average calculation (default 20)
    - std_dev: number of standard deviations for Bollinger bands (default 1.6)
    - ma_type: 'SMA' for Simple Moving Average or 'EMA' for Exponential Moving Average
    - figsize: tuple for figure size (default (12, 8))
    """
    # Extract price data for the specified instrument
    price_data = prices[instrument_idx, :]
    time_points = np.arange(len(price_data))

    # Calculate moving average
    if ma_type.upper() == "SMA":
        # Simple Moving Average
        ma_values = []
        for i in range(len(price_data)):
            if i < period - 1:
                ma_values.append(np.nan)
            else:
                ma_values.append(np.mean(price_data[i - period + 1 : i + 1]))
        ma_values = np.array(ma_values)
    elif ma_type.upper() == "EMA":
        # Exponential Moving Average
        alpha = 2 / (period + 1)
        ma_values = np.zeros_like(price_data)
        ma_values[0] = price_data[0]
        for i in range(1, len(price_data)):
            ma_values[i] = alpha * price_data[i] + (1 - alpha) * ma_values[i - 1]
    else:
        raise ValueError("ma_type must be 'SMA' or 'EMA'")

    # Calculate Bollinger Bands
    upper_band = np.zeros_like(price_data)
    lower_band = np.zeros_like(price_data)

    for i in range(len(price_data)):
        if i < period - 1:
            upper_band[i] = np.nan
            lower_band[i] = np.nan
        else:
            window_prices = price_data[i - period + 1 : i + 1]
            window_ma = np.mean(window_prices)
            window_std = np.std(window_prices)
            upper_band[i] = window_ma + (std_dev * window_std)
            lower_band[i] = window_ma - (std_dev * window_std)

    # Create the plot
    plt.figure(figsize=figsize)

    # Plot price data
    plt.plot(time_points, price_data, label="Price", color="black", linewidth=1.5)

    # Plot moving average
    plt.plot(
        time_points, ma_values, label=f"{ma_type} ({period})", color="blue", linewidth=2
    )

    # Plot Bollinger Bands
    plt.plot(
        time_points,
        upper_band,
        label=f"Upper Band ({std_dev}σ)",
        color="red",
        linestyle="--",
        alpha=0.7,
    )
    plt.plot(
        time_points,
        lower_band,
        label=f"Lower Band ({std_dev}σ)",
        color="red",
        linestyle="--",
        alpha=0.7,
    )

    # Fill the area between Bollinger Bands
    plt.fill_between(
        time_points,
        upper_band,
        lower_band,
        alpha=0.1,
        color="gray",
        label="Bollinger Band Area",
    )

    # Customize the plot
    plt.title(
        f"Instrument {instrument_idx} - Price with {ma_type} and Bollinger Bands",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Price", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Add some statistics
    stats_text = f"Period: {period}\nStd Dev: {std_dev}\nCurrent Price: {price_data[-1]:.2f}\nMA: {ma_values[-1]:.2f}"
    plt.text(
        0.02,
        0.98,
        stats_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    plt.show()

    return ma_values, upper_band, lower_band

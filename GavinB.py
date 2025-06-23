import numpy as np
import pandas as pd


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


class MeanReversionStrategy(BaseStrategy):
    """Mean Reversion Strategy (Stateful)"""

    def __init__(
        self,
        n_inst,
        lookback_period=14,
        standard_deviations=2,
        max_look_back=110,
        capital_allocation=7000,
    ):
        super().__init__(n_inst)
        self.lookback_period = lookback_period
        self.standard_deviations = standard_deviations
        self.max_look_back = max_look_back
        self.capital_allocation = capital_allocation

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
            sma = np.mean(prices)
            std_dev = np.std(prices)
            upper_band = sma + (self.standard_deviations * std_dev)
            lower_band = sma - (self.standard_deviations * std_dev)
            trend_start_price = prcSoFar[i, -self.max_look_back - 1]
            trend_end_price = current_price
            trend_direction = np.sign(trend_end_price - trend_start_price)

            # Generate buy/sell signal based on mean reversion
            signal = 0
            if current_price < lower_band and trend_direction >= 0:
                signal = 1  # Buy signal - price below lower band and trend is up
            elif current_price > upper_band and trend_direction <= 0:
                signal = -1  # Sell signal - price above upper band and trend is down

            # Add to existing position based on signal
            if current_price > 0:
                position_change = int(self.capital_allocation * signal / current_price)
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

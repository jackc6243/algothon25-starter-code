import numpy as np
import pandas as pd
from itertools import combinations, product
import pprint


def reset_function_attributes(func):
    # Collect attribute names first to avoid changing the dict while iterating
    attrs = list(func.__dict__.keys())
    for attr in attrs:
        delattr(func, attr)


class BackTester:
    commissionFee = 0.0005  # Same as commRate in correct backtest
    maxPos = 10000  # Same as dlrPosLimit in correct backtest

    def __init__(self, dataFileName):
        df = pd.read_csv(dataFileName, sep=r"\s+", header=None, index_col=None)
        self.nt, self.nInst = df.shape
        self.data = df.values.T  # Transpose to (nInst, nt) format
        self.strategy_class = None
        self.strategy_params = {}

    def set_getPos(self, strategy_class, **params):
        """Set the strategy class and its parameters"""
        self.strategy_class = strategy_class
        self.strategy_params = params

    def backtest(self, numTestDays, params=None):
        """Backtest matching the correct logic while returning all information"""
        if self.strategy_class is None:
            raise ValueError("Strategy class not set. Use set_getPos() first.")

        # Merge default params with provided params
        if params:
            strategy_params = {**self.strategy_params, **params}
        else:
            strategy_params = self.strategy_params.copy()

        # Create strategy instance
        strategy = self.strategy_class(self.nInst, **strategy_params)

        cash = 0
        curPos = np.zeros(self.nInst)
        totDVolume = 0
        value = 0
        todayPLL = []
        dailyPositions = []

        # Calculate start day index (1-based index in correct backtest)
        startDay = (
            self.nt + 1 - numTestDays
        )  # Matches correct backtest's startDay calculation

        # Loop through test days (inclusive of last day)
        for t in range(startDay, self.nt + 1):
            # Get price history up to current day (0-indexed)
            prcHistSoFar = self.data[:, :t]
            curPrices = prcHistSoFar[:, -1]  # Current day's prices

            # Record position at beginning of day
            dailyPositions.append(curPos.copy())

            # Trading logic (skip trading on very last day)
            if t < self.nt:  # Not the last day
                # Get new positions using strategy instance
                newPosOrig = strategy(prcHistSoFar)

                # Apply position limits
                posLimits = np.array(
                    [
                        int(x)
                        for x in self.maxPos / np.where(curPrices > 0, curPrices, 1)
                    ]
                )
                newPos = np.clip(newPosOrig, -posLimits, posLimits)

                # Calculate trades
                deltaPos = newPos - curPos
                dvolumes = curPrices * np.abs(deltaPos)
                dvolume = np.sum(dvolumes)
                totDVolume += dvolume
                comm = dvolume * self.commissionFee
                cash -= np.dot(curPrices, deltaPos) + comm

                # Update positions
                curPos = newPos.copy()

            # Calculate portfolio value
            posValue = np.dot(curPos, curPrices)
            todayPL = cash + posValue - value
            value = cash + posValue

            # Record P&L (skip first day as in correct backtest)
            if t > startDay:
                todayPLL.append(todayPL)

        # Prepare outputs
        dailyPositions = np.array(dailyPositions).T  # Transpose to (nInst, days)

        # Calculate statistics
        pll = np.array(todayPLL)
        plmu = np.mean(pll)
        plstd = np.std(pll)
        annSharpe = np.sqrt(249) * plmu / plstd if plstd > 0 else 0.0
        ret = value / totDVolume if totDVolume > 0 else 0.0
        score = plmu - 0.1 * plstd

        stats = {
            "meanPL": plmu,
            "return": ret,
            "stdDev": plstd,
            "annualSharpe": annSharpe,
            "totalVolume": totDVolume,
            "score": score,
        }

        return dailyPositions, todayPLL, stats

    def gridsearch(self, param_ranges, numTestDays):
        best_score = -np.inf
        best_params = None
        best_stats = None
        param_names = list(param_ranges.keys())

        # Get all value combinations for all parameters at once
        value_lists = [param_ranges[p] for p in param_names]

        for value_combo in product(*value_lists):
            params = dict(zip(param_names, value_combo))

            # Run backtest with current parameters
            _, _, stats = self.backtest(numTestDays, params=params)

            print(f"\nTesting parameters: {params}")
            print("Statistics:")
            pprint.pprint(stats)

            if stats["score"] > best_score:
                best_score = stats["score"]
                best_params = params
                best_stats = stats

        print("\n" + "=" * 50)
        print("GRID SEARCH COMPLETE")
        print("=" * 50)
        print(f"Best parameters: {best_params}")
        print("Best statistics:")
        pprint.pprint(best_stats)

        return best_params, best_stats

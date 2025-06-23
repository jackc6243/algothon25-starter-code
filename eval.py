#!/usr/bin/env python

import numpy as np
import pandas as pd

from GavinB import *

# dont touch
nInst = 0
nt = 0
commRate = 0.0005
dlrPosLimit = 10000


def loadPrices(fn):
    global nt, nInst
    df = pd.read_csv(fn, sep=r"\s+", header=None, index_col=None)
    (nt, nInst) = df.shape
    return (df.values).T


pricesFile = "./prices_train.txt"
prcAll = loadPrices(pricesFile)
print("Loaded %d instruments for %d days" % (nInst, nt))
# end dont touch


# Create strategy instances with default parameters
# You can uncomment the strategy you want to use and comment out the others
base_strategy = BaseStrategyClass(nInst, capital_allocation=5000)
linear_regression_strategy = LinearRegressionStrategy(
    nInst, look_back_period=7, min_threshold=0.001, capital_allocation=5000
)
stat_arb_strategy = StatArbStrategy(
    nInst, lookback=60, z_entry=2.0, max_pairs=5, capital_per_pair=5000
)
lead_lag_strategy = LeadLagStrategy(
    nInst, update_interval_days=30, return_threshold=0.01, capital_allocation=1000
)
momentum_strategy = MomentumStrategy(
    nInst, momentum_period=5, momentum_threshold=0.02, capital_allocation=5000
)
mean_reversion_strategy = MeanReversionStrategy(
    nInst,
    lookback_period=14,
    standard_deviations=2,
    max_look_back=110,
    capital_allocation=7000,
)
momentum_roc_strategy = MomentumROCStrategy(
    nInst, roc_period=6, ma_period=3, roc_threshold=0.01, capital_allocation=5000
)

# Create mixed strategy combining mean reversion and momentum
strategy_classes = [MeanReversionStrategy, MomentumStrategy]
strategy_params_list = [
    {
        "lookback_period": 14,
        "standard_deviations": 2,
        "max_look_back": 110,
        "capital_allocation": 7000,
    },
    {"momentum_period": 5, "momentum_threshold": 0.02, "capital_allocation": 5000},
]
decays = [0.8, 0.8]  # Decay factors for strategy scores
mixed_strategy = MixedStrategy(nInst, strategy_classes, strategy_params_list, decays)

# Set the active strategy (change this to test different strategies)
active_strategy = (
    mixed_strategy  # Using mixed strategy with mean reversion and momentum
)


def getPosition(prcHistSoFar):
    """Wrapper function that maintains the same interface as the old functions"""
    global active_strategy, nInst
    # Update strategy with current number of instruments if it changed
    if active_strategy.curPos.shape[0] != nInst:
        active_strategy.reset_positions(nInst)
    return active_strategy(prcHistSoFar)


# Don't touch below


def calcPL(prcHist, numTestDays):
    cash = 0
    curPos = np.zeros(nInst)
    totDVolume = 0
    totDVolumeSignal = 0
    totDVolumeRandom = 0
    value = 0
    todayPLL = []
    (_, nt) = prcHist.shape
    startDay = nt + 1 - numTestDays
    for t in range(startDay, nt + 1):
        prcHistSoFar = prcHist[:, :t]
        curPrices = prcHistSoFar[:, -1]
        if t < nt:
            # Trading, do not do it on the very last day of the test
            newPosOrig = getPosition(prcHistSoFar)
            posLimits = np.array([int(x) for x in dlrPosLimit / curPrices])
            newPos = np.clip(newPosOrig, -posLimits, posLimits)
            deltaPos = newPos - curPos
            dvolumes = curPrices * np.abs(deltaPos)
            dvolume = np.sum(dvolumes)
            totDVolume += dvolume
            comm = dvolume * commRate
            cash -= curPrices.dot(deltaPos) + comm
        else:
            newPos = np.array(curPos)
        curPos = np.array(newPos)
        posValue = curPos.dot(curPrices)
        todayPL = cash + posValue - value
        value = cash + posValue
        ret = 0.0
        if totDVolume > 0:
            ret = value / totDVolume
        if t > startDay:
            print(
                "Day %d value: %.2lf todayPL: $%.2lf $-traded: %.0lf return: %.5lf"
                % (t, value, todayPL, totDVolume, ret)
            )
            todayPLL.append(todayPL)
    pll = np.array(todayPLL)
    (plmu, plstd) = (np.mean(pll), np.std(pll))
    annSharpe = 0.0
    if plstd > 0:
        annSharpe = np.sqrt(249) * plmu / plstd
    return (plmu, ret, plstd, annSharpe, totDVolume)


(meanpl, ret, plstd, sharpe, dvol) = calcPL(prcAll, 200)
score = meanpl - 0.1 * plstd
print("=====")
print("mean(PL): %.1lf" % meanpl)
print("return: %.5lf" % ret)
print("StdDev(PL): %.2lf" % plstd)
print("annSharpe(PL): %.2lf " % sharpe)
print("totDvolume: %.0lf " % dvol)
print("Score: %.2lf" % score)

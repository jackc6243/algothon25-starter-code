#!/usr/bin/env python

import numpy as np
import pandas as pd
from backtester import BackTester
from GavinB import getMyPositionMomentum1
from GavinB_fixed import getMyPositionMomentum1 as getMyPositionMomentum1_fixed

print("=== COMPARISON: Original vs Fixed Strategies ===")
print("Testing getMyPositionMomentum1 with default parameters")
print("momentum_period=10, momentum_threshold=0.03, capital_allocation=8000\n")

# Create backtester
bt = BackTester("prices_train.txt")

# Test 1: Original strategy (with global state)
print("1. Original Strategy (with global state):")
bt.set_getPos(getMyPositionMomentum1)
_, _, stats_original = bt.backtest(200)
print(f"meanPL: {stats_original['meanPL']:.1f}")
print(f"return: {stats_original['return']:.5f}")
print(f"stdDev: {stats_original['stdDev']:.2f}")
print(f"annualSharpe: {stats_original['annualSharpe']:.2f}")
print(f"totalVolume: {stats_original['totalVolume']:.0f}")
print(f"score: {stats_original['score']:.2f}")

print("\n2. Fixed Strategy (no global state):")
bt.set_getPos(getMyPositionMomentum1_fixed)
_, _, stats_fixed = bt.backtest(200)
print(f"meanPL: {stats_fixed['meanPL']:.1f}")
print(f"return: {stats_fixed['return']:.5f}")
print(f"stdDev: {stats_fixed['stdDev']:.2f}")
print(f"annualSharpe: {stats_fixed['annualSharpe']:.2f}")
print(f"totalVolume: {stats_fixed['totalVolume']:.0f}")
print(f"score: {stats_fixed['score']:.2f}")

print("\n=== DIFFERENCES ===")
print(f"meanPL difference: {stats_fixed['meanPL'] - stats_original['meanPL']:.1f}")
print(f"return difference: {stats_fixed['return'] - stats_original['return']:.5f}")
print(f"stdDev difference: {stats_fixed['stdDev'] - stats_original['stdDev']:.2f}")
print(f"score difference: {stats_fixed['score'] - stats_original['score']:.2f}")

print("\n=== EXPLANATION ===")
print("The main differences are caused by:")
print("1. Global state management: Original strategies accumulate positions")
print("2. Strategy selection: eval.py only uses the last imported strategy")
print("3. Parameter handling: eval.py uses defaults, backtester can pass parameters")
print("\nTo match eval.py exactly, use the fixed strategies without global state.")

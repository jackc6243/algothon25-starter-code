# Example: How to use the functions individually in Jupyter notebook

# Step 1: Import the functions
from mean_reversion_backtest import (
    backtest_mean_reversion_strategy,
    plot_instrument_analysis,
    plot_summary_statistics,
    plot_single_instrument,
    print_performance_summary,
    get_default_strategy_params,
)

# Step 2: Run the backtest to get all data
(
    backtester,
    daily_positions,
    daily_pnl,
    stats,
    daily_stock_profits,
    all_daily_positions,
    all_daily_stock_profits,
    price_data,
) = backtest_mean_reversion_strategy("prices_train.txt", num_test_days=100)

# Step 3: Print performance summary
print_performance_summary(stats)

# Step 4: Use individual visualization functions

# Option A: Plot single instrument analysis
plot_single_instrument(
    all_daily_positions, all_daily_stock_profits, price_data, instrument_idx=0
)

# Option B: Plot analysis for multiple instruments
plot_instrument_analysis(
    all_daily_positions,
    daily_pnl,
    all_daily_stock_profits,
    price_data,
    num_instruments=3,
)

# Option C: Plot summary statistics
plot_summary_statistics(daily_pnl, stats, all_daily_stock_profits)

# Option D: Custom parameters example
custom_params = {
    "lookback_period": 20,
    "standard_deviations": 1.5,
    "max_look_back": 90,
    "capital_allocation": 5000,
}

# Run with custom parameters
(
    backtester_custom,
    daily_positions_custom,
    daily_pnl_custom,
    stats_custom,
    daily_stock_profits_custom,
    all_daily_positions_custom,
    all_daily_stock_profits_custom,
    price_data_custom,
) = backtest_mean_reversion_strategy(
    "prices_train.txt", num_test_days=100, strategy_params=custom_params
)

# Plot with custom parameters
plot_single_instrument(
    all_daily_positions_custom,
    all_daily_stock_profits_custom,
    price_data_custom,
    instrument_idx=1,
)

# Option E: Access raw data for custom analysis
print(f"Data shapes:")
print(f"  All daily positions: {all_daily_positions.shape}")
print(f"  All daily stock profits: {all_daily_stock_profits.shape}")
print(f"  Price data: {price_data.shape}")
print(f"  Daily P&L: {len(daily_pnl)}")

# Calculate custom metrics
import numpy as np

cumulative_pnl = np.cumsum(daily_pnl)
max_drawdown = np.min(cumulative_pnl - np.maximum.accumulate(cumulative_pnl))
win_rate = np.sum(np.array(daily_pnl) > 0) / len(daily_pnl)

print(f"\nCustom metrics:")
print(f"  Max drawdown: {max_drawdown:.2f}")
print(f"  Win rate: {win_rate:.2%}")
print(f"  Final cumulative P&L: {cumulative_pnl[-1]:.2f}")

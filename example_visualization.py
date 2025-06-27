"""
Example code to visualize SMA and Bollinger Bands
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from visualisation import (
    plot_price_with_indicators,
    plot_multiple_instruments_with_indicators,
    plot_price_with_two_indicators,
    plot_multiple_prices_with_two_indicators,
)
from GavinB import plot_price_with_indicators as gavin_plot_price_with_indicators
from backtester import BackTester


def load_price_data(filename="prices_train.txt"):
    """Load price data from file"""
    df = pd.read_csv(filename, sep=r"\s+", header=None, index_col=None)
    return df.values.T  # Transpose to (nInst, nt) format


def example_single_instrument_visualization():
    """Example 1: Visualize a single instrument with SMA and Bollinger Bands"""
    print("Example 1: Single Instrument Visualization")
    print("=" * 50)

    # Load data
    prices = load_price_data("prices_train.txt")

    # Plot instrument 0 with SMA and Bollinger Bands
    plot_price_with_indicators(
        prices=prices,
        instrument_idx=0,  # First instrument
        period=20,  # 20-day SMA
        std_dev=2.0,  # 2 standard deviations for Bollinger Bands
        ma_type="SMA",  # Simple Moving Average
        figsize=(12, 8),
    )


def example_multiple_instruments_visualization():
    """Example 2: Visualize multiple instruments in a grid"""
    print("\nExample 2: Multiple Instruments Visualization")
    print("=" * 50)

    # Load data
    prices = load_price_data("prices_train.txt")

    # Plot first 4 instruments
    plot_multiple_instruments_with_indicators(
        prices=prices,
        instrument_indices=[0, 1, 2, 3],  # First 4 instruments
        period=14,  # 14-day SMA
        std_dev=1.6,  # 1.6 standard deviations
        ma_type="SMA",  # Simple Moving Average
        nrows=2,  # 2 rows
        ncols=2,  # 2 columns
        figsize=(15, 10),
    )


def example_two_moving_averages():
    """Example 3: Visualize with two different moving averages"""
    print("\nExample 3: Two Moving Averages Visualization")
    print("=" * 50)

    # Load data
    prices = load_price_data("prices_train.txt")

    # Plot with both SMA and EMA
    plot_price_with_two_indicators(
        prices=prices,
        instrument_idx=0,  # First instrument
        ma1_type="SMA",  # First MA: Simple Moving Average
        ma1_period=20,  # 20-day SMA
        ma2_type="EMA",  # Second MA: Exponential Moving Average
        ma2_period=50,  # 50-day EMA
        figsize=(12, 6),
    )


def example_ema_visualization():
    """Example 4: Visualize with EMA and Bollinger Bands"""
    print("\nExample 4: EMA with Bollinger Bands")
    print("=" * 50)

    # Load data
    prices = load_price_data("prices_train.txt")

    # Plot with EMA instead of SMA
    plot_price_with_indicators(
        prices=prices,
        instrument_idx=1,  # Second instrument
        period=30,  # 30-day EMA
        std_dev=1.8,  # 1.8 standard deviations
        ma_type="EMA",  # Exponential Moving Average
        figsize=(12, 8),
    )


def example_different_parameters():
    """Example 5: Compare different parameter settings"""
    print("\nExample 5: Different Parameter Settings")
    print("=" * 50)

    # Load data
    prices = load_price_data("prices_train.txt")

    # Create subplots to compare different settings
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Setting 1: Short period, tight bands
    ax1 = axes[0, 0]
    ma_values1, upper1, lower1 = plot_price_with_indicators(
        prices=prices, instrument_idx=0, period=10, std_dev=1.0, ma_type="SMA"
    )
    ax1.set_title("Short Period (10), Tight Bands (1.0σ)")

    # Setting 2: Long period, wide bands
    ax2 = axes[0, 1]
    ma_values2, upper2, lower2 = plot_price_with_indicators(
        prices=prices, instrument_idx=0, period=50, std_dev=2.5, ma_type="SMA"
    )
    ax2.set_title("Long Period (50), Wide Bands (2.5σ)")

    # Setting 3: EMA with medium settings
    ax3 = axes[1, 0]
    ma_values3, upper3, lower3 = plot_price_with_indicators(
        prices=prices, instrument_idx=0, period=20, std_dev=1.6, ma_type="EMA"
    )
    ax3.set_title("EMA (20), Medium Bands (1.6σ)")

    # Setting 4: Very short period for quick signals
    ax4 = axes[1, 1]
    ma_values4, upper4, lower4 = plot_price_with_indicators(
        prices=prices, instrument_idx=0, period=5, std_dev=1.2, ma_type="SMA"
    )
    ax4.set_title("Very Short Period (5), Tight Bands (1.2σ)")

    plt.tight_layout()
    plt.show()


def example_backtest_visualization():
    """Example 6: Visualize after running a backtest"""
    print("\nExample 6: Backtest Visualization")
    print("=" * 50)

    from GavinB import MeanReversionStrategy2

    # Initialize backtester
    bt = BackTester("prices_train.txt")
    bt.set_getPos(
        MeanReversionStrategy2,
        lookback_period=20,
        standard_deviations=2.0,
        capital_allocation=7000,
    )

    # Run backtest
    (
        daily_positions,
        daily_pnl,
        stats,
        daily_stock_profits,
        all_daily_positions,
        all_daily_stock_profits,
    ) = bt.backtest(100)

    # Load price data
    prices = bt.data

    # Visualize the first instrument with the strategy's parameters
    plot_price_with_indicators(
        prices=prices,
        instrument_idx=0,
        period=20,  # Same as strategy's lookback_period
        std_dev=2.0,  # Same as strategy's standard_deviations
        ma_type="SMA",
        figsize=(12, 8),
    )

    print(f"Backtest Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")


def example_custom_analysis():
    """Example 7: Custom analysis with multiple instruments and parameters"""
    print("\nExample 7: Custom Analysis")
    print("=" * 50)

    # Load data
    prices = load_price_data("prices_train.txt")

    # Plot multiple instruments with two moving averages
    plot_multiple_prices_with_two_indicators(
        prices=prices,
        instrument_indices=[0, 1, 2, 3, 4, 5],  # First 6 instruments
        ma1_type="SMA",  # First MA: SMA
        ma1_period=10,  # 10-day SMA
        ma2_type="EMA",  # Second MA: EMA
        ma2_period=30,  # 30-day EMA
        nrows=3,  # 3 rows
        ncols=2,  # 2 columns
        figsize=(16, 12),
    )


def main():
    """Run all examples"""
    print("SMA and Bollinger Bands Visualization Examples")
    print("=" * 60)

    try:
        # Run examples
        example_single_instrument_visualization()
        example_multiple_instruments_visualization()
        example_two_moving_averages()
        example_ema_visualization()
        example_different_parameters()
        example_backtest_visualization()
        example_custom_analysis()

    except FileNotFoundError:
        print("Error: Make sure 'prices_train.txt' exists in the current directory")
        print(
            "You can use 'prices.txt' instead by changing the filename in the examples"
        )
    except Exception as e:
        print(f"Error running examples: {e}")


if __name__ == "__main__":
    main()

"""
Simple example to visualize SMA and Bollinger Bands
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from visualisation import plot_price_with_indicators


def load_data(filename="prices_train.txt"):
    """Load price data from file"""
    try:
        df = pd.read_csv(filename, sep=r"\s+", header=None, index_col=None)
        return df.values.T  # Transpose to (nInst, nt) format
    except FileNotFoundError:
        print(f"File {filename} not found. Trying prices.txt...")
        df = pd.read_csv("prices.txt", sep=r"\s+", header=None, index_col=None)
        return df.values.T


def basic_sma_bollinger_example():
    """Basic example: Plot SMA and Bollinger Bands for one instrument"""
    print("Basic SMA and Bollinger Bands Visualization")
    print("=" * 50)

    # Load data
    prices = load_data()

    # Plot the first instrument with SMA and Bollinger Bands
    plot_price_with_indicators(
        prices=prices,
        instrument_idx=0,  # First instrument
        period=20,  # 20-day SMA
        std_dev=2.0,  # 2 standard deviations for Bollinger Bands
        ma_type="SMA",  # Simple Moving Average
        figsize=(12, 8),
    )


def ema_example():
    """Example with Exponential Moving Average"""
    print("\nEMA and Bollinger Bands Visualization")
    print("=" * 50)

    # Load data
    prices = load_data()

    # Plot with EMA instead of SMA
    plot_price_with_indicators(
        prices=prices,
        instrument_idx=1,  # Second instrument
        period=30,  # 30-day EMA
        std_dev=1.6,  # 1.6 standard deviations
        ma_type="EMA",  # Exponential Moving Average
        figsize=(12, 8),
    )


def compare_parameters():
    """Compare different parameter settings"""
    print("\nComparing Different Parameter Settings")
    print("=" * 50)

    # Load data
    prices = load_data()

    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Different parameter combinations
    params = [
        {
            "period": 10,
            "std_dev": 1.0,
            "title": "Short Period (10), Tight Bands (1.0σ)",
        },
        {"period": 50, "std_dev": 2.5, "title": "Long Period (50), Wide Bands (2.5σ)"},
        {
            "period": 20,
            "std_dev": 1.6,
            "title": "Medium Period (20), Standard Bands (1.6σ)",
        },
        {
            "period": 5,
            "std_dev": 1.2,
            "title": "Very Short Period (5), Tight Bands (1.2σ)",
        },
    ]

    for i, param in enumerate(params):
        row = i // 2
        col = i % 2
        ax = axes[row, col]

        # Get the plot data
        ma_values, upper_band, lower_band = plot_price_with_indicators(
            prices=prices,
            instrument_idx=0,
            period=param["period"],
            std_dev=param["std_dev"],
            ma_type="SMA",
        )

        ax.set_title(param["title"])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("SMA and Bollinger Bands Visualization Examples")
    print("=" * 60)

    try:
        # Run basic example
        basic_sma_bollinger_example()

        # Run EMA example
        ema_example()

        # Run parameter comparison
        compare_parameters()

    except Exception as e:
        print(f"Error: {e}")
        print(
            "Make sure you have the required data files (prices_train.txt or prices.txt)"
        )

"""
visualisation.py

Utility for plotting price series with moving averages and Bollinger bands.
"""

import numpy as np
import matplotlib.pyplot as plt
import math


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


def plot_multiple_instruments_with_indicators(
    prices,
    instrument_indices=None,
    period=20,
    std_dev=1.6,
    ma_type="SMA",
    nrows=None,
    ncols=None,
    figsize=None,
    sharey=False,
):
    """
    Plot multiple instruments in a grid, each with price, MA, and Bollinger bands.

    Parameters:
    - prices: numpy array of shape (n_instruments, n_timepoints)
    - instrument_indices: list of instrument indices to plot (default: all)
    - period: period for moving average (default 20)
    - std_dev: std dev for Bollinger bands (default 1.6)
    - ma_type: 'SMA' or 'EMA' (default 'SMA')
    - nrows, ncols: grid shape (auto if not specified)
    - figsize: overall figure size (auto if not specified)
    - sharey: whether to share y-axis across plots (default False)
    """
    n_instruments = prices.shape[0]
    if instrument_indices is None:
        instrument_indices = list(range(n_instruments))
    n_plots = len(instrument_indices)

    # Auto-calculate grid shape if not provided
    if nrows is None and ncols is None:
        ncols = math.ceil(math.sqrt(n_plots))
        nrows = math.ceil(n_plots / ncols)
    elif nrows is None:
        nrows = math.ceil(n_plots / ncols)
    elif ncols is None:
        ncols = math.ceil(n_plots / nrows)

    # Auto-calculate figsize if not provided
    if figsize is None:
        max_total_width = 12
        height_per_row = 3
        # Cap the total width at max_total_width, divide evenly among ncols
        total_width = min(max_total_width, ncols * (max_total_width / ncols))
        width_per_plot = total_width / ncols
        figsize = (total_width, height_per_row * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=sharey)
    axes = np.array(axes).reshape(-1)  # Flatten in case of 2D

    for idx, instrument_idx in enumerate(instrument_indices):
        ax = axes[idx]
        price_data = prices[instrument_idx, :]
        time_points = np.arange(len(price_data))

        # Calculate MA
        if ma_type.upper() == "SMA":
            ma_values = [
                (
                    np.nan
                    if i < period - 1
                    else np.mean(price_data[i - period + 1 : i + 1])
                )
                for i in range(len(price_data))
            ]
            ma_values = np.array(ma_values)
        elif ma_type.upper() == "EMA":
            alpha = 2 / (period + 1)
            ma_values = np.zeros_like(price_data)
            ma_values[0] = price_data[0]
            for i in range(1, len(price_data)):
                ma_values[i] = alpha * price_data[i] + (1 - alpha) * ma_values[i - 1]
        else:
            raise ValueError("ma_type must be 'SMA' or 'EMA'")

        # Bollinger Bands
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

        # Plot
        ax.plot(time_points, price_data, label="Price", color="black", linewidth=1.2)
        ax.plot(
            time_points,
            ma_values,
            label=f"{ma_type} ({period})",
            color="blue",
            linewidth=1.5,
        )
        ax.plot(
            time_points,
            upper_band,
            label=f"Upper Band ({std_dev}σ)",
            color="red",
            linestyle="--",
            alpha=0.7,
        )
        ax.plot(
            time_points,
            lower_band,
            label=f"Lower Band ({std_dev}σ)",
            color="red",
            linestyle="--",
            alpha=0.7,
        )
        ax.fill_between(time_points, upper_band, lower_band, alpha=0.08, color="gray")
        ax.set_title(f"Instrument {instrument_idx}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.2)
        if idx % ncols == 0:
            ax.set_ylabel("Price")
        if idx >= (nrows - 1) * ncols:
            ax.set_xlabel("Time")
        # Add stats box
        stats_text = f"Cur: {price_data[-1]:.2f}\nMA: {ma_values[-1]:.2f}"
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
        )
        if idx == 0:
            ax.legend(fontsize=8)

    # Hide unused axes
    for j in range(n_plots, nrows * ncols):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def plot_price_with_two_indicators(
    prices,
    instrument_idx=0,
    ma1_type="SMA",
    ma1_period=20,
    ma2_type="EMA",
    ma2_period=50,
    figsize=(12, 6),
):
    """
    Plot prices for a single instrument with two moving averages (SMA or EMA), each with its own period and type.

    Parameters:
    - prices: numpy array of shape (n_instruments, n_timepoints)
    - instrument_idx: which instrument to plot (default 0)
    - ma1_type: 'SMA' or 'EMA' for the first moving average (default 'SMA')
    - ma1_period: period for the first moving average (default 20)
    - ma2_type: 'SMA' or 'EMA' for the second moving average (default 'EMA')
    - ma2_period: period for the second moving average (default 50)
    - figsize: tuple for figure size (default (12, 6))
    """
    price_data = prices[instrument_idx, :]
    time_points = np.arange(len(price_data))

    # First MA
    if ma1_type.upper() == "SMA":
        ma1 = [
            (
                np.nan
                if i < ma1_period - 1
                else np.mean(price_data[i - ma1_period + 1 : i + 1])
            )
            for i in range(len(price_data))
        ]
        ma1 = np.array(ma1)
    elif ma1_type.upper() == "EMA":
        alpha = 2 / (ma1_period + 1)
        ma1 = np.zeros_like(price_data)
        ma1[0] = price_data[0]
        for i in range(1, len(price_data)):
            ma1[i] = alpha * price_data[i] + (1 - alpha) * ma1[i - 1]
    else:
        raise ValueError("ma1_type must be 'SMA' or 'EMA'")

    # Second MA
    if ma2_type.upper() == "SMA":
        ma2 = [
            (
                np.nan
                if i < ma2_period - 1
                else np.mean(price_data[i - ma2_period + 1 : i + 1])
            )
            for i in range(len(price_data))
        ]
        ma2 = np.array(ma2)
    elif ma2_type.upper() == "EMA":
        alpha = 2 / (ma2_period + 1)
        ma2 = np.zeros_like(price_data)
        ma2[0] = price_data[0]
        for i in range(1, len(price_data)):
            ma2[i] = alpha * price_data[i] + (1 - alpha) * ma2[i - 1]
    else:
        raise ValueError("ma2_type must be 'SMA' or 'EMA'")

    plt.figure(figsize=figsize)
    plt.plot(time_points, price_data, label="Price", color="black", linewidth=1.5)
    plt.plot(
        time_points,
        ma1,
        label=f"{ma1_type.upper()} ({ma1_period})",
        color="blue",
        linewidth=2,
    )
    plt.plot(
        time_points,
        ma2,
        label=f"{ma2_type.upper()} ({ma2_period})",
        color="orange",
        linewidth=2,
    )
    plt.title(
        f"Instrument {instrument_idx} - Price with Two Moving Averages",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Price", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return ma1, ma2


def plot_multiple_prices_with_two_indicators(
    prices,
    instrument_indices=None,
    ma1_type="SMA",
    ma1_period=20,
    ma2_type="EMA",
    ma2_period=50,
    nrows=None,
    ncols=None,
    figsize=None,
    sharey=False,
):
    """
    Plot multiple instruments in a grid, each with two moving averages (SMA or EMA, independently chosen for each line).

    Parameters:
    - prices: numpy array of shape (n_instruments, n_timepoints)
    - instrument_indices: list/range of instrument indices to plot (default: all)
    - ma1_type: 'SMA' or 'EMA' for the first moving average (default 'SMA')
    - ma1_period: period for the first moving average (default 20)
    - ma2_type: 'SMA' or 'EMA' for the second moving average (default 'EMA')
    - ma2_period: period for the second moving average (default 50)
    - nrows, ncols: grid shape (auto if not specified)
    - figsize: overall figure size (auto if not specified)
    - sharey: whether to share y-axis across plots (default False)
    """
    import math
    import numpy as np
    import matplotlib.pyplot as plt

    n_instruments = prices.shape[0]
    if instrument_indices is None:
        instrument_indices = list(range(n_instruments))
    n_plots = len(instrument_indices)

    # Auto-calculate grid shape if not provided
    if nrows is None and ncols is None:
        ncols = math.ceil(math.sqrt(n_plots))
        nrows = math.ceil(n_plots / ncols)
    elif nrows is None:
        nrows = math.ceil(n_plots / ncols)
    elif ncols is None:
        ncols = math.ceil(n_plots / nrows)

    # Auto-calculate figsize if not provided
    if figsize is None:
        max_total_width = 12
        height_per_row = 3
        total_width = min(max_total_width, ncols * (max_total_width / ncols))
        figsize = (total_width, height_per_row * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=sharey)
    axes = np.array(axes).reshape(-1)

    for idx, instrument_idx in enumerate(instrument_indices):
        ax = axes[idx]
        price_data = prices[instrument_idx, :]
        time_points = np.arange(len(price_data))

        # First MA
        if ma1_type.upper() == "SMA":
            ma1 = [
                (
                    np.nan
                    if i < ma1_period - 1
                    else np.mean(price_data[i - ma1_period + 1 : i + 1])
                )
                for i in range(len(price_data))
            ]
            ma1 = np.array(ma1)
        elif ma1_type.upper() == "EMA":
            alpha = 2 / (ma1_period + 1)
            ma1 = np.zeros_like(price_data)
            ma1[0] = price_data[0]
            for i in range(1, len(price_data)):
                ma1[i] = alpha * price_data[i] + (1 - alpha) * ma1[i - 1]
        else:
            raise ValueError("ma1_type must be 'SMA' or 'EMA'")

        # Second MA
        if ma2_type.upper() == "SMA":
            ma2 = [
                (
                    np.nan
                    if i < ma2_period - 1
                    else np.mean(price_data[i - ma2_period + 1 : i + 1])
                )
                for i in range(len(price_data))
            ]
            ma2 = np.array(ma2)
        elif ma2_type.upper() == "EMA":
            alpha = 2 / (ma2_period + 1)
            ma2 = np.zeros_like(price_data)
            ma2[0] = price_data[0]
            for i in range(1, len(price_data)):
                ma2[i] = alpha * price_data[i] + (1 - alpha) * ma2[i - 1]
        else:
            raise ValueError("ma2_type must be 'SMA' or 'EMA'")

        ax.plot(time_points, price_data, label="Price", color="black", linewidth=1.2)
        ax.plot(
            time_points,
            ma1,
            label=f"{ma1_type.upper()} ({ma1_period})",
            color="blue",
            linewidth=1.5,
        )
        ax.plot(
            time_points,
            ma2,
            label=f"{ma2_type.upper()} ({ma2_period})",
            color="orange",
            linewidth=1.5,
        )
        ax.set_title(f"Instrument {instrument_idx}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.2)
        if idx % ncols == 0:
            ax.set_ylabel("Price")
        if idx >= (nrows - 1) * ncols:
            ax.set_xlabel("Time")
        stats_text = f"Cur: {price_data[-1]:.2f}\n{ma1_type.upper()}: {ma1[-1]:.2f}\n{ma2_type.upper()}: {ma2[-1]:.2f}"
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
        )
        if idx == 0:
            ax.legend(fontsize=8)

    # Hide unused axes
    for j in range(n_plots, nrows * ncols):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

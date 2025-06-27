import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from backtester import BackTester
import warnings

warnings.filterwarnings("ignore")

# Set style for better looking plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def backtest_strategy(
    data_file, num_test_days=100, strategy=None, strategy_params=None
):
    """
    Backtest a given strategy and return results for visualization

    Parameters:
    - data_file: Path to the price data file
    - num_test_days: Number of days to test
    - strategy: Strategy class to use for backtesting
    - strategy_params: Dictionary of strategy parameters

    Returns:
    - backtester: BackTester instance
    - daily_positions: Array of positions (nInst, days) - test period only
    - daily_pnl: List of daily portfolio P&L
    - stats: Dictionary of performance statistics
    - daily_stock_profits: Array of stock profits (nInst, days) - test period only
    - all_daily_positions: Array of positions (nInst, all_days) - including warm-up
    - all_daily_stock_profits: Array of stock profits (nInst, all_days) - including warm-up
    - price_data: Array of price data (nInst, all_days)
    """

    if strategy is None:
        raise ValueError("You must provide a strategy class to backtest_strategy.")

    # Initialize backtester
    backtester = BackTester(data_file)
    backtester.set_getPos(strategy)

    # Run backtest
    (
        daily_positions,
        daily_pnl,
        stats,
        daily_stock_profits,
        all_daily_positions,
        all_daily_stock_profits,
    ) = backtester.backtest(num_test_days)

    # Get price data for the entire period
    price_data = backtester.data

    return (
        backtester,
        daily_positions,
        daily_pnl,
        stats,
        daily_stock_profits,
        all_daily_positions,
        all_daily_stock_profits,
        price_data,
    )


def plot_instrument_analysis(
    all_daily_positions,
    daily_pnl,
    all_daily_stock_profits,
    price_data,
    num_instruments=5,
    save_plot=True,
    filename="instrument_analysis.png",
):
    """
    Create comprehensive visualization showing price, profit, and positions for each instrument

    Parameters:
    - all_daily_positions: Array of positions (nInst, all_days) - including warm-up
    - daily_pnl: List of daily portfolio P&L
    - all_daily_stock_profits: Array of stock profits (nInst, all_days) - including warm-up
    - price_data: Array of price data (nInst, all_days)
    - num_instruments: Number of instruments to display (default 5)
    - save_plot: Whether to save the plot (default True)
    - filename: Filename to save the plot (default 'instrument_analysis.png')
    """

    # Limit number of instruments for visualization
    num_instruments = min(num_instruments, price_data.shape[0])

    # Calculate number of rows needed (2 rows per instrument pair)
    num_rows = (num_instruments + 1) // 2 * 2  # Round up to nearest even number

    # Create figure with subplots (2 instruments per row, 2 rows per instrument pair)
    fig, axes = plt.subplots(num_rows, 2, figsize=(20, 4 * num_rows))

    # Create time axis for all days
    all_days = np.arange(1, price_data.shape[1] + 1)

    # Calculate test period start (for highlighting)
    test_start_day = len(all_daily_stock_profits[0]) - len(daily_pnl) + 1

    for i in range(num_instruments):
        # Calculate which row and column this instrument goes in
        row_start = (i // 2) * 2  # Each instrument pair takes 2 rows
        col = i % 2  # Left or right column

        # Calculate cumulative profit for this instrument
        cumulative_profit = np.cumsum(all_daily_stock_profits[i, :])

        # Plot 1: Cumulative Profit (all days) - top row
        axes[row_start, col].plot(
            all_days, cumulative_profit, "g-", linewidth=2, label=f"Instrument {i+1}"
        )
        axes[row_start, col].axhline(y=0, color="r", linestyle="--", alpha=0.5)
        axes[row_start, col].axvline(
            x=test_start_day,
            color="orange",
            linestyle="--",
            alpha=0.7,
            label="Test Start",
        )
        axes[row_start, col].set_title(
            f"Instrument {i+1} - Cumulative Profit (All Days)"
        )
        axes[row_start, col].set_ylabel("Cumulative Profit")
        axes[row_start, col].grid(True, alpha=0.3)
        axes[row_start, col].legend()

        # Plot 2: Price and Position (overlaid) - bottom row
        ax1 = axes[row_start + 1, col]  # Primary y-axis for price
        ax2 = ax1.twinx()  # Secondary y-axis for position

        # Plot price on primary y-axis
        line1 = ax1.plot(all_days, price_data[i, :], "b-", linewidth=2, label="Price")
        ax1.set_ylabel("Price", color="b")
        ax1.tick_params(axis="y", labelcolor="b")
        ax1.grid(True, alpha=0.3)

        # Plot position on secondary y-axis
        line2 = ax2.plot(
            all_days, all_daily_positions[i, :], "purple", linewidth=2, label="Position"
        )
        ax2.set_ylabel("Position", color="purple")
        ax2.tick_params(axis="y", labelcolor="purple")
        ax2.axhline(y=0, color="r", linestyle="--", alpha=0.5)
        ax2.axvline(x=test_start_day, color="orange", linestyle="--", alpha=0.7)

        # Set title and x-label
        ax1.set_title(f"Instrument {i+1} - Price and Position (All Days)")
        ax1.set_xlabel("Day")

        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper left")

    # Hide unused subplots if odd number of instruments
    if num_instruments % 2 == 1:
        # Hide the unused subplot in the last row
        last_row = num_rows - 2
        axes[last_row, 1].set_visible(False)
        axes[last_row + 1, 1].set_visible(False)

    plt.tight_layout()
    if save_plot:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


def plot_summary_statistics(
    daily_pnl,
    stats,
    all_daily_stock_profits,
    save_plot=True,
    filename="summary_statistics.png",
):
    """
    Create summary plots for overall performance

    Parameters:
    - daily_pnl: List of daily portfolio P&L
    - stats: Dictionary of performance statistics
    - all_daily_stock_profits: Array of stock profits (nInst, all_days) - including warm-up
    - save_plot: Whether to save the plot (default True)
    - filename: Filename to save the plot (default 'summary_statistics.png')
    """

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Cumulative Portfolio P&L
    cumulative_pnl = np.cumsum(daily_pnl)
    axes[0, 0].plot(cumulative_pnl, "b-", linewidth=2)
    axes[0, 0].set_title("Cumulative Portfolio P&L (Test Period)")
    axes[0, 0].set_ylabel("Cumulative P&L")
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Daily P&L Distribution
    axes[0, 1].hist(daily_pnl, bins=30, alpha=0.7, color="green", edgecolor="black")
    axes[0, 1].set_title("Daily P&L Distribution (Test Period)")
    axes[0, 1].set_xlabel("Daily P&L")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Cumulative Stock Profit Heatmap (all days)
    cumulative_profit_heatmap = np.cumsum(all_daily_stock_profits, axis=1)
    im = axes[1, 0].imshow(
        cumulative_profit_heatmap,
        aspect="auto",
        cmap="RdYlGn",
        vmin=-np.abs(cumulative_profit_heatmap).max(),
        vmax=np.abs(cumulative_profit_heatmap).max(),
    )
    axes[1, 0].set_title("Cumulative Stock Profit Heatmap (All Days)")
    axes[1, 0].set_xlabel("Day")
    axes[1, 0].set_ylabel("Instrument")
    plt.colorbar(im, ax=axes[1, 0])

    # Plot 4: Performance Statistics
    stats_text = f"""
    Mean P&L: {stats['meanPL']:.2f}
    Return: {stats['return']:.4f}
    Std Dev: {stats['stdDev']:.2f}
    Annual Sharpe: {stats['annualSharpe']:.2f}
    Total Volume: {stats['totalVolume']:.0f}
    Score: {stats['score']:.2f}
    """
    axes[1, 1].text(
        0.1,
        0.5,
        stats_text,
        transform=axes[1, 1].transAxes,
        fontsize=12,
        verticalalignment="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
    )
    axes[1, 1].set_title("Performance Statistics")
    axes[1, 1].axis("off")

    plt.tight_layout()
    if save_plot:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


def plot_single_instrument(
    all_daily_positions,
    all_daily_stock_profits,
    price_data,
    daily_pnl,
    instrument_idx=0,
    save_plot=False,
    filename="single_instrument.png",
):
    """
    Plot detailed analysis for a single instrument

    Parameters:
    - all_daily_positions: Array of positions (nInst, all_days) - including warm-up
    - all_daily_stock_profits: Array of stock profits (nInst, all_days) - including warm-up
    - price_data: Array of price data (nInst, all_days)
    - daily_pnl: List of daily portfolio P&L (for calculating test period start)
    - instrument_idx: Index of instrument to plot (default 0)
    - save_plot: Whether to save the plot (default False)
    - filename: Filename to save the plot (default 'single_instrument.png')
    """

    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    # Create time axis for all days
    all_days = np.arange(1, price_data.shape[1] + 1)

    # Calculate test period start (for highlighting)
    test_start_day = len(all_daily_stock_profits[0]) - len(daily_pnl) + 1

    # Calculate cumulative profit for this instrument
    cumulative_profit = np.cumsum(all_daily_stock_profits[instrument_idx, :])

    # Plot 1: Cumulative Profit (all days)
    axes[0].plot(
        all_days, cumulative_profit, "g-", linewidth=2, label="Cumulative Profit"
    )
    axes[0].axhline(y=0, color="r", linestyle="--", alpha=0.5)
    axes[0].axvline(
        x=test_start_day, color="orange", linestyle="--", alpha=0.7, label="Test Start"
    )
    axes[0].set_title(f"Instrument {instrument_idx+1} - Cumulative Profit (All Days)")
    axes[0].set_ylabel("Cumulative Profit")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Plot 2: Price and Position (overlaid) - all days
    ax1 = axes[1]  # Primary y-axis for price
    ax2 = ax1.twinx()  # Secondary y-axis for position

    # Plot price on primary y-axis
    line1 = ax1.plot(
        all_days, price_data[instrument_idx, :], "b-", linewidth=2, label="Price"
    )
    ax1.set_ylabel("Price", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.grid(True, alpha=0.3)

    # Plot position on secondary y-axis
    line2 = ax2.plot(
        all_days,
        all_daily_positions[instrument_idx, :],
        "purple",
        linewidth=2,
        label="Position",
    )
    ax2.set_ylabel("Position", color="purple")
    ax2.tick_params(axis="y", labelcolor="purple")
    ax2.axhline(y=0, color="r", linestyle="--", alpha=0.5)
    ax2.axvline(x=test_start_day, color="orange", linestyle="--", alpha=0.7)

    # Set title and x-label
    ax1.set_title(f"Instrument {instrument_idx+1} - Price and Position (All Days)")
    ax1.set_xlabel("Day")

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left")

    plt.tight_layout()
    if save_plot:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


def print_performance_summary(stats, strategy_params=None):
    """
    Print a formatted performance summary

    Parameters:
    - stats: Dictionary of performance statistics
    - strategy_params: Dictionary of strategy parameters (optional)
    """

    print("=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)

    if strategy_params:
        print(f"\nStrategy Parameters:")
        for key, value in strategy_params.items():
            print(f"  {key}: {value}")

    print(f"\nPerformance Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")


def get_default_strategy_params():
    """
    Get default parameters for mean reversion strategy

    Returns:
    - Dictionary of default parameters
    """
    return {
        "lookback_period": 14,
        "standard_deviations": 2,
        "max_look_back": 110,
        "capital_allocation": 7000,
        "stop_loss_pct": 0.05,  # 5% stop loss
    }


def run_complete_analysis(
    data_file="prices_train.txt",
    num_test_days=100,
    strategy_params=None,
    num_instruments=5,
    save_plots=True,
):
    """
    Complete analysis pipeline for mean reversion strategy

    Parameters:
    - data_file: Path to price data file
    - num_test_days: Number of days to test
    - strategy_params: Strategy parameters dictionary
    - num_instruments: Number of instruments to visualize
    - save_plots: Whether to save plots (default True)

    Returns:
    - Dictionary containing all results
    """

    # Use default parameters if none provided
    if strategy_params is None:
        strategy_params = get_default_strategy_params()

    # Run backtest
    (
        backtester,
        daily_positions,
        daily_pnl,
        stats,
        daily_stock_profits,
        all_daily_positions,
        all_daily_stock_profits,
        price_data,
    ) = backtest_strategy(data_file, num_test_days, strategy_params=strategy_params)

    # Print summary
    print_performance_summary(stats, strategy_params)

    # Create visualizations
    print(f"\nCreating visualizations...")

    # Plot instrument analysis
    plot_instrument_analysis(
        all_daily_positions,
        daily_pnl,
        all_daily_stock_profits,
        price_data,
        num_instruments,
        save_plots,
        "instrument_analysis.png",
    )

    # Plot summary statistics
    plot_summary_statistics(
        daily_pnl, stats, all_daily_stock_profits, save_plots, "summary_statistics.png"
    )

    # Return all results in a dictionary
    results = {
        "backtester": backtester,
        "daily_positions": daily_positions,
        "daily_pnl": daily_pnl,
        "stats": stats,
        "daily_stock_profits": daily_stock_profits,
        "all_daily_positions": all_daily_positions,
        "all_daily_stock_profits": all_daily_stock_profits,
        "price_data": price_data,
        "strategy_params": strategy_params,
    }

    return results


if __name__ == "__main__":
    # Example usage with different parameter sets
    print("Running Mean Reversion Strategy Analysis...")

    # Default parameters
    default_params = {
        "lookback_period": 14,
        "standard_deviations": 2,
        "max_look_back": 110,
        "capital_allocation": 7000,
    }

    # Run analysis
    results = run_complete_analysis(
        data_file="prices_train.txt",
        num_test_days=100,
        strategy_params=default_params,
        num_instruments=5,
        save_plots=True,
    )

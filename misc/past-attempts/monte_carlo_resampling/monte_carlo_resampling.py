import numpy as np
import pandas as pd
import yfinance as yf

from misc.utils.ticker_fetcher import fetch_ticker_data
from misc.utils.monthly_returns import calculate_monthly_returns
from misc.utils.yearly_returns import calculate_yearly_returns
from misc.utils.load_assets import load_assets
from misc.utils.optimize_weights_monte_carlo import optimize_weights_monte_carlo


# =============================================================================
# PRINTING HELPERS
# =============================================================================
def print_header(title: str, width: int = 60):
    """Print a section header."""
    print(f"\n{'=' * width}")
    print(f" {title}")
    print(f"{'=' * width}")


def print_subheader(title: str, width: int = 60):
    """Print a subsection header."""
    print(f"\n{'-' * width}")
    print(f" {title}")
    print(f"{'-' * width}")


def print_key_value(key: str, value, indent: int = 2):
    """Print a key-value pair with consistent formatting."""
    print(f"{' ' * indent}{key}: {value}")


def print_table_row(columns: list, widths: list):
    """Print a formatted table row."""
    row = ""
    for col, width in zip(columns, widths):
        row += f"{str(col):>{width}}  "
    print(f"  {row}")


def run(
    shrinkage_intensity: float = 0.7,
    period: str = "10y",
    num_simulations: int = 500,
    num_target_returns: int = 10,
):
    """
    Run Monte Carlo resampling portfolio optimization.

    Args:
        shrinkage_intensity: Weight for shrinkage target vs sample mean (0-1)
        period: Data period for fetching ticker data (e.g., "5y", "10y")
        num_simulations: Number of Monte Carlo simulations
        num_target_returns: Number of target return levels for efficient frontier
    """
    # =========================================================================
    # DATA LOADING
    # =========================================================================
    tickers = load_assets()

    data = [fetch_ticker_data(ticker, period=period) for ticker in tickers]

    monthly_returns = [calculate_monthly_returns(d) for d in data]

    yearly_returns = [calculate_yearly_returns(d) for d in data]

    # =========================================================================
    # EXPECTED RETURNS ESTIMATION (SHRINKAGE)
    # =========================================================================
    mean_sample = np.mean(yearly_returns, axis=1)
    mean_target = np.divide(mean_sample, 2.2)

    mean_shrunk = (
        shrinkage_intensity * mean_target + (1 - shrinkage_intensity) * mean_sample
    ).squeeze()

    # =========================================================================
    # COVARIANCE MATRIX
    # =========================================================================
    yearly_returns_array = np.array([df.values.flatten() for df in yearly_returns])

    T = yearly_returns_array.shape[1]
    cov_matrix = np.cov(yearly_returns_array)
    estimation_uncertainty = cov_matrix / T

    # =========================================================================
    # TARGET RETURNS (EFFICIENT FRONTIER)
    # =========================================================================
    min_target_return = float(np.min(mean_shrunk))
    max_target_return = float(np.max(mean_shrunk))
    target_return = np.linspace(
        min_target_return, max_target_return, num_target_returns
    )

    # =========================================================================
    # MONTE CARLO OPTIMIZATION
    # =========================================================================
    weight_storage = np.zeros((num_simulations, len(target_return), len(tickers)))

    print_header("MONTE CARLO PORTFOLIO OPTIMIZATION")
    print_key_value("Assets", ", ".join(tickers))
    print_key_value("Period", period)
    print_key_value("Shrinkage Intensity", f"{shrinkage_intensity:.1%}")
    print_key_value("Simulations", num_simulations)
    print_key_value("Target Return Levels", num_target_returns)

    print_subheader("Running Simulations")
    for i in range(num_simulations):
        sampled_returns = np.random.multivariate_normal(
            mean_shrunk, estimation_uncertainty
        )
        for k, selected_target_return in enumerate(target_return):
            optimal_weights = optimize_weights_monte_carlo(
                sampled_returns, selected_target_return, cov_matrix
            )
            weight_storage[i, k, :] = optimal_weights

        # Progress indicator
        if (i + 1) % 100 == 0 or i == num_simulations - 1:
            print(f"  Progress: {i + 1:>5} / {num_simulations} simulations")

    # =========================================================================
    # RESULTS
    # =========================================================================
    weight_storage_avg = np.mean(weight_storage, axis=0)

    print_subheader("Efficient Frontier Results")

    # Table header
    ticker_width = max(len(t) for t in tickers)
    col_widths = [8] + [ticker_width + 2] * len(tickers) + [10]
    header = ["Return"] + tickers + ["Volatility"]
    print_table_row(header, col_widths)
    print(f"  {'-' * sum(w + 2 for w in col_widths)}")

    # Table rows
    for i, return_level in enumerate(target_return):
        volatility = np.sqrt(
            np.dot(weight_storage_avg[i].T, np.dot(cov_matrix, weight_storage_avg[i]))
        )
        weights_pct = [f"{w * 100:.1f}%" for w in weight_storage_avg[i]]
        row = [f"{return_level * 100:.2f}%"] + weights_pct + [f"{volatility * 100:.2f}%"]
        print_table_row(row, col_widths)

    print_header("OPTIMIZATION COMPLETE")

    return weight_storage_avg, target_return, tickers, cov_matrix

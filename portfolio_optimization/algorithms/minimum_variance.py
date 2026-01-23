"""Minimum Variance Portfolio Optimization."""

import numpy as np

from portfolio_optimization.algorithms.base import BaseOptimizer, OptimizationResult
from portfolio_optimization.utils.solvers import (
    minimize_variance_portfolio,
    minimize_volatility_portfolio,
)
from portfolio_optimization.utils.formatting import (
    print_header,
    print_subheader,
    print_key_value,
    print_results,
)


class MinimumVariance(BaseOptimizer):
    """
    Minimum Variance portfolio optimizer.

    Finds the portfolio that minimizes the portfolio variance.
    """

    name = "minimum_variance"
    description = "Minimum Variance portfolio optimization"

    def optimize(
        self,
        verbose: bool = True,
        **kwargs,
    ) -> OptimizationResult:
        """
        Find the minimum variance portfolio.

        Args:
            verbose: If True, print results

        Returns:
            OptimizationResult with the single minimum variance portfolio
        """
        self._ensure_data_loaded()

        if verbose:
            print_header("MINIMUM VARIANCE OPTIMIZATION")
            print_key_value("Algorithm", self.description)
            print_key_value("Assets", ", ".join(self.tickers))
            print_key_value("Period", self.period)
            print_subheader("Computing Minimum Variance Portfolio")

        # Find the global minimum variance portfolio
        weights = minimize_volatility_portfolio(
            covariance_matrix=self.covariance_matrix,
            allow_short=False,
        )

        # Calculate portfolio volatility
        volatility = self.calculate_portfolio_volatility(weights)

        # Calculate expected return using historical average
        expected_returns = np.array([
            df.values.mean() for df in self.yearly_returns
        ])
        portfolio_return = float(np.sum(weights * expected_returns))

        result = OptimizationResult(
            weights=weights.reshape(1, -1),  # Shape: (1, num_assets)
            expected_returns=np.array([portfolio_return]),
            volatilities=np.array([volatility]),
            tickers=self.tickers,
            covariance_matrix=self.covariance_matrix,
            algorithm_name=self.name,
            metadata={"historical_returns": expected_returns},
        )

        if verbose:
            print_results(result)
            print_header("OPTIMIZATION COMPLETE")

        return result

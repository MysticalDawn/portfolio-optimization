import numpy as np
import scipy.optimize as optimize


def optimize_weights_monte_carlo(
    sample_returns: np.ndarray, target_return: float, covariance_matrix: np.ndarray
) -> np.ndarray:
    """
    For a given target return, find the minimum volatility portfolio that achieves that target return.
    Args:
        sample_returns: The returns of the portfolio.
        target_return: The target return of the portfolio.
    Returns:
        The weights of the portfolio.
    """

    def objective(weights):
        portfolio_variance = weights.T @ covariance_matrix @ weights
        return portfolio_variance

    constraint = [
        {"type": "eq", "fun": lambda weights: np.sum(weights) - 1},
        {
            "type": "eq",
            "fun": lambda weights: np.sum(weights * sample_returns) - target_return,
        },
    ]
    bounds = tuple((0, 1) for _ in range(len(sample_returns)))
    initial_guess = np.ones(len(sample_returns)) / len(sample_returns)
    result = optimize.minimize(
        objective,
        initial_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraint,
    )
    return result.x

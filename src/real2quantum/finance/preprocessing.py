import numpy as np

def calculate_portfolio_parameters(prices, annualize=True):
    """
    Calculate historical returns, expected returns (mu), and covariance matrix (Sigma).

    Parameters
    ----------
    prices : np.ndarray
        Array of closing prices with shape (days, assets).
        Each row is a trading day, each column is an asset.

    annualize : bool
        If True, convert daily statistics to annual statistics
        using 252 trading days/year.

    Returns
    -------
    mu : np.ndarray
        Expected return vector.

    Sigma : np.ndarray
        Covariance matrix.
    """

    # Calculate daily returns
    returns = prices[1:] / prices[:-1] - 1

    # Mean daily returns
    mu = np.mean(returns, axis=0)

    # Covariance matrix
    Sigma = np.cov(returns, rowvar=False)

    if annualize:
        trading_days = 252
        mu = trading_days * mu
        Sigma = trading_days * Sigma

    return mu, Sigma

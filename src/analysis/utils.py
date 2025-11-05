import pandas as pd


def compute_car(stock_returns: pd.Series, event_index: int, window: int = 5) -> float:
    """Calculate the Cummulative Abnormal Returns (CAR) for a given asset.

    Parameters
    ----------
    `stock_returns : pd.Series`
        Time series of asset abnormal returns
    `event_index : int`
        Integer index of event to calculate CAR about
    `window : int`
        Integer representing how many indices the CAR should be computed over

    Returns
    -------
    `car : float`
        The cummulative abonormal return (CAR) for the underlying asset
    """
    return stock_returns.iloc[event_index : event_index + window + 1].sum()


def rolling_residual(
    Y: pd.DataFrame, X: pd.Series, window: int = 35, expanding: bool = True
) -> pd.Series:
    """Perform a rolling OLS regression on X against Y with a constant, and
    return the residual of the regression.

    Parameters
    ----------
    `X : pd.Series`
        Time series of asset returns to regress Y on
    `Y : int`
        Time series of target returns for the regression  of X on Y to predict
    `window : int`
        Integer representing the sliding window the regression should be performed on
    `expanding : boolean`
        Boolean indicating if we want to enable expanding argument into
        `statsmodels.regression.RollingOLS` object

    Returns
    -------
    `residual : pd.Series`
        The residual of the rolling regression of X on Y
    """

    import statsmodels.api as sm
    from statsmodels.regression.rolling import RollingOLS

    X = sm.add_constant(X)

    reg = RollingOLS(Y, X, window=window, expanding=expanding).fit()

    alpha = reg.params["const"].bfill()
    beta = reg.params[X.columns[-1]].bfill()

    residual = Y - (X[X.columns[-1]] * beta) - alpha

    return residual

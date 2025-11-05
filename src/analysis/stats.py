import pandas as pd
import scipy.stats as stats

from src.analysis.utils import compute_car


def random_sample(
    stock_returns: pd.Series, n: int = 50, filtered_idx: list = []
) -> pd.Series:
    """Sample randomly from asset return data by date, with option to filter
    out certain dates.

    Parameters
    ----------
    `stock_returns : pd.Series`
        Time series representing asset of interest's returns
    `n : int`
        Number of random samples we want to take
    `filtered_idx : list`
        List of date indices to ignore when we sample from `stock_returns`

    Returns
    -------
    `sampled_stock_returns : pd.Series`
        Randomly sampled indcies from the `stock_returns` input
    """

    if not filtered_idx:
        return stock_returns.sample(n)
    return stock_returns.loc[filtered_idx].sample(n)


def single_sample_test(
    returns: pd.Series,
    event_indices: list,
    windows: list,
    test_mean: float = 0.0,
) -> pd.DataFrame:
    """Perform a single-sample t-test of CARs over different windows for
    multiple events, against a provided population mean.

    Parameters
    ----------
    `returns : pd.Series`
        Timer series of abnormal returns for an underlying asset
    `event_indices : list`
        List of time series indices of events of interest to calculate CARs at
    `windows : list`
        List of windows to calculate the CAR for
    `test_mean : float`
        Provided population mean to pass as comparison in single-sample t-test

    Returns
    -------
    `results : pd.DataFrame`
        DataFrame containing the results of each t-test for each window, with the
        mean CAR, the standard deviation of the CAR, number of CAR samples, the
        t_stat from the test, and the p-value of the test.
    """

    results = pd.DataFrame(
        index=windows, columns=["Mean", "Std", "N", "t_stat", "p_value"]
    )

    for window in windows:
        cars = []
        for date in event_indices:
            if date in returns.index:
                idx = returns.index.get_loc(date)
                car = compute_car(returns, idx, window=window)
                cars.append(car)

        cars = pd.Series(cars).dropna()

        if len(cars) == 0:
            continue

        t_stat, p_val = stats.ttest_1samp(cars, test_mean)

        results.loc[window, "Mean"] = cars.mean()
        results.loc[window, "Std"] = cars.std()
        results.loc[window, "N"] = len(cars)
        results.loc[window, "t_stat"] = t_stat
        results.loc[window, "p_value"] = p_val

    return results


def two_sample_test(
    returns: pd.Series,
    event_indices: list,
    compare_events: list,
    windows: list,
) -> pd.DataFrame:
    """Perform a two-sample t-test of CARs over different windows for multiple
    events.

    Parameters
    ----------
    `returns : pd.Series`
        Timer series of abnormal returns for an underlying asset
    `event_indices : list`
        List of time series indices of events of interest to calculate CARs at
    `compare_returns : pd.Series`
        Pandas series of events to act as null comparison in two-sample test
        (usually random)
    `windows : list`
        List of windows to calculate the CAR for

    Returns
    -------
    `results : pd.DataFrame`
        DataFrame containing the results of each t-test for each window, with the
        ean CAR, the standard deviation of the CAR, number of CAR samples, the
        t_stat from the test, and the p-value of the test.
    """

    results = pd.DataFrame(
        index=windows, columns=["Mean", "Std", "N", "t_stat", "p_value"]
    )

    event_cars = pd.DataFrame(columns=windows)

    for window in windows:
        cars = []
        for event_date in event_indices:
            if event_date in returns.index:
                idx = returns.index.get_loc(event_date)
                car = compute_car(returns, idx, window=window)
                cars.append(car)
        event_cars[window] = cars

    for window in windows:
        cars = []
        for date in compare_events:
            idx = returns.index.get_loc(date)
            car = compute_car(returns, idx, window=window)
            cars.append(car)

        compare_cars = pd.Series(cars).dropna()

        if len(compare_cars) == 0:
            continue

        t_stat, p_val = stats.ttest_ind(event_cars[window], compare_cars)

        results.loc[window, "Mean"] = compare_cars.mean()
        results.loc[window, "Std"] = compare_cars.std()
        results.loc[window, "N"] = len(compare_cars)
        results.loc[window, "t_stat"] = t_stat
        results.loc[window, "p_value"] = p_val

    return results

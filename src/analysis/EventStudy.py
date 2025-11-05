import pandas as pd

from src.analysis.stats import random_sample, single_sample_test, two_sample_test
from src.analysis.utils import rolling_residual


class EventStudy:
    def __init__(
        self,
        ticker: str,
        market_returns: pd.Series,
        car_windows: list = [1, 5, 10, 20, 30],
    ) -> None:
        self.ticker = ticker
        self.market_returns = market_returns
        self.car_windows = car_windows
        self.residuals = None

    def fit_residuals(
        self,
        stock_returns: pd.Series,
        window: int = 35,
        expanding: bool = True,
    ) -> pd.Series:
        self.residuals = rolling_residual(
            stock_returns, self.market_returns, window, expanding
        )
        return self

    def event_single_test(
        self, event_dates: list, pop_mean: float = 0.0
    ) -> pd.DataFrame:
        results = single_sample_test(
            self.residuals, event_dates, self.car_windows, pop_mean
        )
        return results

    def random_single_test(
        self,
        filtered_dates: list = [],
        n: int = 50,
        M: int = 1000,
        pop_mean: float = 0.0,
    ) -> pd.DataFrame:
        random_results = pd.DataFrame(
            index=self.car_windows, columns=["t{}".format(i) for i in range(M)]
        )
        for i in range(M):
            random_residuals = random_sample(
                self.residuals, n=n, filtered_idx=filtered_dates
            )
            results = single_sample_test(
                self.residuals,
                random_residuals.index,
                self.car_windows,
                pop_mean,
            )

            random_results.iloc[:, i] = results["t_stat"]

        mean_random = random_results.mean(axis=1).astype(float)
        std_random = random_results.std(axis=1).astype(float)

        return mean_random, std_random

    def event_two_sample_test(
        self,
        event_dates: list,
        filtered_dates: list = [],
        n: int = 50,
        M: int = 1000,
    ) -> pd.DataFrame:
        two_sample_results = pd.DataFrame(
            index=self.car_windows, columns=["t{}".format(i) for i in range(M)]
        )
        for i in range(M):
            random_residuals = random_sample(
                self.residuals, n=n, filtered_idx=filtered_dates
            )
            results = two_sample_test(
                self.residuals, event_dates, random_residuals.index, self.car_windows
            )

            two_sample_results.iloc[:, i] = results["t_stat"]

        mean_two_sample = two_sample_results.mean(axis=1).astype(float)
        std_two_sample = two_sample_results.std(axis=1).astype(float)

        return mean_two_sample, std_two_sample

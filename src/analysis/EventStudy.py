import pandas as pd

from src.analysis.stats import random_sample, single_sample_test, two_sample_test
from src.analysis.utils import rolling_residual


class EventStudy:
    """
    Class that implements event driven returns study using CARs and t-tests.

    Attributes
    ----------
    ticker : str
        String representing the ticker of the asset of the study
    market_returns : pd.Series
        Time series of our market index's returns
    car_windows : list[int]
        List of windows to calculate CAR over for the tests
    residuals : pd.Series
        Time series of the residuals of regressing the asset being tested's
        returns on `market_return`, representing abnormal returns

    """

    def __init__(
        self,
        ticker: str,
        market_returns: pd.Series,
        car_windows: list[int] = [1, 5, 10, 20, 30],
    ) -> None:
        """
        Constructor method to build EventStudy object with proper
        instance attributes

        Parameters
        ----------
        ticker : str
            String representing the ticker of the asset of the study
        market_returns : pd.Series
            Time series of our market index's returns
        car_windows : list[int]
            List of windows to calculate CAR over for the tests

        Returns
        -------
        None
        """
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
        """
        Method to perform rolling regression of asset returns on
        `self.market_returns`, and set the `self.residuals` attribute
        with the resulting regression residuals

        Parameters
        ----------
        stock_returns : pd.Series
            Time series of the returns of the asset of the study (`self.ticker`)
        window : int
            Window to run the rolling regression over
        expanding : bool
            Boolean flag indicating if you want expanding to be turned on for the
            rolling regressing

        Returns
        -------
        None
        """
        self.residuals = rolling_residual(
            stock_returns, self.market_returns, window, expanding
        )
        return self

    def event_single_test(
        self, event_dates: list[pd.Timestamp], pop_mean: float = 0.0
    ) -> pd.DataFrame:
        """
        Method to run a single-sample t-test, using the calculated
        abnormal returns (`self.residuals`) and a provided population
        mean

        Parameters
        ----------
        event_dates : list[pd.Timestamp]
            List of event dates to calculate CARs for
        pop_mean : float
            Float representing the population mean as the comparison value
            in the t-test

        Returns
        -------
        results : pd.DataFrame
            DataFrame containing resulting statistics and scores from the
            single-sample t-test
        """

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
        replacement: bool = False,
    ) -> pd.DataFrame:
        """
        Method to run a single-sample t-test on pseudo event dates,
        using the calculated abnormal returns (`self.residuals`) and
        a provided population mean, and return aggregated test results
        over the trials run

        Parameters
        ----------
        filtered_dates : list[pd.Timestamp]
            List of pseudo dates to sample randomly from
        n : int
            Number of samples to take per trial
        M : int
            Number of trials to run
        pop_mean : float
            Float representing the population mean as the comparison value
            in the t-test
        replacement : bool
            Boolean flag indicating if random sampling shold be done with
            replacement

        Returns
        -------
        mean_random : float
            Aggregate mean of t-score from `M` single-sample t-tests of randomly
            sampled pseudo-events
        std_random : float
            Aggregate standard deviation of t-score from `M` single-sample t-tests
            of randomly sampled pseudo-events
        """
        random_results = pd.DataFrame(
            index=self.car_windows, columns=["t{}".format(i) for i in range(M)]
        )
        for i in range(M):
            random_residuals = random_sample(
                self.residuals,
                n=n,
                filtered_idx=filtered_dates,
                replacement=replacement,
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
        replacement: bool = False,
    ) -> pd.DataFrame:
        """
        Method to run a two-sample t-test, comparing CAR of event dates
        to CAR of pseudo event dates using the calculated abnormal returns
        (`self.residuals`), and return
        aggregated test results over the trials run

        Parameters
        ----------
        event_dates : list[pd.Timestamp]
            List of event dates to calculate CARs for
        filtered_dates : list[pd.Timestamp]
            List of pseudo dates to sample randomly from
        n : int
            Number of samples to take per trial
        M : int
            Number of trials to run
        replacement : bool
            Boolean flag indicating if random sampling shold be done with
            replacement

        Returns
        -------
        mean_random : float
            Aggregate mean of t-score from `M` single-sample t-tests of randomly
            sampled pseudo-events
        std_random : float
            Aggregate standard deviation of t-score from `M` single-sample t-tests
            of randomly sampled pseudo-events
        """

        two_sample_results = pd.DataFrame(
            index=self.car_windows, columns=["t{}".format(i) for i in range(M)]
        )
        for i in range(M):
            random_residuals = random_sample(
                self.residuals,
                n=n,
                filtered_idx=filtered_dates,
                replacement=replacement,
            )
            results = two_sample_test(
                self.residuals, event_dates, random_residuals.index, self.car_windows
            )

            two_sample_results.iloc[:, i] = results["t_stat"]

        mean_two_sample = two_sample_results.mean(axis=1).astype(float)
        std_two_sample = two_sample_results.std(axis=1).astype(float)

        return mean_two_sample, std_two_sample

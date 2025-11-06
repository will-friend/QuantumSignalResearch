import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme()


def plot_test_results(
    ticker: str,
    test_result: pd.DataFrame,
    random_result_mean: float = None,
    random_result_std: float = None,
    figsize: tuple[int] = (),
    single_test: bool = True,
    enable_plotly: bool = False,
) -> None:
    """
    Function to display result of event study tests. Allows for single
    sample and two-sample plots, display of random variable results on
    top of event driven results, as well as plotly plots if desired

    Parameters
    ----------
    ticker : str
        String representing the ticker of the asset being analyzed
    test_results : pd.DataFrame
        DataFrame with the results of teh event drive test
    random_result_mean : float
        Float representing the aggregate mean of random trials as null
        test results
    random_result_std : float
        Float representing the aggregate std of random trials as null
        test results
    figsize : tuple[int]
        Tuple representing the dimensions of figure to be plotted (for
        matplotlib use only)
    single_test : bool
        Boolean flag indicating if test being displayed is single sample
        or two-sample (mainly for creating figure title and legend labels)
    enable_plotly : bool
        Boolean flag indicating if you would like plot to be a `plotly` plot
        rather than a `matplotlib` plot

    Returns
    -------
    None
    """

    if not enable_plotly:
        if single_test:
            str_label = "Random"
        else:
            str_label = "Two-Sample"

        if figsize:
            fig, ax = plt.subplots(figsize=figsize)

        _ = plt.plot(test_result.index, test_result["t_stat"], label="Event T-Stat")
        if random_result_mean is not None:
            _ = plt.plot(
                test_result.index,
                random_result_mean,
                label=str_label + " Mean",
                color="orange",
            )
        if random_result_std is not None:
            _ = plt.fill_between(
                test_result.index,
                random_result_mean - 2 * random_result_std,
                random_result_mean + 2 * random_result_std,
                color="orange",
                alpha=0.2,
                label=str_label + " ±2σ",
            )

        if single_test:
            _ = plt.title(ticker + " CAR Single-Sample T-Test Results")
        else:
            _ = plt.title(ticker + " CAR Single-Sample and Two-Sample T-Test Results")

        _ = plt.legend()
        plt.show()

    else:
        import plotly.graph_objects as go

        if single_test:
            str_label = "Random Sample"
        else:
            str_label = "Two-Sample"

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=test_result.index,
                y=random_result_mean,
                mode="lines",
                name="Mean " + str_label + " T-Stat",
                line=dict(color="orange"),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=list(test_result.index) + list(test_result.index[::-1]),
                y=list(random_result_mean + 2 * random_result_std)
                + list((random_result_mean - 2 * random_result_std)[::-1]),
                fill="toself",
                fillcolor="rgba(255,165,0,0.2)",
                line=dict(color="rgba(255,165,0,0)"),
                hoverinfo="skip",
                showlegend=True,
                name=str_label + " ±2σ",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=test_result.index,
                y=test_result["t_stat"],
                mode="lines",
                name="Event Single Sample T-Stat",
                line=dict(color="blue", width=2),
            )
        )

        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.4)

        fig.update_layout(
            title=ticker + "Single T-Test: Event CARs vs Random CARs"
            if single_test
            else ticker + " Two-Sample T-Test and Single-Sample T-Test",
            xaxis_title="Window Size",
            yaxis_title="T-Statistic",
            template="plotly_white",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            width=850,
            height=550,
        )

        fig.show()

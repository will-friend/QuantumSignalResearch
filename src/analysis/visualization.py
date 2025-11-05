import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set()


def plot_test_results(
    ticker: str,
    test_result: pd.Series,
    random_result_mean: float = None,
    random_result_std: float = None,
    figsize: tuple = (),
    single_test: bool = True,
    enable_plotly: bool = False,
) -> None:
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

        # Mean line for random t-stats
        fig.add_trace(
            go.Scatter(
                x=test_result.index,
                y=random_result_mean,
                mode="lines",
                name="Mean " + str_label + " T-Stat",
                line=dict(color="orange"),
            )
        )

        # ±2σ confidence band
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

        # Event t-stats
        fig.add_trace(
            go.Scatter(
                x=test_result.index,
                y=test_result["t_stat"],
                mode="lines",
                name="Event Single Sample T-Stat",
                line=dict(color="blue", width=2),
            )
        )

        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.4)

        # Layout customization
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

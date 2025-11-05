import sys

import arxiv
import pandas as pd
import yfinance as yf


def get_returns(
    ticker: str,
    start_date: str,
    end_date: str,
    to_pickle: bool = False,
    file_name: str = None,
) -> pd.Series:
    """
    Get ticker data from yfinance, calculate returns, and return the returns time
    series data

    Parameters
    ----------
    `ticker : str`
        Ticker for the asset you wish to collect data on
    `start_date : str`
        Start date to start data collection from
    `end_date : str`
        End date to end data collection on
    `to_pickle : bool`
        Boolean indicating if user wishes to save return data to pickle file
    `file_name : str`
        Full filename (including path) to save pickle file to

    Returns
    -------
    `ticker_returns : pd.Series`
        Time series of the ticker's returns over inputed time horizon
    """

    ticker_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)

    ticker_returns = ticker_data["Close"][ticker].pct_change().fillna(0)

    if to_pickle:
        try:
            ticker_returns.to_pickle(file_name)
        except TypeError as e:
            print(e + " Please provide a file to save pickle to.")
            sys.exit(-1)

    return ticker_returns


def scrape_arxiv(
    query: str,
    start_date: str,
    max_results: int = 100,
    to_pickle: bool = False,
    file_name: str = None,
) -> pd.DataFrame:
    """
    Get publication data from arxiv based on user query and return dataframe
    with results

    Parameters
    ----------
    `query : str`
        String specifying arxiv query user wishes to make
    `start_date : str`
        Start date to start data collection from
    `max_results : int`
        maximum number of results the query should return (most recent to least)
    `to_pickle : bool`
        Boolean indicating if user wishes to save return data to pickle file
    `file_name : str`
        Full filename (including path) to save pickle file to

    Returns
    -------
    `arxiv_df : pd.Series`
        Dataframe with query result data for each publication collected
    """

    search = arxiv.Search(
        query="Quantum AND IonQ",
        max_results=120,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    papers = []
    for result in search.results():
        if pd.Timestamp(result.published.date()) >= start_date:
            papers.append(
                {
                    "title": result.title,
                    "authors": [a.name for a in result.authors],
                    "published": pd.Timestamp(result.published.date()),
                    "url": result.entry_id,
                }
            )

    arxiv_df = pd.DataFrame(papers)
    arxiv_df = arxiv_df.sort_values(by="published", ascending=True)
    arxiv_df = arxiv_df.set_index("published")

    if to_pickle:
        try:
            arxiv_df.to_pickle(file_name)
        except TypeError as e:
            print(e + " Please provide a file to save pickle to.")
            sys.exit(-1)

    return arxiv_df

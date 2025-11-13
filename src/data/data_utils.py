import sys
import time

import arxiv
import feedparser
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
    ticker : str
        Ticker for the asset you wish to collect data on
    start_date : str
        Start date to start data collection from
    end_date : str
        End date to end data collection on
    to_pickle : bool
        Boolean indicating if user wishes to save return data to pickle file
    file_name : str
        Full filename (including path) to save pickle file to

    Returns
    -------
    ticker_returns : pd.Series
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
    query : str
        String specifying arxiv query user wishes to make
    start_date : str
        Start date to start data collection from
    max_results : int
        maximum number of results the query should return (most recent to least)
    to_pickle : bool
        Boolean indicating if user wishes to save return data to pickle file
    file_name : str
        Full filename (including path) to save pickle file to

    Returns
    -------
    arxiv_df : pd.Series
        Dataframe with query result data for each publication collected
    """

    fetched = 0

    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        papers = []
        for result in search.results():
            if pd.Timestamp(result.published.date()) >= pd.Timestamp(start_date):
                papers.append(
                    {
                        "title": result.title,
                        "authors": [a.name for a in result.authors],
                        "published": pd.Timestamp(result.published.date()),
                        "url": result.entry_id,
                    }
                )
                fetched += 1
            if fetched >= max_results:
                break

    except arxiv.UnexpectedEmptyPageError:
        pass

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


def build_arxiv_query(query: str) -> str:
    """
    Prepare a valid arXiv search_query.

    Paramters
    ---------
    query : str
        String with the query to be made as if passing to the arxiv package

    Returns
    -------
    query : str
        Updated query string to match api url formatting
    """
    query = query.strip()
    if "AND" in query.upper():
        parts = [p.strip() for p in query.split("AND")]
        joined = "+AND+".join(
            f'"{p}"' if " " in p and not p.startswith('"') else p for p in parts
        )
        query = f"all:({joined})"
    else:
        query = f'all:"{query}"' if " " in query else f"all:{query}"
    return query


def scrape_arxiv_raw(
    query: str,
    start_year: int,
    end_year: int,
    max_results_per_year: int = 1000,
    batch_size: int = 200,
    delay: float = 0.8,
    save_data: bool = False,
    file_name: str = None,
) -> pd.DataFrame:
    """
    Query arXiv using the raw Atom API, building a safe query string from `query`,
    and iterating year-by-year to avoid server truncation.

    Parameters
    ----------
    query : str
        Human readable query string, e.g. "Quantum AND IBM Quantum"
    start_year, end_year : int
        Inclusive year range to fetch.
    max_results_per_year : int
        Safety cap per year (should be >= expected number for busy queries).
    batch_size : int
        Items per API request (<= 300 recommended).
    delay : float
        Seconds to sleep between requests.
    save_yearly : bool
        Whether to pickle each year's partial results to disk.
    yearly_path_template : str or None
        Template for year file names, e.g. "data/arxiv_{}_ibm.pkl" where {} will
        be year.

    Return
    ------
    df : pd.DataFrame
        DataFrame constaining publications found in specified inputted year range
    """
    base_url = "http://export.arxiv.org/api/query?"
    q = build_arxiv_query(query)

    all_results = []

    for year in range(start_year, end_year + 1):
        fetched = 0
        year_results = []

        while fetched < max_results_per_year:
            start = fetched
            url = (
                f"{base_url}search_query={q}"
                f"&start={start}"
                f"&max_results={batch_size}"
                f"&sortBy=submittedDate&sortOrder=descending"
            )

            feed = feedparser.parse(url)
            entries = feed.entries

            if not entries:
                break

            batch_kept = []
            for e in entries:
                try:
                    pub_date = pd.Timestamp(e.published).tz_localize(None).normalize()
                except Exception:
                    continue

                if pub_date.year == year:
                    batch_kept.append(
                        {
                            "title": e.title,
                            "authors": [a.name for a in getattr(e, "authors", [])],
                            "published": pub_date,
                            "url": e.id,
                            # keep other fields if desired:
                            # "summary": getattr(e, "summary", ""),
                            # "primary_category": e.tags[0]['term'] if
                            # getattr(e, 'tags', None) else None
                        }
                    )
                else:
                    continue

            year_results.extend(batch_kept)
            fetched += len(entries)

            if len(entries) < batch_size:
                break

            time.sleep(delay)

        all_results.extend(year_results)

    df = pd.DataFrame(all_results)
    if df.empty:
        print("No results collected.")
        return df

    df = df.drop_duplicates(subset="url")
    df = df.sort_values("published", ascending=True).set_index("published")

    if save_data:
        df.to_pickle(file_name)

    return df

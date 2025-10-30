import pandas as pd

def compute_car(stock_returns: pd.Series, event_index: int, window: int=5) -> float:
    return stock_returns[event_index:event_index+window+1].sum()
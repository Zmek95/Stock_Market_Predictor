import pandas as pd

def moving_avg(dataset: pd.DataFrame, feature: str, window_size='30D', min_periods=None) -> pd.Series:
    '''
    Calculate moving average for a column in a Pandas DataFrame
    '''
    moving_avg_series = dataset.groupby(['Symbol'])\
                        .rolling(window=window_size, on='Date', min_periods = min_periods)[feature]\
                        .mean().reset_index()[feature]
    return moving_avg_series

def rolling_med(dataset: pd.DataFrame, feature: str, window_size='30D', min_periods=None) -> pd.Series:
    '''
    Calculate moving average for a column in a Pandas DataFrame
    '''
    rolling_med_series =    dataset.groupby(['Symbol'])\
                            .rolling(window=window_size, on='Date', min_periods = min_periods)[feature]\
                            .median().reset_index()[feature]
    return rolling_med_series

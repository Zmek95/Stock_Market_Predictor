import os
import sys
from pathlib import Path
import pandas as pd
from pyprojroot import here as get_project_root
os.chdir(get_project_root())
sys.path.append(str(get_project_root()))

import data_processing.feature_engineering as fe


if __name__ == '__main__':
    path = get_project_root() / Path('data_acquisition/data')
    df = pd.read_parquet(path / 'combined_df.parquet')

    df['Date'] = pd.to_datetime(df['Date'])
    df['vol_moving_avg'] = fe.moving_avg(df, 'Volume', window_size='30D', min_periods= 5)
    df['adj_close_rolling_med'] = fe.rolling_med(df, 'Volume', window_size='30D', min_periods= 5)
    df = df.astype({'Date': 'string'})

    path = get_project_root() / Path('data_processing/data')
    df.to_parquet(path / 'feat_eng_df.parquet')

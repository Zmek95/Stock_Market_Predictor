# https://towardsdatascience.com/downloading-datasets-from-kaggle-for-your-ml-project-b9120d405ea4

import os
import string
import sys
import zipfile
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from pathlib import Path
from pyprojroot import here as get_project_root
os.chdir(get_project_root())
sys.path.append(str(get_project_root()))


def download_kaggle_dataset(dataset_url: string, path: Path) -> None:
    api = KaggleApi()
    api.authenticate()
    print('Downloading files...')
    api.dataset_download_files(dataset_url, path=path)
    print('Download complete!\nUnpacking archive...')

    with zipfile.ZipFile(path / 'stock-market-dataset.zip','r') as zip_ref:
        zip_ref.extractall(path=path)
    print('Extraction succesful!')


def combine_exchange_data(data_path: Path) -> pd.DataFrame:
    etf_path = data_path / 'etfs'
    stock_path = data_path / 'stocks'
    symbol_paths = []

    for directory in [etf_path, stock_path]:
        for file in directory.iterdir():
            if file.is_file():
                symbol_paths.append(file)
    
    symbol_dfs = []

    for symbol_path in symbol_paths:
        df = pd.read_csv(symbol_path)
        df['Symbol'] = symbol_path.name[:-4]
        symbol_dfs.append(df.copy(deep=True))

    df_combined_symbols = pd.concat(symbol_dfs, ignore_index=True)
    del symbol_dfs
    df_labels = pd.read_csv(data_path / 'symbols_valid_meta.csv')[['Symbol', 'Security Name']]
    
    return df_labels.merge(df_combined_symbols, how='left', on='Symbol')

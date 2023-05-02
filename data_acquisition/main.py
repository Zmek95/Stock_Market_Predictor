import os
import sys
import json
from pyprojroot import here as get_project_root
os.chdir(get_project_root())
sys.path.append(str(get_project_root()))
from pathlib import Path


if __name__ == '__main__':
    with open(get_project_root() / Path('data_acquisition/kaggle.json'), 'r') as f:
        kaggle_creds = json.load(f)

    os.environ['KAGGLE_USERNAME'] = kaggle_creds['username']
    os.environ['KAGGLE_KEY'] = kaggle_creds['key']
   
    from data_acquisition.data_extraction import download_kaggle_dataset, combine_exchange_data

    path=get_project_root() / Path('data_acquisition/data')
    dataset_url = 'jacksoncrow/stock-market-dataset'
    download_kaggle_dataset(dataset_url=dataset_url, path=path)
    combined_df = combine_exchange_data(path)
    combined_df.to_parquet(path / 'combined_df.parquet')


import os
import sys
import json
from pyprojroot import here as get_project_root
os.chdir(get_project_root())
sys.path.append(str(get_project_root()))
from pathlib import Path

import data_acquisition.config as cfg


def main():
    with open(cfg.KAGGLE_CREDS_PATH, 'r') as f:
        kaggle_creds = json.load(f)

    os.environ['KAGGLE_USERNAME'] = kaggle_creds['username']
    os.environ['KAGGLE_KEY'] = kaggle_creds['key']
   
    from data_acquisition.data_extraction import download_kaggle_dataset, combine_exchange_data

    path = cfg.COMBINED_DATA_PATH
    dataset_url = cfg.DATA_KAGGLE_URL
    download_kaggle_dataset(dataset_url=dataset_url, path=path)
    combined_df = combine_exchange_data(path)
    combined_df.to_parquet(path / 'combined_df.parquet')


if __name__ == '__main__':
    main()    


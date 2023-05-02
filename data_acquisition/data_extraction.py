# https://towardsdatascience.com/downloading-datasets-from-kaggle-for-your-ml-project-b9120d405ea4

import os
import string
import sys
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
from pathlib import Path
from pyprojroot import here as get_project_root
os.chdir(get_project_root())
sys.path.append(str(get_project_root()))


def download_kaggle_dataset(dataset_url: string, path: Path) -> None:
    api = KaggleApi()
    api.authenticate()
    print('Downloading files...')
    #api.dataset_download_files(dataset_url, path=path)
    print('Download complete!\nUnpacking archive...')

    with zipfile.ZipFile(path / 'stock-market-dataset.zip','r') as zip_ref:
        zip_ref.extractall(path=path)
    print('Extraction succesful!')

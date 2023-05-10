import os
import sys
from pathlib import Path
from pyprojroot import here as get_project_root
os.chdir(get_project_root())
sys.path.append(str(get_project_root()))

DATA_ARCHIVE_NAME = "stock-market-dataset.zip"
DATA_KAGGLE_URL = 'jacksoncrow/stock-market-dataset'
DTYPE_SETTER = {'Symbol': 'string', 'Security Name': 'string', 
                'Date': 'string', 'Volume': 'int64'}
SYMBOL_METADATA = 'symbols_valid_meta.csv'
METADATA_FEATURES = ['Symbol', 'Security Name']
KAGGLE_CREDS_PATH = get_project_root() / Path('data_acquisition/kaggle.json')
COMBINED_DATA_PATH = get_project_root() / Path('data_acquisition/data')

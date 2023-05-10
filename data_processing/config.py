import os
import sys
from pathlib import Path
from pyprojroot import here as get_project_root
os.chdir(get_project_root())
sys.path.append(str(get_project_root()))

MIN_PERIODS_ROLLING = 5
FEATURE_ENG_DATA_PATH = get_project_root() / Path('data_processing/data')
import os
import sys
from pathlib import Path
from pyprojroot import here as get_project_root
os.chdir(get_project_root())
sys.path.append(str(get_project_root()))

MODEL_FEATURES = ['vol_moving_avg', 'adj_close_rolling_med']
MODEL_TARGET = 'Volume'
TRAINED_MODELS_PATH = get_project_root() / 'model_training'
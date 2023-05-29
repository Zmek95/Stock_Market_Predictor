import os
import sys
from pathlib import Path
from pyprojroot import here as get_project_root
os.chdir(get_project_root())
sys.path.append(str(get_project_root()))

MODEL_FEATURES = ['vol_moving_avg', 'adj_close_rolling_med']
MODEL_TARGET = 'Volume'
TRAINED_MODELS_PATH = get_project_root() / 'model_training'
SGD_reg_params = {'alpha': 0.0001, 
                  'max_iter': 1000, 
                  'eta_0': 0.01, 
                  'power_t': 0.25}
SGD_reg_params_grid = {'sgdregressor__alpha': (0.001, 0.0001, 0.00001),
                       'sgdregressor__eta_0': (0.1, 0.01, 0.001),
                       'sgdregressor__power_t': (0.20, 0.25, 0.30)}
RF_reg_params = {'n_estimators': 100}
RF_reg_params_grid = {'randomforestregressor__n_estimators': (50, 100, 250, 500),
                      'randomforestregressor__min_samples_leaf': (1, 2, 4),
                      'randomforestregressor__min_samples_split': (2, 5, 10)}
grid_search_niter = 25
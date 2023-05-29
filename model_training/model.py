import pandas as pd
import numpy as np
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV

import model_training.config as cfg

def SGDregressor_train(X: np.array,y: np.array, hyperparameters=dict(), verbose=0, grid_search=False):
    '''
    hyperparameters of intrest : alpha, max_iter, eta0, power_t
    '''
    if grid_search:
        SGDreg_pipeline = make_pipeline(StandardScaler(),
                                        SGDRegressor(verbose=verbose))
        grid_search_reg = RandomizedSearchCV(estimator=SGDreg_pipeline, n_iter=cfg.grid_search_niter, n_jobs=3, verbose=verbose)
        grid_search_reg.fit(X, y)
        return grid_search_reg
    else:
        SGDreg_pipeline = make_pipeline(StandardScaler(),
                                        SGDRegressor(**hyperparameters, verbose=verbose))

        SGDreg_pipeline.fit(X, y)
        return SGDreg_pipeline


def RFregressor_train(X: np.array,y: np.array, hyperparameters=dict(), verbose=0, grid_search=False):
    '''
    hyperparameters of intrest : alpha, max_iter, eta0, power_t
    '''
    if grid_search:
        RF_reg = RandomForestRegressor(verbose=verbose)
        grid_search_reg = RandomizedSearchCV(estimator=RF_reg, n_iter=cfg.grid_search_niter, n_jobs=3, verbose=verbose)
        grid_search_reg.fit(X, y)
        return grid_search_reg
    else:
        RF_reg = RandomForestRegressor(**hyperparameters, verbose=verbose)

        RF_reg.fit(X, y)
        return RF_reg


def data_load(path, features: List[str], target: str, test_split=0.2) -> List: 
    df = pd.read_parquet(path)
    df = df.dropna()
    X = df.loc[:, features].values
    y = df[target].values
    print(y.shape)
    return train_test_split(X, y, test_size=test_split, random_state=42)



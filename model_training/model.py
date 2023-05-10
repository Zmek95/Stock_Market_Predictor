import pandas as pd
import numpy as np
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


def SGDregressor_train(X: np.array,y: np.array, hyperparameters=dict(), verbose=0):
    '''
    hyperparameters of intrest : alpha, max_iter, eta0, power_t
    '''
    SGDreg_pipeline = make_pipeline(StandardScaler(),
                                    SGDRegressor(**hyperparameters, verbose=verbose))

    SGDreg_pipeline.fit(X, y)
    return SGDreg_pipeline


def data_load(path, features: List[str], target: str, test_split=0.2) -> List: 
    df = pd.read_parquet(path)
    df = df.dropna()
    X = df.loc[:, features].values
    y = df[target].values
    print(y.shape)
    return train_test_split(X, y, test_size=test_split, random_state=42)



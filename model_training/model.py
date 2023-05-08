import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# alpha, max_iter, eta0, power_t
def SGDregressor_train(X: np.array,y: np.array, hyperparameters=dict(), verbose=0):
    SGDreg_pipeline = make_pipeline(StandardScaler(),
                                    SGDRegressor(**hyperparameters, verbose=verbose))

    SGDreg_pipeline.fit(X, y)
    return SGDreg_pipeline



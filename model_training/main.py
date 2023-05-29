import os
import sys
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pprint import pprint
from joblib import dump
from pyprojroot import here as get_project_root
os.chdir(get_project_root())
sys.path.append(str(get_project_root()))

# logging
import mlflow
from mlflow.sklearn import autolog
from logs.mlflow_log import fetch_logged_data

from model_training.model import SGDregressor_train, RFregressor_train, data_load
from data_processing.config import FEATURE_ENG_DATA_PATH
import model_training.config as cfg


def model_train(estimator_func, estimator_func_params=dict(), model_name: str = 'trained_model'):
    autolog()
    path = FEATURE_ENG_DATA_PATH / 'feat_eng_df.parquet'
    print('Creating test and train sets')
    X_train, X_test, y_train, y_test = data_load(path, cfg.MODEL_FEATURES,
                                                 cfg.MODEL_TARGET, test_split=0.2)
    print('Train and test sets created\nTraining model')
    model = estimator_func(X_train, y_train, hyperparameters=estimator_func_params, grid_search=True)
    print('Model trained')

    run_id = mlflow.last_active_run().info.run_id
    print("Logged data and model in run: {}".format(run_id))

    for key, data in fetch_logged_data(run_id).items():
        print("\n---------- logged {} ----------".format(key))
        pprint(data)
    
    path = cfg.TRAINED_MODELS_PATH
    dump(model, path / model_name + '.joblib')

    y_pred = model.predict(X_test)

    # Calculate performance metrics for model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test,y_pred, squared=False)
    r2 = r2_score(y_test,y_pred)
    print(f"MAE:{mae} ,\nMSE:{mse},\nRMSE:{rmse},\nR2:{r2}")
    autolog(disable=True)
    return (model_name + '_RMSE', rmse)


if __name__== '__main__':
    model_train(SGDregressor_train, cfg.SGD_reg_params_grid, model_name='SGDRegressor')
    model_train(RFregressor_train,cfg.RF_reg_params_grid,model_name='RFRegressor')

    
        

    

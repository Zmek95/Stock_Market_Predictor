import os
import sys
from pathlib import Path
import pandas as pd
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pprint import pprint
from joblib import dump
from pyprojroot import here as get_project_root
os.chdir(get_project_root())
sys.path.append(str(get_project_root()))

# logging
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.sklearn import autolog


from model_training.model import SGDregressor_train

def yield_artifacts(run_id, path=None):
    """Yield all artifacts in the specified run"""
    client = MlflowClient()
    for item in client.list_artifacts(run_id, path):
        if item.is_dir:
            yield from yield_artifacts(run_id, item.path)
        else:
            yield item.path

def fetch_logged_data(run_id):
    """Fetch params, metrics, tags, and artifacts in the specified run"""
    client = MlflowClient()
    data = client.get_run(run_id).data
    # Exclude system tags: https://www.mlflow.org/docs/latest/tracking.html#system-tags
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = list(yield_artifacts(run_id))
    return {
        "params": data.params,
        "metrics": data.metrics,
        "tags": tags,
        "artifacts": artifacts,
    }

def data_load(path, features: List[str], target: str, test_split=0.2) -> List: 
    df = pd.read_parquet(path)
    df = df.dropna()
    X = df.loc[:, features].values
    y = df[target].values
    print(y.shape)
    return train_test_split(X, y, test_size=test_split, random_state=42)



if __name__== '__main__':
    autolog()
    path = get_project_root() / Path('data_processing/data/feat_eng_df.parquet')
    print('Creating test and train sets')
    X_train, X_test, y_train, y_test = data_load(path, ['vol_moving_avg', 'adj_close_rolling_med'],
                                                 'Volume', test_split=0.2)
    print('Train and test sets created\nTraining model')
    model = SGDregressor_train(X_train, y_train)
    print('Model trained')

    run_id = mlflow.last_active_run().info.run_id
    print("Logged data and model in run: {}".format(run_id))

    for key, data in fetch_logged_data(run_id).items():
        print("\n---------- logged {} ----------".format(key))
        pprint(data)
    
    path = get_project_root() / 'model_training'
    dump(model, 'SGDRegressor.joblib')

    # Make predictions on test data
    y_pred = model.predict(X_test)

    # Calculate the Mean Absolute Error and Mean Squared Error
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test,y_pred)
    print(f"{mae} , {mse}, {r2}")

    
        

    

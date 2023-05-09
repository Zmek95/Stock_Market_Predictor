import numpy as np
from joblib import load
from flask import Flask, request

app = Flask(__name__)


@app.route("/")
def greeting():
    return "<p>Stock market predictor</p>"


@app.route('/predict')
def predict():
    model = load('model_training/SGDRegressor.joblib')
    features = request.args
    try:
        vol_moving_avg = int(features['vol_moving_avg'])
        adj_close_rolling_med = int(features['adj_close_rolling_med'])
        if  (vol_moving_avg < 0 or adj_close_rolling_med < 0):
           return "<p>key values passed must be positive integers</p>" 
    except KeyError:
        return "<p>Must pass correct key values: vol_moving_avg, adj_close_rolling_med</p>"
    except ValueError:
        return "<p>key values passed must be integers</p>"
    
    prediction = int(model.predict(np.array([vol_moving_avg, adj_close_rolling_med]).reshape((1,-1)))[0])
    
    return f"<p>The predicted trading volume is {prediction}</p>"


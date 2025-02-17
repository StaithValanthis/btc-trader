import logging
from joblib import load, dump
import subprocess
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

MODEL_FILENAME_SGD = "sgd_model.joblib"
MODEL_FILENAME_RF = "rf_model.joblib"
SCALER_FILENAME = "ensemble_scaler.save"

def load_ensemble():
    models = {}
    try:
        models["sgd"] = load(MODEL_FILENAME_SGD)
        models["rf"] = load(MODEL_FILENAME_RF)
        scaler = load(SCALER_FILENAME)
        models["scaler"] = scaler
        logging.info("Ensemble models and scaler loaded.")
        return models
    except Exception as e:
        logging.error("Error loading ensemble models: %s", e)
        return None

def partial_update_ensemble(X_new, y_new, models):
    try:
        models["sgd"].partial_fit(X_new, y_new)
        logging.info("SGD model updated incrementally.")
    except Exception as e:
        logging.error("Error in partial_fit: %s", e)

def ensemble_predict(X, models):
    pred_sgd = models["sgd"].predict(X)
    pred_rf = models["rf"].predict(X)
    return (pred_sgd + pred_rf) / 2

def retrain_ensemble():
    logging.info("Retraining ensemble models using hyperparameter optimization (stub)...")
    # Here, you might integrate Hyperopt or Optuna. For now, we simply call the training script.
    subprocess.run(["python", "train_model.py"], check=True)
    logging.info("Ensemble retraining complete.")

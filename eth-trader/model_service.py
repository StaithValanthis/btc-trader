from flask import Flask, request, jsonify
from ml_model import load_ensemble, ensemble_predict
import numpy as np
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

models = load_ensemble()
if models is None:
    app.logger.error("Ensemble models not available.")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array(data.get("features"))
        preds = ensemble_predict(features, models)
        return jsonify({"prediction": preds.tolist()})
    except Exception as e:
        app.logger.error("Prediction error: %s", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

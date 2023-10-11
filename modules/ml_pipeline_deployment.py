"""ML Pipeline to Predict Call Worthiness deployed as a REST service

The module implements the deployment of the ML pipeline as a REST service.

"""

import os
from flask import Flask, request, jsonify
from ml_pipeline import predict_call_worthiness
import pandas as pd


app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get data from the request
        data = request.get_json()
        # to pandas dataframe
        data = pd.DataFrame(data)

        # Make predictions using the model
        predictions = predict_call_worthiness(
            data,
            model_path=os.path.join(os.getcwd(), "models/best_rf_classifier.sav"),
            threshold_path=os.path.join(os.getcwd(), "models/best_threshold.json"),
            ohe_path=os.path.join(os.getcwd(), "models/ohe_encoder.pkl"),
        )

        # Prepare the response
        response = predictions.to_json()

        return response

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

"""ML Pipeline to Predict Call Worthiness Deployed as a REST service

The deployment of the ML pipeline as a REST service.

"""

import os
from flask import Flask, request, jsonify
from ml_pipeline import Predictor
import pandas as pd


app = Flask(__name__)

predictor = Predictor()


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get data from the request
        data = request.get_json()
        data = pd.DataFrame(data)

        # Make predictions using the model
        predictions = predictor.predict_call_worthiness(data)

        # Prepare the response
        response = predictions.to_json()

        return response

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

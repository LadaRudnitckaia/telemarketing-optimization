"""ML Pipeline to Predict Call Worthiness

The module implements the function that preprocesses the input data and applies the model to classify clients as worth calling to or not.

"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import json
import pickle


def predict_call_worthiness(
    input_data: pd.DataFrame, model_path: str, threshold_path: str, ohe_path: str
) -> pd.DataFrame:
    """The function preprocesses the input Pandas dataframe as follows:
    - filters the required features
    - applies one-hot-encoding to categorical features
    - Missing values in categorical columns are treated as separate categories.
    - Normalization is not needed for a Random Forest model.
    Next, it applies the model to the preprocessed data to calculate the predictions.

    Args:
        input_data (pd.DataFrame): Input data that contains all the necessary features
        model_path (str): Path to the saved model
        threshold_path (str): Path to the saved threshold
        ohe_path (str): Path to saved one-hot-encoder to apply to the new data

    Returns:
        pd.DataFrame: Predictions

    """
    # filter the required features
    required_features = required_features = [
        "age",
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "day_of_week",
        "campaign",
        "pdays",
        "previous",
        "poutcome",
        "emp.var.rate",
        "cons.price.idx",
        "cons.conf.idx",
        "euribor3m",
        "nr.employed",
    ]
    data_preprocessed = input_data.filter(items=required_features)

    # apply One Hot Encoding from training data to categorical features in new data
    ohe_encoder = pickle.load(open(ohe_path, 'rb'))
    data_preprocessed = ohe_encoder.transform(data_preprocessed)

    # load model and threshold
    model = joblib.load(model_path)
    f = open(threshold_path)
    threshold = json.load(f)["best_threshold"]
    f.close()

    # calculate predictions
    predicted_probabilities = model.predict_proba(data_preprocessed)
    predictions = (predicted_probabilities[:, 1] > threshold).astype(int)
    
    predictions = pd.Series(predictions, name = 'Prediction')
    predicted_probabilities = pd.DataFrame(predicted_probabilities, columns = ['Probability-no', 'Probability-yes'])
    predictions = pd.concat([predictions, predicted_probabilities], axis=1)
    
    return predictions


if __name__ == "__main__":
    data_path = os.path.join(os.getcwd(), "../data/data_extracted")
    df = pd.read_csv(os.path.join(data_path, "bank.csv"), delimiter=";")

    predictions = predict_call_worthiness(
        df,
        model_path=os.path.join(os.getcwd(), "../models/best_rf_classifier.sav"),
        threshold_path=os.path.join(os.getcwd(), "../models/best_threshold.json"),
        ohe_path=os.path.join(os.getcwd(), "../models/ohe_encoder.pkl")
    )
"""Data preprocessing

The module fine tunes and trains a Random Forest model to predict worthy and useless calls to clients.

"""

import os
import json

import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn import metrics
from sklearn.metrics import make_scorer, recall_score
from sklearn.metrics import confusion_matrix
import scipy

import joblib
import pickle


def custom_metric(
    y_true: scipy.sparse._csr.csr_matrix,
    y_pred_proba: scipy.sparse._csr.csr_matrix,
    threshold: float,
) -> int:
    """Helper function.
    Calculates gain from using a model given the the business problems requirements and a custom threshold.
    - The prfofit from the conversion is 80$
    - A call cost is 8$
    - Total gain: [correctly predicted as worthy calls] * 80$ - [all predicted as worthy calls] * 8$
    - Opportunity costs: [falsely predicted as useless calls] * (80$ - 8$)
    - Clean gain: Total gain - Opportunity costs
    This metric can be used to find the optimal threshold for the trained Random Forest model.

    Args:
        y_true (scipy.sparse._csr.csr_matrix): True labels
        y_pred_proba (scipy.sparse._csr.csr_matrix): Predicted labels
        threshold (float): Custom threshold for the trained model

    Returns:
        int: Gain from using a model given the the business problems requirements and a custom threshold
    """
    y_pred = (y_pred_proba[:, 1] > threshold).astype(int)
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
    gain = TP * 80 - (TP + FP) * 8
    opp_cost = FN * 72
    clean_gain = gain - opp_cost
    return clean_gain


def model_training(
    tr_x: scipy.sparse._csr.csr_matrix,
    tr_y: scipy.sparse._csr.csr_matrix,
    val_x: scipy.sparse._csr.csr_matrix,
    val_y: scipy.sparse._csr.csr_matrix,
) -> dict:
    """The function preprocesses the input Pandas dataframe as follows:


    Args:
        input_data (pd.DataFrame): Input data that contains all the necessary features and the
        target_col_name (str): Name of the target column
        required_features (list[str]): A list of features required for training
        tr_x (scipy.sparse._csr.csr_matrix): Features of training data
        tr_y (scipy.sparse._csr.csr_matrix): Target of training data
        val_x (scipy.sparse._csr.csr_matrix): Features of validation data
        val_y (scipy.sparse._csr.csr_matrix): Target of validation data

    Returns:
        dict: Dictionary consisting of two values: 'model' and 'threshold'

    """
    # define a classifier
    rf_classifier = RandomForestClassifier(
        class_weight="balanced", bootstrap=True, random_state=42
    )
    # “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies

    # define hyperparameters for fine tuning
    hyperparams = {
        "n_estimators": [100, 500],
        "max_depth": [10, 20],
        "min_samples_leaf": [2, 3],
    }

    # define performance metric - Recall to minimize False Negatives
    recall_scorer = make_scorer(recall_score, average="binary")

    # perform random search over hyperparameters with cross validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    random_search = RandomizedSearchCV(
        estimator=rf_classifier,
        param_distributions=hyperparams,
        cv=cv,
        scoring=recall_scorer,
        verbose=4,
    )
    random_search.fit(tr_x, tr_y)

    # train final model with the best hyperparameters
    best_rf_classifier = random_search.best_estimator_
    best_rf_classifier.fit(tr_x, tr_y)

    # use validation data to optimize the threshold
    y_pred_proba = best_rf_classifier.predict_proba(val_x)
    thresholds = np.linspace(0, 1, 100)
    best_threshold = None
    best_gain = float("-inf")
    for (
        threshold
    ) in thresholds:  # find the best threshold to maximize gain from using the model
        gain = custom_metric(val_y, y_pred_proba, threshold)
        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold

    return {"model": best_rf_classifier, "threshold": best_threshold}

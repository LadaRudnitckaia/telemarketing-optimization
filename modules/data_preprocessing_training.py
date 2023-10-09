"""Data preprocessing

The module implements the function that preprocesses the input data to train a random forest model.

"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def data_preprocessing_training(
    input_data: pd.DataFrame, target_col_name: str, required_features: list[str]
) -> dict:
    """The function preprocesses the input Pandas dataframe as follows:
    - splits the input Pandas dataframe to features and target
    - filters the required features
    - applies one-hot-encoding to categorical features and converts target to integer
    - applies stratified sampling by target and splits data to training, validation, and test sets
    Missing values in categorical columns are treated as separate categories.
    Normalization is not needed for a Random Forest model.

    Args:
        input_data (pd.DataFrame): Input data that contains all the necessary features and the
        target_col_name (str): Name of the target column
        required_features (list[str]): A list of features required for training

    Returns:
        dict: Dictionary consisting of two values: tuple 'splits': (tr_x, tr_y, val_x, val_y, test_x, test_y) and 'ohe_encoder'

    """
    # split to features and target
    df_y = input_data[[target_col_name]]
    df_x = input_data.filter(items=required_features)

    # encode string class values as integers
    y_encoder = LabelEncoder()
    y_encoder = y_encoder.fit(df_y)
    df_y = y_encoder.transform(df_y)

    # stratified train-val-test split
    # note: the train-val-test split is done before OHE to avoid data leakage
    tr_val_x, test_x, tr_val_y, test_y = train_test_split(
        df_x, df_y, test_size=0.10, random_state=42, stratify=df_y
    )
    tr_x, val_x, tr_y, val_y = train_test_split(
        tr_val_x, tr_val_y, test_size=0.10, random_state=42, stratify=tr_val_y
    )

    # apply One Hot Encoding to categorical features
    ohe_encoder = OneHotEncoder(handle_unknown="ignore")
    ohe_encoder = ohe_encoder.fit(tr_x)  # fit on training data
    tr_x = ohe_encoder.transform(tr_x)  # apply to training data
    val_x = ohe_encoder.transform(val_x)  # apply to validation data
    test_x = ohe_encoder.transform(test_x)  # apply to test data

    return {
        "splits": (tr_x, tr_y, val_x, val_y, test_x, test_y),
        "ohe_encoder": ohe_encoder,
    }


if __name__ == "__main__":
    data_path = os.path.join(os.getcwd(), "../data/data_extracted/bank-additional")
    df = pd.read_csv(os.path.join(data_path, "bank-additional-full.csv"), delimiter=";")
    required_features = [
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
    preprocessing_results = data_preprocessing_training(df, "y", required_features)
    tr_x, tr_y, val_x, val_y, test_x, test_y = preprocessing_results["splits"]
    ohe_encoder = preprocessing_results["ohe_encoder"]

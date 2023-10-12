"""Data preprocessing for training

The function that preprocesses the input data to train a random forest model.

"""

import os
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def apply_ohe(
    data: pd.DataFrame,
    ohe_encoder: sklearn.preprocessing._encoders.OneHotEncoder,
    categorical_cols: pd.core.indexes.base.Index,
    numerical_cols: pd.core.indexes.base.Index,
) -> pd.DataFrame:
    """Helper function for mixed type datasets that applies one-hot-encoding only to categorical features and leaves numerical features unchanged.

    Args:
        data (pd.DataFrame): Data to apply one-hot-encoding to, both numeric and string types
        ohe_encoder (sklearn.preprocessing._encoders.OneHotEncoder): Fitted one hot encoder object
        categorical_cols (pandas.core.indexes.base.Index): categorical columns
        numerical_cols (pandas.core.indexes.base.Index): numerical columns

    Returns:
        pd.DataFrame: Encoded data
    """

    categorical_data_encoded = ohe_encoder.transform(data[categorical_cols])
    encoded_feature_names = ohe_encoder.get_feature_names_out(
        input_features=categorical_cols
    )
    categorical_data_encoded = pd.DataFrame(
        categorical_data_encoded, index=data.index, columns=encoded_feature_names
    )

    encoded_data = pd.concat([categorical_data_encoded, data[numerical_cols]], axis=1)
    return encoded_data


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

    # fit one hot encoder using training data
    categorical_cols = tr_x.select_dtypes(include=["object"]).columns
    numerical_cols = tr_x.select_dtypes(exclude=["object"]).columns
    ohe_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    ohe_encoder = ohe_encoder.fit(tr_x[categorical_cols])  # fit on training data

    # apply one hot encoding to train, validation, and test splits
    tr_x_enc = apply_ohe(
        tr_x, ohe_encoder, categorical_cols, numerical_cols
    )  # apply to training data
    val_x_enc = apply_ohe(
        val_x, ohe_encoder, categorical_cols, numerical_cols
    )  # apply to validation data
    test_x_enc = apply_ohe(
        test_x, ohe_encoder, categorical_cols, numerical_cols
    )  # apply to test data

    return {
        "splits": (tr_x_enc, tr_y, val_x_enc, val_y, test_x_enc, test_y),
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

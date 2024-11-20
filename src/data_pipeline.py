"""
data_pipeline.py

This script preprocesses and cleans the dataset used for training the machine
learning model. It handles missing values, performs necessary transformations,
and saves the processed data to 'processed_data.csv'.

Key Steps:
- Missing data handling: Fills missing values in specified columns.
- Data transformation: Applies necessary data transformations.
- Data saving: Saves the preprocessed data for further model training.

To run this script:
$ python data_pipeline.py
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import yaml


def load_config():
    """
    Loads configuration from 'config.yml'.

    Returns:
        dict: Configuration dictionary.
    """
    with open("config.yml", "r") as config_file:
        return yaml.safe_load(config_file)


def load_data(config):
    """
    Loads the raw dataset.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        pd.DataFrame: Raw dataset.
    """
    data_path = config["paths"]["data_raw"]
    return pd.read_csv(data_path)


def handle_missing_data(data):
    """
    Handles missing data by replacing zeros with column mean values.

    Args:
        data (pd.DataFrame): Dataset with potential missing values.

    Returns:
        pd.DataFrame: Dataset with filled missing values.
    """
    columns_with_missing_values = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for column in columns_with_missing_values:
        data[column] = data[column].replace(0, np.nan)
        data[column] = data[column].fillna(data[column].mean())
    return data


def remove_outliers(data):
    """
    Removes outliers based on the interquartile range.

    Args:
        data (pd.DataFrame): Dataset with potential outliers.

    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    return data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]


def feature_engineering(data):
    """
    Performs feature engineering by creating interaction and polynomial features.

    Args:
        data (pd.DataFrame): Dataset.

    Returns:
        pd.DataFrame: Dataset with engineered features.
    """
    data['BMI_Age_Interaction'] = data['BMI'] * data['Age']
    data['BMI_Squared'] = data['BMI'] ** 2
    data['Age_Squared'] = data['Age'] ** 2
    data['Glucose_BMI'] = data['Glucose'] * data['BMI']
    data['Age_Glucose'] = data['Age'] * data['Glucose']
    data['BloodPressure_BMI'] = data['BloodPressure'] * data['BMI']
    data['SkinThickness_Insulin'] = data['SkinThickness'] * data['Insulin']
    data['Pregnancies_Age'] = data['Pregnancies'] * data['Age']
    data['Glucose_Insulin'] = data['Glucose'] * data['Insulin']
    return data


def scale_data(data):
    """
    Scales numerical features using StandardScaler.

    Args:
        data (pd.DataFrame): Dataset.

    Returns:
        pd.DataFrame: Scaled dataset.
    """
    scaler = StandardScaler()
    scaled_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 
                      'DiabetesPedigreeFunction', 'Age', 'BMI_Squared', 'Age_Squared', 
                      'Glucose_BMI', 'Age_Glucose', 'BloodPressure_BMI', 'SkinThickness_Insulin', 
                      'Pregnancies_Age', 'Glucose_Insulin', 'BMI_Age_Interaction']
    data[scaled_columns] = scaler.fit_transform(data[scaled_columns])
    return data


def process_data(config):
    """
    Preprocesses the raw dataset by handling missing values, removing outliers,
    performing feature engineering, and scaling data.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        pd.DataFrame: Processed dataset.
    """
    data = load_data(config)
    data = handle_missing_data(data)
    data = remove_outliers(data)
    data = feature_engineering(data)
    data = scale_data(data)
    return data


if __name__ == "__main__":
    config = load_config()
    processed_data = process_data(config)
    processed_data_path = config["paths"]["data_processed"]
    processed_data.to_csv(processed_data_path, index=False)
    print(f"Data processing complete. Processed data saved to '{processed_data_path}'.")

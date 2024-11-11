import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

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

def load_data(file_path='../data/diabetes.csv'):
    data = pd.read_csv(file_path)
    return data

def process_data(file_path='../data/diabetes.csv'):
    data = load_data(file_path)
    data = handle_missing_data(data)
    data = remove_outliers(data)
    data = feature_engineering(data)
    data = scale_data(data)
    return data

# Hiányzó adatok kezelése (például 0 értékek átlaggal pótlása)
def handle_missing_data(data):
    columns_with_missing_values = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for column in columns_with_missing_values:
        data[column] = data[column].replace(0, np.nan)
        data[column] = data[column].fillna(data[column].mean())
    return data

# Outlierek kezelése IQR módszerrel
def remove_outliers(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    data_cleaned = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
    return data_cleaned

# Új jellemzők létrehozása (feature engineering)
def feature_engineering(data):
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

# Adatok skálázása (standardizálás)
def scale_data(data):
    scaler = StandardScaler()
    scaled_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 
                      'DiabetesPedigreeFunction', 'Age', 'BMI_Squared', 'Age_Squared', 
                      'Glucose_BMI', 'Age_Glucose', 'BloodPressure_BMI', 'SkinThickness_Insulin', 
                      'Pregnancies_Age', 'Glucose_Insulin', 'BMI_Age_Interaction']
    data[scaled_columns] = scaler.fit_transform(data[scaled_columns])
    return data

# Fő függvény, amely végrehajtja az összes adatfeldolgozási lépést
if __name__ == "__main__":
    processed_data = process_data()

    # Adatok mentése közvetlenül a ../data könyvtárba
    output_path = '../data/processed_data.csv'
    processed_data.to_csv(output_path, index=False)
    print(f"Data processing complete. Processed data saved to '{output_path}'.")

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

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

# Load data from the data directory
def load_data(file_path=None):
    # Define default path if no path is provided
    if file_path is None:
        file_path = os.path.join(os.path.dirname(__file__), '../data/diabetes.csv')
    data = pd.read_csv(file_path)
    return data

# Handle missing values in specific columns
def handle_missing_data(data):
    columns_with_missing_values = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for column in columns_with_missing_values:
        data[column] = data[column].replace(0, np.nan)
        data[column] = data[column].fillna(data[column].mean())
    return data

# Remove outliers using the IQR method
def remove_outliers(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    data_cleaned = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
    return data_cleaned

# Feature engineering to create new columns
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

# Scale data for model training
def scale_data(data):
    scaler = StandardScaler()
    scaled_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 
                      'DiabetesPedigreeFunction', 'Age', 'BMI_Squared', 'Age_Squared', 
                      'Glucose_BMI', 'Age_Glucose', 'BloodPressure_BMI', 'SkinThickness_Insulin', 
                      'Pregnancies_Age', 'Glucose_Insulin', 'BMI_Age_Interaction']
    data[scaled_columns] = scaler.fit_transform(data[scaled_columns])
    return data

# Main processing function that combines all steps
def process_data(file_path=None):
    data = load_data(file_path)
    data = handle_missing_data(data)
    data = remove_outliers(data)
    data = feature_engineering(data)
    data = scale_data(data)
    return data

# Run the script and save processed data
if __name__ == "__main__":
    processed_data = process_data()
    processed_data_path = os.path.join(os.path.dirname(__file__), '../data/processed_data.csv')
    processed_data.to_csv(processed_data_path, index=False)
    print(f"Data processing complete. Processed data saved to '{processed_data_path}'.")

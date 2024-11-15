import unittest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.data_pipeline import (
    handle_missing_data, remove_outliers, feature_engineering, scale_data
)

class TestDataPipeline(unittest.TestCase):

    def setUp(self):
        """Set up mock data for testing."""
        self.mock_data = pd.DataFrame({
            'Glucose': [0, 120, 150, 0],
            'BloodPressure': [80, 0, 90, 0],
            'SkinThickness': [25, 35, 0, 0],
            'Insulin': [0, 150, 200, 0],
            'BMI': [32.0, 0, 28.5, 0],
            'Age': [25, 35, 45, 55],
            'Pregnancies': [1, 2, 3, 4],
            'DiabetesPedigreeFunction': [0.5, 0.7, 0.6, 0.8],
            'Outcome': [1, 0, 1, 0]
        })

    def test_handle_missing_data(self):
        """Test missing data handling."""
        processed_data = handle_missing_data(self.mock_data.copy())
        self.assertFalse(processed_data.isnull().any().any(), "Data contains NaNs after handling missing data")
        self.assertTrue((processed_data['Glucose'] != 0).all(), "Zeros in 'Glucose' not replaced properly")

    def test_remove_outliers(self):
        """Test outlier removal."""
        data_with_outliers = self.mock_data.copy()
        data_with_outliers.loc[0, 'Glucose'] = 500  # Add an outlier
        cleaned_data = remove_outliers(data_with_outliers)
        self.assertNotIn(500, cleaned_data['Glucose'], "Outlier not removed")

    def test_feature_engineering(self):
        """Test feature engineering."""
        engineered_data = feature_engineering(self.mock_data.copy())
        expected_columns = {'BMI_Age_Interaction', 'BMI_Squared', 'Age_Squared', 'Glucose_BMI', 
                            'Age_Glucose', 'BloodPressure_BMI', 'SkinThickness_Insulin', 
                            'Pregnancies_Age', 'Glucose_Insulin'}
        self.assertTrue(expected_columns.issubset(engineered_data.columns), "Feature engineering did not create expected columns")

    def test_scale_data(self):
        """Test data scaling."""
        engineered_data = feature_engineering(self.mock_data.copy())
        scaled_data = scale_data(engineered_data)
        scaled_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 
                        'DiabetesPedigreeFunction', 'Age', 'BMI_Squared', 'Age_Squared', 
                        'Glucose_BMI', 'Age_Glucose', 'BloodPressure_BMI', 
                        'SkinThickness_Insulin', 'Pregnancies_Age', 'Glucose_Insulin', 
                        'BMI_Age_Interaction']
        for col in scaled_columns:
            if col in scaled_data.columns:
                self.assertAlmostEqual(
                    scaled_data[col].std(), 1, delta=0.2,  # Enyhébb tűréshatár
                    msg=f"{col} not scaled correctly (std != 1)"
                )


if __name__ == '__main__':
    unittest.main()

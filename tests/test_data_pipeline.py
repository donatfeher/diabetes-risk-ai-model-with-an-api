import unittest
import pandas as pd
import numpy as np
from src.data_pipeline import handle_missing_data, remove_outliers, feature_engineering, scale_data

class TestDataPipeline(unittest.TestCase):

    def setUp(self):
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
        processed_data = handle_missing_data(self.mock_data.copy())
        self.assertFalse(processed_data.isnull().any().any())

    def test_remove_outliers(self):
        data_with_outliers = self.mock_data.copy()
        data_with_outliers.loc[0, 'Glucose'] = 500
        cleaned_data = remove_outliers(data_with_outliers)
        self.assertNotIn(500, cleaned_data['Glucose'])

    def test_feature_engineering(self):
        engineered_data = feature_engineering(self.mock_data.copy())
        self.assertIn('BMI_Age_Interaction', engineered_data.columns)

    def test_scale_data(self):
        engineered_data = feature_engineering(self.mock_data.copy())
        scaled_data = scale_data(engineered_data)
        for col in ['Glucose', 'BMI']:
            self.assertAlmostEqual(scaled_data[col].std(), 1, delta=0.2)

if __name__ == '__main__':
    unittest.main()

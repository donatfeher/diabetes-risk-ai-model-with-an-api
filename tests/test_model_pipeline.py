import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from models.model_pipeline import train_and_select_model, save_model

class TestModelPipeline(unittest.TestCase):

    def setUp(self):
        """Bővített adatokat használunk, hogy SMOTE működjön."""
        self.mock_data = pd.DataFrame({
            'Glucose': [120, 150, 130, 140, 160, 170],
            'BloodPressure': [80, 85, 90, 88, 75, 77],
            'SkinThickness': [25, 30, 28, 27, 22, 20],
            'Insulin': [130, 140, 135, 150, 120, 125],
            'BMI': [32.0, 28.5, 30.0, 29.5, 27.5, 31.5],
            'Age': [25, 35, 45, 55, 30, 40],
            'Pregnancies': [1, 2, 3, 4, 2, 3],
            'DiabetesPedigreeFunction': [0.5, 0.7, 0.6, 0.8, 0.4, 0.6],
            'Outcome': [1, 0, 1, 0, 1, 0]  # Több minta mindkét osztályban
        })

    @patch("models.model_pipeline.GridSearchCV")
    @patch("models.model_pipeline.train_test_split")
    def test_train_and_select_model(self, mock_train_test_split, mock_grid_search):
        """Valós predikciós eredmények használata a tesztben."""
        # Mock train-test split
        mock_train_test_split.return_value = (
            self.mock_data.drop('Outcome', axis=1).iloc[:4],  # X_train
            self.mock_data.drop('Outcome', axis=1).iloc[4:],  # X_test
            self.mock_data['Outcome'].iloc[:4],               # y_train
            self.mock_data['Outcome'].iloc[4:]                # y_test
        )

        # Mock GridSearchCV
        mock_model = MagicMock()
        mock_model.best_estimator_ = MagicMock()
        mock_model.best_estimator_.predict.return_value = [1, 0]  # Predikciók hossza egyezik a y_test-tel
        mock_model.best_score_ = 0.95
        mock_grid_search.return_value = mock_model

        # Teszteljük a train_and_select_model függvényt
        best_model = train_and_select_model(self.mock_data)
        self.assertIsNotNone(best_model)
        self.assertEqual(mock_model.best_estimator_, best_model)

    @patch("models.model_pipeline.pickle.dump")
    def test_save_model(self, mock_pickle_dump):
        """Teszteljük a modell mentését."""
        model = MagicMock()
        save_model(model, "test_model.pkl")
        mock_pickle_dump.assert_called_once()

if __name__ == '__main__':
    unittest.main()

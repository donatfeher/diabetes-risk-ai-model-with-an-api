import unittest
import pandas as pd
from sklearn.linear_model import LogisticRegression
from models.model_pipeline import train_and_select_model, save_model
from unittest.mock import patch, MagicMock


class TestModelPipeline(unittest.TestCase):

    def setUp(self):
        """Set up mock data for testing."""
        self.mock_data = pd.DataFrame({
            'Glucose': [120, 150, 130, 140],
            'BloodPressure': [80, 85, 90, 88],
            'SkinThickness': [25, 30, 28, 27],
            'Insulin': [130, 140, 135, 150],
            'BMI': [32.0, 28.5, 30.0, 29.5],
            'Age': [25, 35, 45, 55],
            'Pregnancies': [1, 2, 3, 4],
            'DiabetesPedigreeFunction': [0.5, 0.7, 0.6, 0.8],
            'Outcome': [1, 0, 1, 0]
        })

    @patch("models.model_pipeline.GridSearchCV")
    @patch("models.model_pipeline.train_test_split")
    def test_train_and_select_model(self, mock_train_test_split, mock_grid_search):
        """Test model training and selection."""
        mock_train_test_split.return_value = (
            self.mock_data.drop('Outcome', axis=1),
            self.mock_data.drop('Outcome', axis=1),
            self.mock_data['Outcome'],
            self.mock_data['Outcome']
        )

        mock_model = MagicMock()
        mock_model.best_estimator_ = LogisticRegression()
        mock_model.best_estimator_.fit = MagicMock()
        mock_model.best_estimator_.predict = MagicMock(return_value=[1, 0, 1, 0])
        mock_model.best_score_ = 0.95

        mock_grid_search.return_value = mock_model

        best_model = train_and_select_model(self.mock_data)
        self.assertEqual(mock_grid_search.return_value.best_estimator_, best_model)

    @patch("models.model_pipeline.pickle.dump")
    def test_save_model(self, mock_pickle_dump):
        """Test model saving."""
        model = MagicMock()
        save_model(model, "mock_path.pkl")
        mock_pickle_dump.assert_called_once_with(model, unittest.mock.ANY)


if __name__ == '__main__':
    unittest.main()

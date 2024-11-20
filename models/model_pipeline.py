import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE  # SMOTE hozzáadása

def load_processed_data(file_path=None):
    """
    Loads the processed dataset.

    Args:
        file_path (str): Path to the processed data file. Defaults to '../data/processed_data.csv'.

    Returns:
        pd.DataFrame: Processed dataset.
    """
    if file_path is None:
        file_path = os.path.join(os.path.dirname(__file__), '../data/processed_data.csv')
    return pd.read_csv(file_path)

def train_and_select_model(data):
    """
    Trains and evaluates multiple machine learning models, and selects the best one.

    Args:
        data (pd.DataFrame): Dataset containing features and target variable.

    Returns:
        model: The best-performing trained model.
    """
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # SMOTE használata az osztályok kiegyensúlyozásához
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    models = {
        'KNN': (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}),
        'Logistic Regression': (LogisticRegression(class_weight='balanced'), {'C': [0.1, 1, 10], 'max_iter': [200]}),
        'Decision Tree': (DecisionTreeClassifier(), {'max_depth': [3, 5, 7]}),
        'Random Forest': (RandomForestClassifier(), {'n_estimators': [50, 100], 'max_depth': [5, 7]}),
        'XGBoost': (XGBClassifier(eval_metric='logloss'), {'n_estimators': [50, 100], 'learning_rate': [0.1, 0.2], 'max_depth': [3, 5]})
    }

    best_model = None
    best_score = 0
    best_model_name = ""

    for model_name, (model, params) in models.items():
        grid = GridSearchCV(model, params, cv=5, scoring='accuracy')
        grid.fit(X_train, y_train)
        print(f"{model_name} best accuracy: {grid.best_score_:.2f}")

        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_model = grid.best_estimator_
            best_model_name = model_name

    y_pred = best_model.predict(X_test)
    print(f"\nBest Model: {best_model_name}")
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred))

    return best_model

def save_model(model, file_path):
    """
    Saves the trained model to a file.

    Args:
        model: The trained machine learning model.
        file_path (str): Path to save the model file.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to '{file_path}'.")

if __name__ == "__main__":
    data = load_processed_data()
    best_model = train_and_select_model(data)
    model_save_path = os.path.join(os.path.dirname(__file__), '../models/final_model.pkl')
    save_model(best_model, model_save_path)

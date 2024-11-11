"""
model_pipeline.py

This script trains multiple machine learning models on a dataset, evaluates
their performance, and saves the best-performing model as 'final_model.pkl'.
The models tested include K-Nearest Neighbors, Logistic Regression,
Decision Tree, Random Forest, and XGBoost.

Key Functions:
- Model training: Trains each model and calculates the accuracy on a validation set.
- Model selection: Selects the model with the best performance based on accuracy.
- Model saving: Saves the best model to a file ('final_model.pkl').

To run this script:
$ python model_pipeline.py
"""

import pandas as pd
import pickle
import os
import yaml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load configuration from config.yml
with open("config.yml", "r") as config_file:
    config = yaml.safe_load(config_file)

def load_processed_data():
    data_path = config["paths"]["data_processed"]
    data = pd.read_csv(data_path)
    return data

def train_and_select_model(data):
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config["model_params"]["test_size"], random_state=config["model_params"]["random_state"])

    models = {
        'KNN': (KNeighborsClassifier(), config["models"]["KNN"]),
        'LogisticRegression': (LogisticRegression(max_iter=config["models"]["LogisticRegression"]["max_iter"]), config["models"]["LogisticRegression"]),
        'DecisionTree': (DecisionTreeClassifier(), config["models"]["DecisionTree"]),
        'RandomForest': (RandomForestClassifier(), config["models"]["RandomForest"]),
        'XGBoost': (XGBClassifier(eval_metric=config["models"]["XGBoost"]["eval_metric"]), config["models"]["XGBoost"])
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

    model_save_path = config["paths"]["model"]
    with open(model_save_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"{best_model_name} model saved to '{model_save_path}'.")

if __name__ == "__main__":
    data = load_processed_data()
    train_and_select_model(data)

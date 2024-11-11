import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

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

# Load the processed data from the updated path
def load_processed_data(file_path='../data/processed_data.csv'):
    data = pd.read_csv(file_path)
    return data

# Train and select the best model
def train_and_select_model(data):
    X = data.drop('Outcome', axis=1)  # Features
    y = data['Outcome']  # Target variable

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define model parameters for GridSearch
    models = {
        'KNN': (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}),
        'Logistic Regression': (LogisticRegression(max_iter=200), {'C': [0.1, 1, 10]}),
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

    # Evaluate the best model on the test set
    y_pred = best_model.predict(X_test)
    print(f"\nBest Model: {best_model_name}")
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred))

    # Save the best model to the updated path
    with open('../models/final_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print(f"{best_model_name} model saved to '../models/final_model.pkl'.")

# Main function to run the model pipeline
if __name__ == "__main__":
    data = load_processed_data()
    train_and_select_model(data)

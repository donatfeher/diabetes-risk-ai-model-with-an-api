import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

def load_processed_data(file_path="../data/processed_data.csv"):
    """Betölti a feldolgozott adatokat."""
    return pd.read_csv(file_path)

def train_with_mlflow(data):
    """MLFlow-val követett modell tanítás."""
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # SMOTE az osztályok kiegyensúlyozásához
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Modell beállítása
    model = XGBClassifier(eval_metric="logloss", n_estimators=100, max_depth=5, learning_rate=0.1)

    # MLFlow Tracking kezdése
    with mlflow.start_run():
        # Paraméterek logolása
        mlflow.log_param("n_estimators", model.n_estimators)
        mlflow.log_param("max_depth", model.max_depth)
        mlflow.log_param("learning_rate", model.learning_rate)

        # Modell tanítása
        model.fit(X_train, y_train)

        # Pontosság kiértékelése
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)

        # Modell mentése az MLFlow-ba
        mlflow.sklearn.log_model(model, "model")
        print(f"Model logged with accuracy: {accuracy}")

if __name__ == "__main__":
    data = load_processed_data()
    train_with_mlflow(data)

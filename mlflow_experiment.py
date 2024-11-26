import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

def load_processed_data(file_path="data/processed_data.csv"):
    """Betölti a feldolgozott adatokat."""
    return pd.read_csv(file_path)

def train_with_mlflow(data):
    """MLFlow-val követett modell tanítás."""
    # Adatok szétválasztása
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # SMOTE az osztályok kiegyensúlyozásához
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Modell beállítása
    model = XGBClassifier(eval_metric="logloss", n_estimators=100, max_depth=5, learning_rate=0.1)

    # Artifact mappa definiálása
    artifact_dir = "artifacts"
    os.makedirs(artifact_dir, exist_ok=True)  # Létrehozzuk az artifact mappát, ha nem létezik

    # MLFlow Tracking kezdése
    with mlflow.start_run() as run:
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

        # Konfúziós mátrix létrehozása és mentése
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title("Confusion Matrix")
        cm_path = os.path.join(artifact_dir, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path)

        # ROC Görbe létrehozása és mentése
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange',
                 lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        roc_path = os.path.join(artifact_dir, "roc_curve.png")
        plt.savefig(roc_path)
        plt.close()
        mlflow.log_artifact(roc_path)

        # Eredmények HTML mentése (opcionális)
        results_html = os.path.join(artifact_dir, "Results.html")
        with open(results_html, "w") as f:
            f.write(f"<html><body><h1>Results</h1><p>Accuracy: {accuracy:.2f}</p></body></html>")
        mlflow.log_artifact(results_html)

        # Követelmények mentése
        requirements_path = "requirements.txt"
        if os.path.exists(requirements_path):
            # Ha létezik a requirements.txt, mentjük az artifactok közé
            mlflow.log_artifact(requirements_path, artifact_path=artifact_dir)
        else:
            # Ha nem létezik a requirements.txt, létrehozzuk
            with open(requirements_path, "w") as f:
                # Itt érdemes a pip freeze-t használni a pontos csomaglistához
                os.system("pip freeze > requirements.txt")
            mlflow.log_artifact(requirements_path, artifact_path=artifact_dir)

        # Modell mentése az MLFlow-ba
        mlflow.sklearn.log_model(model, os.path.join(artifact_dir, "model"))
        print(f"Model logged with accuracy: {accuracy}")

def download_artifact(run_id, artifact_path, download_dir="./downloaded_artifacts"):
    """Artifact letöltése MLflow futásból."""
    client = mlflow.tracking.MlflowClient()
    client.download_artifacts(run_id, artifact_path, download_dir)
    print(f"Artifact {artifact_path} from run {run_id} downloaded to {download_dir}")

if __name__ == "__main__":
    data = load_processed_data()
    train_with_mlflow(data)


"""# Ez a kód egy gépi tanulási (machine learning) kísérletet hajt végre, amely során egy modellt tanít és követi az MLflow segítségével.

import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

def load_processed_data(file_path="../data/processed_data.csv"):
    return pd.read_csv(file_path)

def train_with_mlflow(data):
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
"""
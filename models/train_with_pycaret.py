import os
import logging
import yaml
import pandas as pd
import mlflow
from pycaret.classification import setup, compare_models, pull, save_model

# Logging beállítása
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Config fájl betöltése
CONFIG_PATH = "config.yml"

def load_config(config_path):
    """Betölti a konfigurációs fájlt."""
    if not os.path.exists(config_path):
        logger.critical(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    logger.info("Config file loaded successfully.")
    return config

def load_data(data_path):
    """Betölti az adatokat a megadott útvonalról."""
    if not os.path.exists(data_path):
        logger.critical(f"Data file not found: {data_path}")
        raise FileNotFoundError(f"Data file not found: {data_path}")
    data = pd.read_csv(data_path)
    logger.info("Data loaded successfully.")
    return data

def clean_active_mlflow_runs():
    """Befejezi az összes aktív MLflow futást."""
    logger.info("Cleaning active MLflow runs...")
    try:
        while mlflow.active_run() is not None:  # Ellenőrzi, hogy van-e aktív futás
            mlflow.end_run()  # Befejezi az aktív futást
        logger.info("All active MLflow runs cleaned.")
    except Exception as e:
        logger.warning(f"Failed to clean MLflow runs: {e}")

def main():
    # Config fájl betöltése
    config = load_config(CONFIG_PATH)
    
    # Adatok betöltése
    data_path = config["data"]["path"]
    target_column = config["data"]["target_column"]
    data = load_data(data_path)

    # MLflow beállítások
    mlflow_tracking_uri = config["mlflow"]["tracking_uri"]
    experiment_name = config["mlflow"]["experiment_name"]
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Aktív MLflow futások tisztítása
    clean_active_mlflow_runs()

    # PyCaret setup
    logger.info("Starting PyCaret setup...")
    clf1 = setup(
        data=data,
        target=target_column,
        log_experiment=True,
        experiment_name=experiment_name,
        html=False  # HTML kimenet letiltása
    )
    logger.info("PyCaret setup completed successfully.")

    # Modellek összehasonlítása
    logger.info("Comparing models...")
    best_model = compare_models()
    logger.info(f"Best model identified: {best_model}")

    # MLflow-ba logolás
    logger.info("Logging best model and metrics to MLflow...")
    with mlflow.start_run(nested=True):  # Nested használata az aktív futások elkerülése érdekében
        # Modell mentése
        model_save_path = config["output"]["model_save_path"]
        save_model(best_model, model_save_path)
        logger.info(f"Model saved successfully at {model_save_path}")

        # Modellek logolása MLflow-ba
        mlflow.sklearn.log_model(best_model, "model")
        logger.info("Model logged to MLflow.")

        # Példa metrikák naplózása MLflow-hoz
        metrics = {
            "Accuracy": 0.78,
            "TT (Sec)": 0.016,  # Ez a hibás metrika
            "Precision": 0.73,
        }

        # Metrikák naplózása
        for metric_name, metric_value in metrics.items():
            sanitized_name = metric_name.replace(" ", "_").replace("(", "").replace(")", "")
            mlflow.log_metric(sanitized_name, metric_value)
        logger.info("Metrics logged to MLflow.")
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Training process failed: {e}", exc_info=True)

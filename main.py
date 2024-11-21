import os
import pickle
import yaml
import logging
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from src.data_pipeline import feature_engineering, scale_data
import subprocess
import uvicorn
from models.train_with_pycaret import train_with_pycaret, load_processed_data # pycaret

# Logger beállítása
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Konfiguráció betöltése
with open("config.yml", "r") as config_file:
    config = yaml.safe_load(config_file)

# Modell betöltése
model_path = config["paths"]["model"]
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully from %s.", model_path)
except FileNotFoundError:
    logger.error("Model file not found at %s. Run /train to create it.", model_path)
    model = None
except Exception as e:
    logger.error("Error loading model: %s", str(e))
    model = None

# FastAPI alkalmazás
app = FastAPI()

# Input adatmodell
class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# Adat előfeldolgozó függvény
def preprocess_input(input_data: DiabetesInput):
    data = pd.DataFrame([input_data.dict()])
    data = feature_engineering(data)
    data = scale_data(data)
    return data.values

# Predikciós végpont
@app.post("/predict")
def predict_diabetes(input_data: DiabetesInput):
    if model is None:
        return {"error": "Model is not loaded. Run /train to train the model."}

    try:
        processed_data = preprocess_input(input_data)
        probabilities = model.predict_proba(processed_data)
        prediction = 1 if probabilities[0][1] > 0.25 else 0  # 0.25 küszöb
        return {"prediction": prediction, "probabilities": probabilities[0].tolist()}
    except Exception as e:
        logger.error("Error during prediction: %s", str(e))
        return {"error": "Prediction failed. Check logs for more details."}

# Tanító végpont
@app.post("/train")
def train():
    """
    Train the model by invoking the `model_pipeline.py` script.
    """
    try:
        script_path = os.path.join(os.getcwd(), "models/model_pipeline.py")
        logger.info("Starting training script at %s.", script_path)
        
        result = subprocess.run(["python", script_path], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("Training completed successfully.")
            return {"message": "Training completed successfully.", "output": result.stdout}
        else:
            logger.error("Training script failed with return code %d.", result.returncode)
            logger.error("Error details: %s", result.stderr)
            return {"error": "Training failed.", "details": result.stderr}
    except Exception as e:
        logger.error("Failed to execute training: %s", str(e))
        return {"error": f"Failed to execute training: {str(e)}"}

# Pycaret
@app.post("/train_pycaret")
def train_pycaret():
    """
    Train a model using PyCaret and track experiments with MLflow.
    """
    try:
        data = load_processed_data()
        model_path = train_with_pycaret(data)
        return {"message": "PyCaret training completed.", "model_path": model_path}
    except Exception as e:
        logger.error("Error during PyCaret training: %s", str(e))
        return {"error": f"PyCaret training failed: {str(e)}"}


# Fő futtatás
if __name__ == "__main__":
    uvicorn.run(app, host=config["server"]["host"], port=config["server"]["port"])

"""
main.py

This is the main entry point of the FastAPI-based machine learning API.
It loads a pre-trained machine learning model (final_model.pkl) and serves
it via API endpoints.

Key Endpoints:
- /predict: Accepts input data as JSON and returns the prediction from the model.
- /docs: Provides automatically generated API documentation.

To run the API server:
$ uvicorn main:app --reload
"""

import pickle
import yaml
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import uvicorn
import logging
from src.data_pipeline import feature_engineering, scale_data

# Load configuration from config.yml
with open("config.yml", "r") as config_file:
    config = yaml.safe_load(config_file)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

try:
    with open(config["paths"]["model"], 'rb') as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully.")
except FileNotFoundError:
    logger.error("Model file not found. Run model_pipeline.py to create it.")
    model = None

class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

def preprocess_input(input_data: DiabetesInput):
    data = pd.DataFrame([input_data.dict()])
    data = feature_engineering(data)
    data = scale_data(data)
    return data.values

@app.post("/predict")
def predict_diabetes(input_data: DiabetesInput):
    if model is None:
        return {"error": "Model is not loaded. Run model_pipeline.py to train the model."}

    processed_data = preprocess_input(input_data)
    prediction = model.predict(processed_data)
    
    return {"prediction": int(prediction[0])}

if __name__ == "__main__":
    uvicorn.run(app, host=config["server"]["host"], port=config["server"]["port"])

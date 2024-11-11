import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from data_pipeline import feature_engineering, scale_data  # Használjuk a data_pipeline függvényeit
import uvicorn
import logging

# Naplózási konfiguráció
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# A FastAPI alkalmazás inicializálása
app = FastAPI()

# Modell betöltése
try:
    with open('final_model.pkl', 'rb') as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully.")
except FileNotFoundError:
    logger.error("Model file not found. Run model_pipeline.py to create it.")
    model = None

# Bemeneti adatmodell létrehozása
class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# Adat előfeldolgozási függvény, amely a bemenetet a modellhez igazítja
def preprocess_input(input_data: DiabetesInput):
    # Adatok átalakítása DataFrame-é
    data = pd.DataFrame([input_data.dict()])
    
    # Feature engineering és skálázás
    data = feature_engineering(data)
    data = scale_data(data)
    
    # Bemeneti adat átalakítása numpy tömbbé (modell predikcióhoz)
    return data.values

# API végpont a predikcióra
@app.post("/predict")
def predict_diabetes(input_data: DiabetesInput):
    if model is None:
        return {"error": "Model is not loaded. Run model_pipeline.py to train the model."}

    # Előfeldolgozás és predikció
    processed_data = preprocess_input(input_data)
    prediction = model.predict(processed_data)
    
    return {"prediction": int(prediction[0])}

# A FastAPI szerver indítása
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

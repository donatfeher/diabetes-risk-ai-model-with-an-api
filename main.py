import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import uvicorn
import logging

# Naplózási konfiguráció
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# A FastAPI alkalmazás inicializálása
app = FastAPI()

# Betöltjük a mentett modellt (például Random Forest)
with open('final_model.pkl', 'rb') as f:
    model = pickle.load(f)
    logger.info("Model loaded successfully.")

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

# API végpont a predikcióra
@app.post("/predict")
def predict_diabetes(input_data: DiabetesInput):
    # A bemenet átalakítása numpy tömbbé
    input_array = np.array([[input_data.Pregnancies, input_data.Glucose, input_data.BloodPressure,
                             input_data.SkinThickness, input_data.Insulin, input_data.BMI,
                             input_data.DiabetesPedigreeFunction, input_data.Age]])

    # Naplózás a bemenethez
    logger.info(f"Input received: {input_array}")

    # Ha a modell 9 jellemzőt vár, adjunk hozzá egy extra oszlopot (pl. alapértelmezett értékkel)
    input_array = np.append(input_array, [[0]])  # Adj hozzá egy 9. jellemzőt
    logger.info(f"Modified input with added feature: {input_array}")

    # Predikció készítése
    prediction = model.predict([input_array])

    # Naplózás a predikcióhoz
    logger.info(f"Prediction made: {prediction[0]}")

    return {"prediction": int(prediction[0])}

# A FastAPI szerver indítása
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

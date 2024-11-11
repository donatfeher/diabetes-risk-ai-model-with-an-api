# FastAPI-based Machine Learning Model Documentation

[Magyar verzió (Hungarian)](README_hu.md)

This documentation provides clear steps for others to easily run the FastAPI-based machine learning model. Follow these steps to set up and run the code in your environment.

### 1. Clone or Download the Code Locally

Ensure that **main.py** and the required model files (e.g., `final_model.pkl`) are available in the directory where you're working.

### 2. Install Required Packages

Navigate to the directory where your FastAPI file is located. Install the required Python packages with specific versions for compatibility:

```bash
pip install fastapi==0.112.2 uvicorn==0.30.0 scikit-learn==1.5.1 pandas==2.2.2 numpy==1.25.0 xgboost==2.1.1
```

### 3. Run the FastAPI Server

Start the FastAPI server with the following command. Make sure that `main` is the name of the file where your FastAPI application is defined:

```bash
uvicorn main:app --reload
```

This will launch the server, which listens on `http://127.0.0.1:8000`.

### 4. API Documentation

FastAPI automatically generates an interactive API documentation interface. Access it in your browser at:

```bash
http://127.0.0.1:8000/docs
```

Here, you can test the API endpoints.

### 5. Sending a Prediction Request

To make a prediction, use a `POST` request to the following URL:

```bash
http://127.0.0.1:8000/predict
```

Send a sample request with this JSON payload:

```json
{
  "Pregnancies": 2,
  "Glucose": 120.0,
  "BloodPressure": 70.0,
  "SkinThickness": 20.0,
  "Insulin": 85.0,
  "BMI": 25.0,
  "DiabetesPedigreeFunction": 0.5,
  "Age": 30
}
```

### 6. Server Response

The server will return the prediction in JSON format, such as:

```json
{
  "prediction": 1
}
```

### Common Errors and Solutions

1. **`ModuleNotFoundError: No module named 'fastapi'`**: Ensure the required packages are installed with correct versions (`pip install fastapi==0.112.2 uvicorn==0.30.0`).

2. **`ValueError: X has 8 features, but RandomForestClassifier is expecting 9 features`**: This error indicates the model expects more input data. Double-check the model requirements and input data.

3. **Port Conflict**: If port `8000` is in use, try a different port:

   ```bash
   uvicorn main:app --reload --port 8080
   ```
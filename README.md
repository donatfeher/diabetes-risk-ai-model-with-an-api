# FastAPI-based Machine Learning Model Documentation

[Magyar verzió (Hungarian)](README_hu.md)

This documentation provides clear steps for others to easily run the FastAPI-based machine learning model. Below are the steps to set up and run the code in your environment.

## FastAPI-based Machine Learning Model Documentation

This documentation provides clear steps for others to easily run the FastAPI-based machine learning model. Below are the steps to set up and run the code in your environment.

### 1. Clone or Download the Code Locally

Ensure that **main.py** and the required model files (e.g., `final_model.pkl`) are available in the directory where you're working.

### 2. Install Required Packages

Open your terminal and navigate to the directory where your FastAPI file is located. You'll need to install a few Python packages:

```bash
pip install fastapi uvicorn scikit-learn numpy
```

### 3. Run the FastAPI Server

You can start the FastAPI server using the following command. Make sure that `main` is the name of the file where your FastAPI application is defined:

```bash
uvicorn main:app --reload
```

This will launch the server, which listens on `http://127.0.0.1:8000`.

### 4. API Documentation

FastAPI automatically generates an interactive API documentation interface. You can access it in your browser at the following URL:

```bash
http://127.0.0.1:8000/docs
```

Here, you can test the API endpoints.

### 5. Sending a Prediction Request

To make a prediction, use a `POST` request to the following URL:

```bash
http://127.0.0.1:8000/predict
```

You can send a sample request with the following JSON payload:

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

1. **`ModuleNotFoundError: No module named 'fastapi'`**: Make sure the required packages are installed in your Python environment (`pip install fastapi uvicorn`).

2. **`ValueError: X has 8 features, but RandomForestClassifier is expecting 9 features`**: The model expects more input data. Check the model's functionality and the input data.

3. **Port Conflict**: If the `8000` port is already in use, try using a different port:

   ```bash
   uvicorn main:app --reload --port 8080
   ```
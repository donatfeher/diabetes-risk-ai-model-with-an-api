# FastAPI-based Machine Learning Model Documentation

[Magyar verzió (Hungarian)](README_hu.md)

This documentation provides clear steps for running the FastAPI-based machine learning model API. Follow these steps to set up and run the code in your environment.

## Project Structure

Here's a high-level overview of the project structure, showing key files and folders:

```plaintext
├── README.md                # Project documentation in English
├── main.py                  # Entry point for the FastAPI server
├── requirements.txt         # List of dependencies
├── data                     # Contains data files
│   ├── diabetes.csv         # Original dataset
│   └── processed_data.csv   # Processed dataset
├── models                   # Directory for models
│   ├── final_model.pkl      # Serialized best model
│   └── model_pipeline.py    # Script for training and saving the model
├── src                      # Contains helper scripts
│   └── data_pipeline.py     # Data preprocessing functions
└── docs                     # Documentation files and images
    ├── all_cicd_pipeline_success.png
    ├── final_result_prediction_success.png
    └── mlops_kurzus_tematikaja.txt
```

## 1. Clone or Download the Code Locally

Ensure that `main.py` and the required model files (e.g., `final_model.pkl`) are available in the directory where you're working.

## 2. Install Required Packages

Navigate to the directory where `requirements.txt` is located, and install the necessary Python packages with the following command:

```bash
pip install -r requirements.txt
```

Alternatively, you can install each required package individually:

```bash
pip install fastapi==0.112.2 uvicorn==0.30.0 scikit-learn==1.5.1 pandas==2.2.2 numpy==1.25.0 xgboost==2.1.1
```

## 3. Run the FastAPI Server

Start the FastAPI server with this command, making sure `main` is the name of the file where your FastAPI application is defined:

```bash
uvicorn main:app --reload
```

This will launch the server, which listens on `http://127.0.0.1:8000`.

## 4. API Documentation

FastAPI automatically generates an interactive API documentation interface. Access it in your browser at:

```plaintext
http://127.0.0.1:8000/docs
```

Here, you can test the API endpoints and see the expected input and output formats.

## 5. Sending a Prediction Request

To make a prediction, use a `POST` request to the following URL:

```plaintext
http://127.0.0.1:8000/predict
```

Send a sample request with this JSON payload:

```json
{
  "Pregnancies": 6,
  "Glucose": 148,
  "BloodPressure": 72,
  "SkinThickness": 35,
  "Insulin": 0,  
  "BMI": 33.6,
  "DiabetesPedigreeFunction": 0.627,
  "Age": 50
}
```

```
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{
    "Pregnancies": 6,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 0,  
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50
}'
```


## 6. Server Response

The server will return the prediction in JSON format, such as:

```json
{
  "prediction": 1
}
```

## Common Errors and Solutions

1. **`ModuleNotFoundError: No module named 'fastapi'`**: Ensure that all required packages are installed correctly by following Step 2.

2. **`ValueError: X has 8 features, but the model expects a different number`**: This error indicates that the model expects more input features than provided. Double-check the model requirements and input data format.

3. **Port Conflict**: If port `8000` is already in use, try a different port with the following command:

   ```bash
   uvicorn main:app --reload --port 8080
   ```

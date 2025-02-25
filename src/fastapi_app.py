import joblib
import pandas as pd
import numpy as np
import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Get the current directory (where fastapi_app.py is located)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the best trained model using relative paths
MODEL_PATH = os.path.join(BASE_DIR, "../models/best_decision_tree.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "../models/scaler.pkl")

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logger.info("Model and scaler loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model or scaler: {e}")
    raise HTTPException(status_code=500, detail="Model loading failed.")

# FastAPI instance
app = FastAPI()

# Define request schema
class InputData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Class labels mapping
class_labels = {0: "setosa", 1: "versicolor", 2: "virginica"}

@app.get("/")
def home():
    """Root endpoint to check API status"""
    logger.info("Root endpoint accessed.")
    return {"message": "FastAPI is running! Use /docs for API documentation."}

@app.post("/predict")
def predict_species(data: InputData):
    """Predict the species of an iris flower"""
    try:
        # Convert input data to numpy array
        input_array = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
        logger.info(f"Received input: {input_array.tolist()}")

        # Scale input data
        input_scaled = scaler.transform(input_array)

        # Predict
        prediction = model.predict(input_scaled)[0]
        species = class_labels[int(prediction)]  # Convert number to class name
        logger.info(f"Prediction: {species}")

        return {"species_prediction": species}

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction error.")

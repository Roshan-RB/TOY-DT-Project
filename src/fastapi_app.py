import joblib
import pandas as pd
import numpy as np
import os
from fastapi import FastAPI
from pydantic import BaseModel

# Get the current directory (where fastapi_app.py is located)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the best trained model using a relative path
MODEL_PATH = os.path.join(BASE_DIR, "../models/best_decision_tree.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "../models/scaler.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

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

@app.post("/predict")
def predict_species(data: InputData):
    """Predict the species of an iris flower"""
    
    # Convert input data to numpy array
    input_array = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])

    # Scale input data
    input_scaled = scaler.transform(input_array)

    # Predict
    prediction = model.predict(input_scaled)[0]
    # If prediction is already a string, return it
    if isinstance(prediction, str):
        species = prediction
    else:
        species = class_labels[int(prediction)]  # Convert number to class name

    return {"species_prediction": species}

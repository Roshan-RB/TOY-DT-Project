import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# Load the best trained model
model = joblib.load(r"C:\Users\rbhaskar\Desktop\decision_tree_project\models\best_decision_tree.pkl")
scaler = joblib.load(r"C:\Users\rbhaskar\Desktop\decision_tree_project\models\scaler.pkl")

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

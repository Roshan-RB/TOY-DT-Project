import streamlit as st
import requests

st.title("Iris Flower Species Prediction 🌸")

# FastAPI URL (Update this with your deployed Render URL)
FASTAPI_URL = "https://toy-dt-project.onrender.com/predict"

# Input fields
sepal_length = st.number_input("Sepal Length", min_value=4.0, max_value=8.0, value=5.8)
sepal_width = st.number_input("Sepal Width", min_value=2.0, max_value=4.5, value=3.0)
petal_length = st.number_input("Petal Length", min_value=1.0, max_value=7.0, value=4.0)
petal_width = st.number_input("Petal Width", min_value=0.1, max_value=2.5, value=1.2)

# Button for prediction
if st.button("Predict Species"):
    input_data = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width
    }
    
    try:
        response = requests.post(FASTAPI_URL, json=input_data, timeout=10)  # Added timeout to avoid hanging requests

        if response.status_code == 200:
            species = response.json().get("species_prediction", "Unknown")
            st.success(f"Predicted Species: **{species}**")
        else:
            st.error("Failed to get prediction. Please check the FastAPI server.")

    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to FastAPI: {e}")

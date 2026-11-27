

import streamlit as st
import pandas as pd
import pickle
import zipfile
import os

# Extract the ZIP model
zip_path = "income_prediction.zip"
extract_path = "/tmp/income_model"

with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_path)

# Find the PKL file
model_path = None

for root, dirs, files in os.walk(extract_path):
    for filename in files:
        if filename == "income_prediction.pkl":
            model_path = os.path.join(root, filename)
            break
    if model_path:
        break

if model_path is None:
    st.error("Model file not found inside ZIP!")
    st.stop()

# Load the trained model
with open(model_path, "rb") as file:
    model = pickle.load(file)

st.success("ML model loaded successfully!")


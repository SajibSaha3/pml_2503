import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("iris_model.pkl", "rb"))

# Page configuration
st.set_page_config(
    page_title="Flower Prediction System",
    page_icon="🌸",
    layout="centered",
)

# Title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🌸 Flower Prediction System 🌸</h1>", unsafe_allow_html=True)
st.write("---")

# Input fields in columns
col1, col2 = st.columns(2)

with col1:
    sep_len = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
    sep_wid = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)

with col2:
    pe_len = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
    pe_wid = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)

st.write("---")

# Predict button
if st.button("Predict Flower 🌺"):
    feature = [sep_len, sep_wid, pe_len, pe_wid]
    pred = model.predict([feature])

    # Optional: map numbers to flower names if model gives numbers
    classes = ["setosa", "versicolor", "virginica"]
    flower_name = classes[int(pred[0])] if isinstance(pred[0], (int, float)) else pred[0]

    st.success(f"Predicted Flower: 🌸 {flower_name} 🌸")

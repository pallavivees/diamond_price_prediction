import streamlit as st
import pickle
import pandas as pd

# Load pipeline
with open("diamond_knn_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("💎 Diamond Price Prediction App")

# Collect user input
carat = st.number_input("Carat", min_value=0.1, max_value=5.0, step=0.01)
cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
color = st.selectbox("Color", ["D", "E", "F", "G", "H", "I", "J"])
clarity = st.selectbox("Clarity", ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2"])
depth = st.number_input("Depth", min_value=50.0, max_value=70.0, step=0.1)
table = st.number_input("Table", min_value=50.0, max_value=70.0, step=0.1)
x = st.number_input("X (mm)", min_value=0.0, step=0.1)
y = st.number_input("Y (mm)", min_value=0.0, step=0.1)
z = st.number_input("Z (mm)", min_value=0.0, step=0.1)

if st.button("Predict Price"):
    # Build DataFrame with same column names as training
    input_df = pd.DataFrame({
        "carat": [carat],
        "cut": [cut],
        "color": [color],
        "clarity": [clarity],
        "depth": [depth],
        "table": [table],
        "x": [x],
        "y": [y],
        "z": [z]
    })

    prediction = model.predict(input_df)
    st.success(f"Estimated Diamond Price: ${prediction[0]:,.2f}")
import streamlit as st
import numpy as np
import pickle
import tensorflow as tf

# Load model and scaler
model = tf.keras.models.load_model('../tf_bridge_model.h5')
with open('../preprocess_pipeline.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("Bridge Load Capacity Predictor")

# Inputs
span = st.number_input("Span (ft)", min_value=0)
width = st.number_input("Deck Width (ft)", min_value=0)
age = st.number_input("Age (years)", min_value=0)
lanes = st.number_input("Number of Lanes", min_value=1)
condition = st.selectbox("Condition Rating (1–5)", [1, 2, 3, 4, 5])
material = st.selectbox("Material", ["Steel", "Concrete", "Composite"])

# One-hot encode material
material_concrete = 1 if material == "Concrete" else 0
material_composite = 1 if material == "Composite" else 0

# Create input array
features = np.array([[span, width, age, lanes, condition, material_concrete, material_composite]])
features_scaled = scaler.transform(features)

# Predict
if st.button("Predict Max Load (Tons)"):
    prediction = model.predict(features_scaled)
    st.success(f"Predicted Max Load Capacity: {prediction[0][0]:.2f} tons")

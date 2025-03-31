import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('tf_bridge_model.h5')

# Load the saved scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Streamlit app title
st.title("Bridge Load Capacity Predictor")

# User input form
span = st.number_input('Bridge Span (feet)', min_value=0.0, format="%.2f")
deck_width = st.number_input('Deck Width (feet)', min_value=0.0, format="%.2f")
age = st.number_input('Age of Bridge (years)', min_value=0)
lanes = st.number_input('Number of Lanes', min_value=1, step=1)
condition = st.slider('Condition Rating (1 = Poor, 5 = Excellent)', min_value=1, max_value=5, step=1)
material = st.selectbox('Material', ['Steel', 'Concrete', 'Composite'])

# One-hot encoding for 'Material'
material_encoding = {
    'Steel': [0, 0],
    'Concrete': [1, 0],
    'Composite': [0, 1]
}
material_features = material_encoding[material]

# Combine input features into an array
features = np.array([[span, deck_width, age, lanes, condition] + material_features])
features_scaled = scaler.transform(features)

# Make prediction
if st.button("Predict Load Capacity"):
    prediction = model.predict(features_scaled)
    st.success(f"Predicted Maximum Load: **{prediction[0][0]:.2f} tons**")

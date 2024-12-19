import os
import tensorflow as tf
import pickle
import numpy as np
import streamlit as st

# Disable GPU if necessary
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load model and scalers
def load_model_and_scalers():
    global model, input_scaler, output_scaler
    model = tf.keras.models.load_model('biochar_model.h5')
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
    with open('biochar_scalers.pkl', 'rb') as f:
        input_scaler, output_scaler = pickle.load(f)

load_model_and_scalers()

# Streamlit UI
st.title("Biochar Prediction")
st.write("Enter the characteristics of feedstock to predict biochar yield and properties:")

# Collect input data
temperature = st.number_input('Temperature (°C)', min_value=-100.0, max_value=1000.0, value=300.0)
residence_time = st.number_input('Residence time (min)', min_value=1.0, max_value=300.0, value=30.0)
heating_rate = st.number_input('Heating rate (°C/min)', min_value=0.1, max_value=100.0, value=10.0)
cellulose = st.number_input('Cellulose content (%)', min_value=0.0, max_value=100.0, value=40.0)
hemicellulose = st.number_input('Hemicellulose content (%)', min_value=0.0, max_value=100.0, value=30.0)
lignin = st.number_input('Lignin content (%)', min_value=0.0, max_value=100.0, value=20.0)
extractives = st.number_input('Extractives (%)', min_value=0.0, max_value=100.0, value=5.0)
moisture = st.number_input('Moisture content (%)', min_value=0.0, max_value=100.0, value=10.0)
fixed_carbon = st.number_input('Fixed carbon (%)', min_value=0.0, max_value=100.0, value=30.0)
volatile_matter = st.number_input('Volatile matter (%)', min_value=0.0, max_value=100.0, value=50.0)
ash = st.number_input('Ash content (%)', min_value=0.0, max_value=100.0, value=5.0)
carbon = st.number_input('C content (%)', min_value=0.0, max_value=100.0, value=50.0)
hydrogen = st.number_input('H content (%)', min_value=0.0, max_value=100.0, value=6.0)
oxygen = st.number_input('O content (%)', min_value=0.0, max_value=100.0, value=40.0)
nitrogen = st.number_input('N content (%)', min_value=0.0, max_value=100.0, value=1.0)

# Create a list of input features (remove sulfur if it is extra)
input_features = [
    temperature, residence_time, heating_rate, cellulose, hemicellulose, lignin,
    extractives, moisture, fixed_carbon, volatile_matter, ash, carbon,
    hydrogen, oxygen, nitrogen
]

# Make prediction when button is clicked
if st.button('Predict'):
    try:
        # Convert input features to numpy array
        input_data = np.array([input_features])
        # Scale input data
        input_scaled = input_scaler.transform(input_data)
        # Make prediction
        prediction_scaled = model.predict(input_scaled)
        # Inverse scale prediction
        prediction = output_scaler.inverse_transform(prediction_scaled)
        
        # Display results
        st.write("### Prediction Results:")
        st.write(f"**Biochar Yield (%)**: {prediction[0][0]}")
        st.write(f"**HHV (MJ/kg)**: {prediction[0][1]}")
        st.write(f"**Energy Yield (%)**: {prediction[0][2]}")
        st.write(f"**Fixed Carbon**: {prediction[0][3]}")
        st.write(f"**Volatile Matter**: {prediction[0][4]}")
        st.write(f"**Ash**: {prediction[0][5]}")
        st.write(f"**C**: {prediction[0][6]}")
        st.write(f"**H**: {prediction[0][7]}")
        st.write(f"**O**: {prediction[0][8]}")
        st.write(f"**N**: {prediction[0][9]}")
    except Exception as e:
        st.write(f"Error: {str(e)}")

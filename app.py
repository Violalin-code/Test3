import subprocess
import sys
import joblib
import logging
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Force install scikit-learn if not found
try:
    import sklearn
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    import sklearn

import gradio as gr
import pandas as pd

# Load the pipeline
try:
    full_pipeline = joblib.load('model_with_preprocessing.pkl')
    logger.info("Pipeline loaded successfully")
    logger.info(f"Pipeline keys: {list(full_pipeline.keys())}")
except Exception as e:
    logger.error(f"Error loading pipeline: {e}")
    sys.exit(1)

def predict_biochar(Fixed_carbon, Volatile_matter, Ash, C, H, O, N, S,
                    Cellulose, Hemicellulose, Lignin, Residence_time, 
                    Temperature, Heating_rate, Type_of_Feedstock):
    try:
        # Create input DataFrame with exact column names
        input_data = pd.DataFrame({
            'Fixed carbon': [float(Fixed_carbon)],
            'Volatile matter': [float(Volatile_matter)],
            'Ash': [float(Ash)],
            'C': [float(C)],
            'H': [float(H)],
            'O': [float(O)],
            'N': [float(N)],
            'S': [float(S)],
            'Cellulose': [float(Cellulose)],
            'Hemicellulose': [float(Hemicellulose)],
            'Lignin': [float(Lignin)],
            'Residence time (min)': [float(Residence_time)],
            'Temperature (°C)': [float(Temperature)],
            'Heating rate (°C/min)': [float(Heating_rate)],
            'Type of Feedstock': [Type_of_Feedstock]
        })
        
        logger.info("Input data created successfully")
        logger.info(f"Input columns: {list(input_data.columns)}")
        logger.info(f"Input data shape: {input_data.shape}")
        
        # Instead of using the preprocessor directly, let's recreate the pipeline
        numeric_features = ['Fixed carbon', 'Volatile matter', 'Ash', 'C', 'H', 'O', 'N', 'S', 
                          'Cellulose', 'Hemicellulose', 'Lignin', 'Residence time (min)', 
                          'Temperature (°C)', 'Heating rate (°C/min)']
        categorical_features = ['Type of Feedstock']
        
        # Create a new preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(drop='first', sparse=False), categorical_features)
            ])
        
        # Try to use the loaded model's components
        if isinstance(full_pipeline['preprocessor'], ColumnTransformer):
            # If we can, use the fitted preprocessor
            preprocessed_data = full_pipeline['preprocessor'].transform(input_data)
        else:
            # Otherwise use our new preprocessor
            preprocessed_data = preprocessor.fit_transform(input_data)
            
        logger.info(f"Preprocessed data shape: {preprocessed_data.shape}")
        
        # Make predictions using the model
        predictions = full_pipeline['model'].predict(preprocessed_data)
        logger.info("Predictions generated successfully")
        
        predictions = predictions.flatten() if isinstance(predictions, np.ndarray) else predictions
        return tuple(float(x) for x in predictions)
    
    except ValueError as e:
        logger.error(f"Value error in input processing: {str(e)}")
        return tuple(["Error: Please check input values"] * 11)
    except Exception as e:
        logger.error(f"Unexpected error in prediction: {str(e)}")
        logger.error(f"Error details: {str(e)}")
        return tuple(["Error: Prediction failed"] * 11)

# Gradio Interface
interface = gr.Interface(
    fn=predict_biochar,
    inputs=[
        gr.Number(label="Fixed carbon"),
        gr.Number(label="Volatile matter"),
        gr.Number(label="Ash"),
        gr.Number(label="C"),
        gr.Number(label="H"),
        gr.Number(label="O"),
        gr.Number(label="N"),
        gr.Number(label="S"),
        gr.Number(label="Cellulose"),
        gr.Number(label="Hemicellulose"),
        gr.Number(label="Lignin"),
        gr.Number(label="Residence time (min)"),
        gr.Number(label="Temperature (°C)"),
        gr.Number(label="Heating rate (°C/min)"),
        gr.Dropdown(
            choices=[
                'Corncob', 'Corn stover', 'Bagasse', 'Cocopeat', 'Coconut shell', 
                'Coconut fiber', 'Wheat straw', 'Rice husk', 'Rice Straw', 'Pine', 
                'Pinewood sawdust', 'Pine wood', 'Bamboo', 'Orange Bagasse', 
                'Orange pomace', 'Rapeseed oil cake', 'Rape stalk', 'Cassava stem', 
                'Cassava rhizome', 'Cotton stalk', 'Palm kernel shell', 'Wood stem', 
                'Wood bark', 'Agro food waste', 'Agro-food waste', 'Canola hull', 
                'Oat hull', 'Straw pallet', 'Vine pruning', 'Poultry litter', 
                'Hinoki cypress'
            ],
            label="Type of Feedstock"
        )
    ],
    outputs=[
        gr.Number(label="Biochar Yield (%)"),
        gr.Number(label="HHV (MJ/kg)"),
        gr.Number(label="Energy Yield (%)"),
        gr.Number(label="Fixed Carbon (%)"),
        gr.Number(label="Volatile Matter (%)"),
        gr.Number(label="Ash (%)"),
        gr.Number(label="C (%)"),
        gr.Number(label="H (%)"),
        gr.Number(label="O (%)"),
        gr.Number(label="N (%)"),
        gr.Number(label="S (%)")
    ],
    title="Biochar Prediction",
    description="Enter the biomass characteristics to predict biochar properties. All percentage inputs should be in %."
)

if __name__ == "__main__":
    interface.launch(share=True)

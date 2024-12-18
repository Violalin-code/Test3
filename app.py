import subprocess
import sys
import joblib
import logging
import numpy as np

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

# Load and inspect the pipeline
try:
    full_pipeline = joblib.load('model_with_preprocessing.pkl')
    logger.info("Pipeline loaded successfully")
    logger.info(f"Pipeline type: {type(full_pipeline)}")
    
    # Debug information about pipeline structure
    if isinstance(full_pipeline, dict):
        logger.info(f"Pipeline keys: {list(full_pipeline.keys())}")
        # Common key variations
        regressor_keys = ['regressor', 'model', 'estimator', 'classifier', 'rf', 'random_forest']
        preprocessor_keys = ['preprocessor', 'preprocess', 'preprocessing', 'transform', 'transformer']
        
        # Find the actual keys
        regressor_key = next((k for k in full_pipeline.keys() if k in regressor_keys), None)
        preprocessor_key = next((k for k in full_pipeline.keys() if k in preprocessor_keys), None)
        
        if regressor_key:
            logger.info(f"Found regressor with key: {regressor_key}")
        else:
            logger.error("No regressor key found!")
            
        if preprocessor_key:
            logger.info(f"Found preprocessor with key: {preprocessor_key}")
        else:
            logger.error("No preprocessor key found!")
    else:
        logger.info(f"Pipeline attributes: {dir(full_pipeline)}")
except Exception as e:
    logger.error(f"Error loading pipeline: {e}")
    sys.exit(1)

def predict_biochar(Fixed_carbon, Volatile_matter, Ash, C, H, O, N, S,
                    Cellulose, Hemicellulose, Lignin, Residence_time, 
                    Temperature, Heating_rate, Type_of_Feedstock):
    try:
        # Convert inputs to float
        numeric_inputs = {
            'Fixed carbon': float(Fixed_carbon),
            'Volatile matter': float(Volatile_matter),
            'Ash': float(Ash),
            'C': float(C),
            'H': float(H),
            'O': float(O),
            'N': float(N),
            'S': float(S),
            'Cellulose': float(Cellulose),
            'Hemicellulose': float(Hemicellulose),
            'Lignin': float(Lignin),
            'Residence time (min)': float(Residence_time),
            'Temperature (°C)': float(Temperature),
            'Heating rate (°C/min)': float(Heating_rate),
            'Type of Feedstock': Type_of_Feedstock
        }
        
        # Create input DataFrame
        input_data = pd.DataFrame([numeric_inputs])
        logger.info("Input data created successfully")
        
        # Check if pipeline is a dictionary and handle different key names
        if isinstance(full_pipeline, dict):
            logger.info("Pipeline is a dictionary, using components directly")
            logger.info(f"Available keys: {list(full_pipeline.keys())}")
            
            # Find the correct keys
            regressor_key = next((k for k in full_pipeline.keys() if k.lower() in ['regressor', 'model', 'estimator', 'classifier', 'rf', 'random_forest']), None)
            preprocessor_key = next((k for k in full_pipeline.keys() if k.lower() in ['preprocessor', 'preprocess', 'preprocessing', 'transform', 'transformer']), None)
            
            if not regressor_key or not preprocessor_key:
                raise KeyError(f"Missing required components. Found keys: {list(full_pipeline.keys())}")
            
            preprocessed_data = full_pipeline[preprocessor_key].transform(input_data)
            predictions = full_pipeline[regressor_key].predict(preprocessed_data)
        else:
            logger.info("Using full pipeline for prediction")
            predictions = full_pipeline.predict(input_data)
        
        logger.info("Predictions generated successfully")
        
        # Format predictions
        predictions = predictions.flatten() if isinstance(predictions, np.ndarray) else predictions
        return tuple(float(x) for x in predictions)
    
    except ValueError as e:
        logger.error(f"Value error in input processing: {str(e)}")
        return tuple(["Error: Please check input values"] * 11)
    except KeyError as e:
        logger.error(f"Pipeline structure error: {str(e)}")
        return tuple(["Error: Pipeline configuration issue"] * 11)
    except Exception as e:
        logger.error(f"Unexpected error in prediction: {str(e)}")
        return tuple(["Error: Prediction failed"] * 11)

# Gradio Interface remains the same
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
        gr.Number(label="Fixed Carbon"),
        gr.Number(label="Volatile Matter"),
        gr.Number(label="Ash"),
        gr.Number(label="C"),
        gr.Number(label="H"),
        gr.Number(label="O"),
        gr.Number(label="N"),
        gr.Number(label="S")
    ],
    title="Biochar Prediction"
)

if __name__ == "__main__":
    interface.launch()

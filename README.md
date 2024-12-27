# Biochar Yield and Composition Prediction System: A Machine Learning Approachm 🪵🔥

## Overview
This project focuses on developing a Biochar Yield and Composition Prediction System using machine learning techniques. Biochar production through pyrolysis of organic waste offers a sustainable approach to reducing dependence on conventional energy sources and mitigating global warming. To address the challenges posed by existing prediction models—such as computational inefficiency, complexity, and limited accuracy for unseen scenario. This project utilizes advanced machine learning models trained on feedstock compositions and pyrolysis process conditions.

## Table of Contents
- [ML Model](#ML-Model)
- [Features](#Features)
- [How to Run the Project](#How-to-Run)
- [Project Files](#Project-Files)
- [Project Workflow](#Project-Workflow)

## ML Model
The machine learning pipeline includes various regression models trained to predict biochar yield and compositions based on feedstock characteristics and pyrolysis conditions. Emphasis is placed on optimizing model performance for generalizability to unseen data.

## Model Details
- Input Features:
Feedstock compositions (e.g., cellulose, lignin, fixed carbon, volatile matter, etc.)
Pyrolysis process conditions (e.g., temperature, residence time, heating rate, etc.)
Output Features:
Biochar yield (%)
Elemental composition (e.g., C, H, O, N, ash content)
- Models Evaluated:
Linear Regression
Random Forest Regressor
Gradient Boosting Machines (GBM)
Support Vector Regressor (SVR)
Artificial Neural Networks (ANN)

## Features
- Accurate Predictions: Predicts biochar yield and composition for diverse feedstock and pyrolysis conditions.
- Efficient Computation: Optimized machine learning models reduce computational demands.
- User-Friendly Interface: Incorporates an intuitive interface powered by Streamlit for input customization and real-time predictions.
- Scalable Design: Supports a wide range of organic waste feedstocks and pyrolysis scenarios.

## How to Run the Project
- Install Required Libraries: Install the following libraries:
- TensorFlow (for ANN)
- scikit-learn
- Streamlit
- NumPy
Use the command:
- bash
- Copy code
- pip install tensorflow scikit-learn streamlit numpy  
- Prepare Dataset and Model Files: Ensure the dataset is preprocessed and models are saved as .pkl or .h5 files in the project directory.

Streamlit : https://d8dmpo3gdtunubwa9bgigr.streamlit.app/

## Project Files
- biochar_dataset.csv: Dataset containing feedstock properties and pyrolysis conditions.
- model.pkl: Trained machine learning model for predictions.
- scaler.pkl: Preprocessing scaler for input features.
- script_name.py: Streamlit-based application script.

## Project Workflow
### 1. Data Preprocessing
- Scaling and Normalization: Feedstock compositions and process conditions are standardized for model training.
- Feature Engineering: Relevant features are selected to enhance model performance.
### 2. Model Training and Validation
- Multiple models are trained and evaluated on the dataset using metrics such as mean absolute error (MAE) and root mean squared error (RMSE).
- The best-performing model is selected and saved for deployment.
### 3. Prediction and Deployment
- nInput data from the Streamlit interface is preprocessed using the saved scaler.
- Predictions are generated using the trained model and displayed in an interactive format.

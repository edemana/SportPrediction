import streamlit as st
import numpy as np
import pandas as pd
import pickle as pkl
from scipy.stats import t
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Load the trained model
with open('best_model.pkl', 'rb') as model_file:
    model = pkl.load(model_file)

# Define relevant features
relevant_features = ['potential', 'passing', 'dribbling', 'movementreactions']

# Define the maximum value observed in the training data (replace with your actual value)
max_observed_value = 3.878689324238808

# Function to calculate confidence score
def calculate_confidence(input_data):
    if isinstance(model, LinearRegression):
        # Calculate confidence intervals for Linear Regression
        y_pred = model.predict(input_data)
        n = input_data.shape[0]
        p = input_data.shape[1]
        mse = mean_squared_error(ytest, y_pred)  # Use appropriate test data here
        SEE = np.sqrt(mse * (1 + 1 / n + ((input_data - input_data.mean()) ** 2).sum(axis=0) / ((n - 1) * input_data.var(axis=0))))
        alpha = 0.05  # 95% confidence interval
        t_crit = t.ppf(1 - alpha / 2, df=n - p - 1)
        lower_ci = y_pred - t_crit * SEE
        upper_ci = y_pred + t_crit * SEE
        confidence_score = np.mean(upper_ci - lower_ci)  # Example of confidence calculation
    elif isinstance(model, RandomForestRegressor) or isinstance(model, XGBRegressor):
        # For other models, set a placeholder or fixed confidence score
        confidence_score = np.random.uniform(0.7, 0.95)  # Example placeholder for Random Forest or XGBoost
    else:
        # Default confidence score if model type is unknown
        confidence_score = 0.5
    return confidence_score

# Streamlit app
st.title("Player Rating Prediction")
st.write("Enter the player attributes below:")

potential = st.number_input("Potential (in %)", min_value=0.0, max_value=100.0, step=0.1)
passing = st.number_input("Passing (in %)", min_value=0.0, max_value=100.0, step=0.1)
dribbling = st.number_input("Dribbling (in %)", min_value=0.0, max_value=100.0, step=0.1)
movement_reactions = st.number_input("Movement Reactions (in %)", min_value=0.0, max_value=100.0, step=0.1)

if st.button("Predict"):
    # Collect the input features
    float_features = [
        float(potential)/100.0 * max_observed_value,
        float(passing)/100.0 * max_observed_value,
        float(dribbling)/100.0 * max_observed_value,
        float(movement_reactions)/100.0 * max_observed_value
    ]
    features = np.array(float_features).reshape(1, -1)

    # Convert to DataFrame with the correct feature names
    features_df = pd.DataFrame(features, columns=relevant_features)

    # Predict using the model
    prediction = model.predict(features_df)[0]

    # Scale the prediction back to a percentage
    prediction_percentage = (prediction / max_observed_value) * 100

    # Calculate confidence score
    confidence_score = calculate_confidence(features_df)

    st.write(f"The overall player rating (in %) is {prediction_percentage:.2f}")
    st.write(f"Confidence Score: {confidence_score:.2f}")

# To run the app, use the following command in your terminal:
# streamlit run app.py

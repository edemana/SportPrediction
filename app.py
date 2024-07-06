from flask import Flask, request, jsonify, render_template
import pickle as pkl
import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

from EDEMANAGBAH_SportsPrediction import ytest

# Create Flask app
app = Flask(__name__)

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

@app.route("/")
def Home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect the input features
        float_features = [float(request.form[feature])/100.0 * max_observed_value for feature in relevant_features]
        features = np.array(float_features).reshape(1, -1)

        # Convert to DataFrame with the correct feature names
        features_df = pd.DataFrame(features, columns=relevant_features)

        # Predict using the model
        prediction = model.predict(features_df)[0]

        # Scale the prediction back to a percentage
        prediction_percentage = (prediction / max_observed_value) * 100

        # Calculate confidence score
        confidence_score = calculate_confidence(features_df)

        # Render the result
        return render_template("index.html", prediction_text=f"The overall player rating (in %) is {prediction_percentage:.2f}", confidence_text=f"Confidence Score: {confidence_score:.2f}")
    except Exception as e:
        # Handle errors if any
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        # Collect the input features
        float_features = [float(request.json[feature])/100.0 * max_observed_value  for feature in relevant_features]
        features = np.array(float_features).reshape(1, -1)

        # Convert to DataFrame with the correct feature names
        features_df = pd.DataFrame(features, columns=relevant_features)

        # Predict using the model
        prediction = model.predict(features_df)[0]

        # Scale the prediction back to a percentage
        prediction_percentage = (prediction / max_observed_value) * 100

        # Calculate confidence score
        confidence_score = calculate_confidence(features_df)

        # Return the prediction as a JSON response
        return jsonify({"prediction": round(prediction_percentage,2), "Confidence Score": round(confidence_score,2)})
    except Exception as e:
        # Handle errors if any
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

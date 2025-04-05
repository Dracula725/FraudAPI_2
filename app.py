from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model or pipeline
model = joblib.load("fraud_detection_model.pkl")

@app.route('/')
def home():
    return "Fraud Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Convert input to DataFrame (assuming preprocessed format)
    input_df = pd.DataFrame([data])

    # Predict
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    return jsonify({
        "prediction": int(prediction),
        "fraud_probability": float(proba)
    })

if __name__ == '__main__':
    app.run(debug=True)

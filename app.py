import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)
MODEL_PATH = os.path.join("models", "churn_pipeline.pkl")
model = joblib.load(MODEL_PATH)

# The fields and options match the IBM Telco Customer Churn dataset.
CATEGORICAL_OPTIONS = {
    "gender": ["Female", "Male"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ]
}

NUMERIC_FIELDS = {
    "SeniorCitizen": (0, 1),
    "tenure": (0, 72),
    "MonthlyCharges": (0, 200),
    "TotalCharges": (0, 10000)
}

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None
    error = None
    submitted = {}

    if request.method == "POST":
        try:
            submitted = request.form.to_dict()
            row = {
                "gender": submitted["gender"],
                "SeniorCitizen": int(submitted["SeniorCitizen"]),
                "Partner": submitted["Partner"],
                "Dependents": submitted["Dependents"],
                "tenure": float(submitted["tenure"]),
                "PhoneService": submitted["PhoneService"],
                "MultipleLines": submitted["MultipleLines"],
                "InternetService": submitted["InternetService"],
                "OnlineSecurity": submitted["OnlineSecurity"],
                "OnlineBackup": submitted["OnlineBackup"],
                "DeviceProtection": submitted["DeviceProtection"],
                "TechSupport": submitted["TechSupport"],
                "StreamingTV": submitted["StreamingTV"],
                "StreamingMovies": submitted["StreamingMovies"],
                "Contract": submitted["Contract"],
                "PaperlessBilling": submitted["PaperlessBilling"],
                "PaymentMethod": submitted["PaymentMethod"],
                "MonthlyCharges": float(submitted["MonthlyCharges"]),
                "TotalCharges": float(submitted["TotalCharges"])
            }

            input_df = pd.DataFrame([row])
            predicted_class = int(model.predict(input_df)[0])
            probability = float(model.predict_proba(input_df)[0][1]) * 100

            if predicted_class == 1:
                prediction = "High churn risk"
            else:
                prediction = "Low churn risk"

        except (ValueError, KeyError) as exc:
            error = "Please enter valid values in all fields."
        except Exception:
            error = "Prediction could not be completed. Check your inputs and model file."

    return render_template(
        "index.html",
        options=CATEGORICAL_OPTIONS,
        numeric_fields=NUMERIC_FIELDS,
        prediction=prediction,
        probability=probability,
        error=error,
        submitted=submitted
    )

if __name__ == "__main__":
    app.run(debug=True)

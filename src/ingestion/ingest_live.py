import pandas as pd
import random
import string
import os
import requests
import logging
from datetime import datetime
import src.utils.log_config as log_config

try:
    STATIC_DATA_PATH = "data/Telco-Customer-Churn.csv"
    static_df = pd.read_csv(STATIC_DATA_PATH)
except Exception as e:
    logging.error(f"[INGESTION] Live ingestion failed: {e}", exc_info=False)
    print(f"Error during live ingestion: {e}")

categorical_cols = static_df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = static_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_values = {col: static_df[col].dropna().unique().tolist() for col in categorical_cols}
numerical_ranges = {col: (static_df[col].min(), static_df[col].max()) for col in numerical_cols}

def generate_customer_id():
    """Generate a random live-data customer ID"""
    suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    return f"LIVE-{suffix}"

def generate_live_data(n=10):
    url = f"https://randomuser.me/api/?results={n}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()["results"]

    live_rows = []

    for user in data:
        # Basic mapping
        gender = "Male" if user["gender"] == "male" else "Female"
        age = user["dob"]["age"]
        senior = 1 if age >= 60 else 0

        row = {
            "customerID": generate_customer_id(),
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": random.choice(["Yes", "No"]),
            "Dependents": random.choice(["Yes", "No"]),
            "tenure": random.randint(1, 72),  # 1–72 months
            "PhoneService": random.choice(["Yes", "No"]),
            "MultipleLines": random.choice(["Yes", "No", "No phone service"]),
            "InternetService": random.choice(["DSL", "Fiber optic", "No"]),
            "OnlineSecurity": random.choice(["Yes", "No", "No internet service"]),
            "OnlineBackup": random.choice(["Yes", "No", "No internet service"]),
            "DeviceProtection": random.choice(["Yes", "No", "No internet service"]),
            "TechSupport": random.choice(["Yes", "No", "No internet service"]),
            "StreamingTV": random.choice(["Yes", "No", "No internet service"]),
            "StreamingMovies": random.choice(["Yes", "No", "No internet service"]),
            "Contract": random.choice(["Month-to-month", "One year", "Two year"]),
            "PaperlessBilling": random.choice(["Yes", "No"]),
            "PaymentMethod": random.choice([
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ]),
            "MonthlyCharges": round(random.uniform(20, 120), 2),
            "TotalCharges": 0.0,
            "Churn": random.choice(["Yes", "No"])
        }

        row["TotalCharges"] = round(row["tenure"] * row["MonthlyCharges"], 2)

        live_rows.append(row)

    df = pd.DataFrame(live_rows)

    raw_dir = "data/raw/live/"
    os.makedirs(raw_dir, exist_ok=True)
    filename = os.path.join(raw_dir, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} live_data.csv")
    df.to_csv(filename, index=False)

    logging.info(f"[INGESTION] Live ingestion successful. Rows: {len(df)}")
    print(f'Live data ingestion complete. Rows: {len(df)}')

    logging.info(f"[INGESTION] Live data saved at: {filename}")

    return df

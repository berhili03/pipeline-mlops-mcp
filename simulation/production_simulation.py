import time
import random
import requests
import numpy as np
import pandas as pd


API_URL = "http://127.0.0.1:8000/predict"

DATA_PATH = "data/telco_churn.csv"

INTERVAL = 5

NORMAL_REQUESTS = 10
DRIFT_REQUESTS = 10


FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


def create_payload(row):

    payload = {
        column: row[column]
        for column in FEATURES
    }

    # Conversion des types NumPy/Pandas
    # vers des types Python compatibles avec JSON
    payload = {
        key: (
            None
            if pd.isna(value)
            else value.item()
            if isinstance(value, np.generic)
            else value
        )
        for key, value in payload.items()
    }

    return payload


def send_request(payload, request_number, phase):

    try:

        response = requests.post(
            API_URL,
            json=payload,
            timeout=10
        )

        print(
            f"[{phase}] "
            f"Request {request_number} | "
            f"Status: {response.status_code} | "
            f"Response: {response.json()}"
        )

    except Exception as e:

        print(
            f"[{phase}] "
            f"Request {request_number} | "
            f"ERROR: {e}"
        )


def main():

    print("=" * 50)
    print("PRODUCTION SIMULATION")
    print("=" * 50)

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset chargé : {df.shape}")
    print()

    # =====================================================
    # PHASE 1 : TRAFIC NORMAL
    # =====================================================

    print("PHASE 1 : TRAFIC NORMAL")
    print("-" * 50)

    for i in range(1, NORMAL_REQUESTS + 1):

        row = df.sample(1).iloc[0]

        payload = create_payload(row)

        send_request(
            payload,
            i,
            "NORMAL"
        )

        time.sleep(INTERVAL)

    # =====================================================
    # PHASE 2 : INJECTION DU DRIFT
    # =====================================================

    print()
    print("PHASE 2 : INJECTION DU DRIFT")
    print("-" * 50)

    for i in range(1, DRIFT_REQUESTS + 1):

        row = df.sample(1).iloc[0]

        payload = create_payload(row)

        # ---------------------------------------------
        # Injection volontaire d'un changement
        # ---------------------------------------------

        payload["tenure"] = random.randint(60, 72)

        payload["MonthlyCharges"] = round(
            random.uniform(100, 120),
            2
        )

        payload["TotalCharges"] = round(
            random.uniform(5000, 9000),
            2
        )

        send_request(
            payload,
            i,
            "DRIFT"
        )

        time.sleep(INTERVAL)

    print()
    print("=" * 50)
    print("SIMULATION TERMINÉE")
    print("=" * 50)


if __name__ == "__main__":
    main()
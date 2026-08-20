import requests
import time
import random


API_URL = "http://127.0.0.1:8000/predict"


def generate_drift_client():

    return {
        "gender": random.choice(["Male", "Female"]),

        "SeniorCitizen": 1,

        "Partner": "No",

        "Dependents": "No",

        # Drift volontaire :
        # valeurs très faibles
        "tenure": random.randint(1, 5),

        "PhoneService": "Yes",

        "MultipleLines": "Yes",

        "InternetService": "Fiber optic",

        "OnlineSecurity": "No",

        "OnlineBackup": "No",

        "DeviceProtection": "No",

        "TechSupport": "No",

        "StreamingTV": "Yes",

        "StreamingMovies": "Yes",

        "Contract": "Month-to-month",

        "PaperlessBilling": "Yes",

        "PaymentMethod": "Electronic check",

        # Drift volontaire :
        # charges très élevées
        "MonthlyCharges": round(
            random.uniform(150, 200), 2
        ),

        "TotalCharges": round(
            random.uniform(100, 500), 2
        )
    }


def send_prediction(client_data):

    try:

        response = requests.post(
            API_URL,
            json=client_data,
            timeout=10
        )

        print(
            f"Status: {response.status_code} | "
            f"Response: {response.text}"
        )

    except Exception as e:

        print(f"Erreur : {e}")


def main():

    print("====================================")
    print(" SIMULATION AVEC DATA DRIFT")
    print("====================================")

    for i in range(20):

        print(f"\nRequête DRIFT #{i + 1}")

        client = generate_drift_client()

        send_prediction(client)

        time.sleep(5)


if __name__ == "__main__":
    main()
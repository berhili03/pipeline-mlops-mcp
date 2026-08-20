import requests
import time
import random


API_URL = "http://127.0.0.1:8000/predict"


def generate_normal_client():

    return {
        "gender": random.choice(["Male", "Female"]),
        "SeniorCitizen": random.choice([0, 0, 0, 1]),
        "Partner": random.choice(["Yes", "No"]),
        "Dependents": random.choice(["Yes", "No"]),
        "tenure": random.randint(1, 72),
        "PhoneService": random.choice(["Yes", "No"]),
        "MultipleLines": random.choice([
            "Yes",
            "No",
            "No phone service"
        ]),
        "InternetService": random.choice([
            "DSL",
            "Fiber optic",
            "No"
        ]),
        "OnlineSecurity": random.choice([
            "Yes",
            "No",
            "No internet service"
        ]),
        "OnlineBackup": random.choice([
            "Yes",
            "No",
            "No internet service"
        ]),
        "DeviceProtection": random.choice([
            "Yes",
            "No",
            "No internet service"
        ]),
        "TechSupport": random.choice([
            "Yes",
            "No",
            "No internet service"
        ]),
        "StreamingTV": random.choice([
            "Yes",
            "No",
            "No internet service"
        ]),
        "StreamingMovies": random.choice([
            "Yes",
            "No",
            "No internet service"
        ]),
        "Contract": random.choice([
            "Month-to-month",
            "One year",
            "Two year"
        ]),
        "PaperlessBilling": random.choice([
            "Yes",
            "No"
        ]),
        "PaymentMethod": random.choice([
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]),
        "MonthlyCharges": round(random.uniform(20, 120), 2),
        "TotalCharges": round(random.uniform(20, 8000), 2)
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
    print(" SIMULATION DE PRODUCTION")
    print("====================================")

    request_number = 1

    while True:

        print(f"\nRequête #{request_number}")

        client = generate_normal_client()

        send_prediction(client)

        request_number += 1

        print("Attente de 5 secondes...")

        time.sleep(5)


if __name__ == "__main__":
    main()
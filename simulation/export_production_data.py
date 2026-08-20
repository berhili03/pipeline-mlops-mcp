import pandas as pd
from database.database import SessionLocal, PredictionLog


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
    "TotalCharges"
]


def export_production_data():

    db = SessionLocal()

    try:

        logs = db.query(PredictionLog).filter(
            PredictionLog.error == 0
        ).all()

        data = []

        for log in logs:

            row = {}

            for feature in FEATURES:
                row[feature] = getattr(log, feature)

            data.append(row)

        df = pd.DataFrame(data)

        df.to_csv(
            "data/production_data.csv",
            index=False
        )

        print("====================================")
        print("PRODUCTION DATA EXPORT")
        print("====================================")

        print(f"Nombre de lignes : {len(df)}")
        print(f"Nombre de colonnes : {len(df.columns)}")

        print("\nFichier créé :")
        print("data/production_data.csv")

        print("\nAperçu :")
        print(df.head())

    finally:

        db.close()


if __name__ == "__main__":
    export_production_data()
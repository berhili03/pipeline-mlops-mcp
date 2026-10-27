from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
import mlflow
import pandas as pd
import os

from database.database import SessionLocal, PredictionLog


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# MLflow
# ============================================================

mlflow.set_tracking_uri("sqlite:///mlflow_old.db")


model = mlflow.pyfunc.load_model(
    "/app/models/telco_churn_model"
)

_mlflow_client = mlflow.MlflowClient()
_champion_version = _mlflow_client.get_model_version_by_alias("telco_churn_model", "champion")
CURRENT_MODEL_VERSION = str(_champion_version.version)

# ============================================================
# Health check
# ============================================================

@app.get("/health")
def health_check():
    return {"status": "ok"}


# ============================================================
# Données client
# ============================================================

class ClientData(BaseModel):

    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


# ============================================================
# Prediction
# ============================================================

@app.post("/predict")
def predict_churn(data: ClientData):

    start_time = time.time()

    db = SessionLocal()

    try:

        # Transformer les données en DataFrame
        df = pd.DataFrame([data.model_dump()])

        # Prédiction
        prediction = model.predict(df)

        prediction_value = int(prediction[0])

        # Temps de réponse
        response_time = time.time() - start_time

        # Création du log
        log = PredictionLog(

            gender=data.gender,
            SeniorCitizen=data.SeniorCitizen,
            Partner=data.Partner,
            Dependents=data.Dependents,
            tenure=data.tenure,
            PhoneService=data.PhoneService,
            MultipleLines=data.MultipleLines,
            InternetService=data.InternetService,
            OnlineSecurity=data.OnlineSecurity,
            OnlineBackup=data.OnlineBackup,
            DeviceProtection=data.DeviceProtection,
            TechSupport=data.TechSupport,
            StreamingTV=data.StreamingTV,
            StreamingMovies=data.StreamingMovies,
            Contract=data.Contract,
            PaperlessBilling=data.PaperlessBilling,
            PaymentMethod=data.PaymentMethod,

            MonthlyCharges=data.MonthlyCharges,
            TotalCharges=data.TotalCharges,

            prediction=prediction_value,

            model_version="3",

            response_time=response_time,

            error=0
        )

        # Sauvegarde
        db.add(log)
        db.commit()
        db.refresh(log)

        return {
            "churn_prediction": prediction_value,
            "model_version": "3",
            "response_time": round(response_time, 4)
        }

    except Exception:

        response_time = time.time() - start_time

        log = PredictionLog(

            gender=data.gender,
            SeniorCitizen=data.SeniorCitizen,
            Partner=data.Partner,
            Dependents=data.Dependents,
            tenure=data.tenure,
            PhoneService=data.PhoneService,
            MultipleLines=data.MultipleLines,
            InternetService=data.InternetService,
            OnlineSecurity=data.OnlineSecurity,
            OnlineBackup=data.OnlineBackup,
            DeviceProtection=data.DeviceProtection,
            TechSupport=data.TechSupport,
            StreamingTV=data.StreamingTV,
            StreamingMovies=data.StreamingMovies,
            Contract=data.Contract,
            PaperlessBilling=data.PaperlessBilling,
            PaymentMethod=data.PaymentMethod,

            MonthlyCharges=data.MonthlyCharges,
            TotalCharges=data.TotalCharges,

            prediction=None,

            model_version="3",

            response_time=response_time,

            error=1
        )

        db.add(log)
        db.commit()

        raise

    finally:

        db.close()


# ============================================================
# Monitoring
# ============================================================

@app.get("/monitoring")
def monitoring():

    db = SessionLocal()

    try:

        logs = db.query(PredictionLog).all()

        total_requests = len(logs)

        if total_requests == 0:
            return {
                "total_requests": 0,
                "successful_predictions": 0,
                "errors": 0,
                "error_rate": 0,
                "average_response_time": 0,
                "churn_predictions": 0,
                "non_churn_predictions": 0,
                "model_version": "3"
            }

        # Nombre d'erreurs
        errors = sum(
            1 for log in logs
            if log.error == 1
        )

        # Nombre de prédictions réussies
        successful_predictions = sum(
            1 for log in logs
            if log.error == 0
        )

        # Taux d'erreur
        error_rate = errors / total_requests

        # Temps de réponse
        response_times = [
            log.response_time
            for log in logs
            if log.response_time is not None
        ]

        average_response_time = (
            sum(response_times) / len(response_times)
            if response_times
            else 0
        )

        # Nombre de churn
        churn_predictions = sum(
            1 for log in logs
            if log.error == 0 and log.prediction == 1
        )

        # Nombre de non-churn
        non_churn_predictions = sum(
            1 for log in logs
            if log.error == 0 and log.prediction == 0
        )

        return {
            "total_requests": total_requests,
            "successful_predictions": successful_predictions,
            "errors": errors,
            "error_rate": round(error_rate, 4),
            "average_response_time": round(average_response_time, 4),
            "churn_predictions": churn_predictions,
            "non_churn_predictions": non_churn_predictions,
            "model_version": "3"
        }

    finally:
        db.close()

"""
Tools MCP pour interroger le monitoring de l'API (predictionlogs.db).

⚠️ À ADAPTER : les noms de colonnes ci-dessous (gender, tenure, MonthlyCharges,
TotalCharges, prediction, model_version, response_time, error, timestamp)
doivent correspondre exactement aux colonnes de ton modèle PredictionLog
défini dans database/database.py. Si un nom diffère, corrige-le ici.
"""

import os
import sys

# Permet d'importer le package `database` situé à la racine du projet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from database.database import SessionLocal, PredictionLog  # noqa: E402


def get_monitoring_stats() -> dict:
    """Calcule les statistiques globales à partir de toutes les prédictions loggées."""
    db = SessionLocal()
    try:
        logs = db.query(PredictionLog).all()
        total = len(logs)

        if total == 0:
            return {"total_requests": 0, "message": "Aucune prédiction enregistrée pour le moment."}

        errors = sum(1 for l in logs if getattr(l, "error", 0) == 1)
        successful = total - errors

        response_times = [
            l.response_time for l in logs if getattr(l, "response_time", None) is not None
        ]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0

        churn = sum(1 for l in logs if getattr(l, "prediction", None) == 1)
        non_churn = successful - churn

        model_versions = [l.model_version for l in logs if getattr(l, "model_version", None)]
        current_version = model_versions[-1] if model_versions else "unknown"

        return {
            "total_requests": total,
            "successful_predictions": successful,
            "errors": errors,
            "error_rate": round(errors / total, 4),
            "average_response_time": round(avg_response_time, 4),
            "churn_predictions": churn,
            "non_churn_predictions": non_churn,
            "model_version": current_version,
        }
    finally:
        db.close()


def query_predictions(limit: int = 10, only_errors: bool = False) -> list:
    """Retourne les dernières prédictions enregistrées, triées de la plus récente à la plus ancienne."""
    db = SessionLocal()
    try:
        query = db.query(PredictionLog)
        if only_errors:
            query = query.filter(PredictionLog.error == 1)

        logs = query.order_by(PredictionLog.timestamp.desc()).limit(limit).all()

        results = []
        for log in logs:
            results.append({
                "timestamp": str(getattr(log, "timestamp", "")),
                "gender": getattr(log, "gender", None),
                "tenure": getattr(log, "tenure", None),
                "MonthlyCharges": getattr(log, "MonthlyCharges", None),
                "TotalCharges": getattr(log, "TotalCharges", None),
                "prediction": getattr(log, "prediction", None),
                "model_version": getattr(log, "model_version", None),
                "response_time": getattr(log, "response_time", None),
                "error": getattr(log, "error", None),
            })
        return results
    finally:
        db.close()
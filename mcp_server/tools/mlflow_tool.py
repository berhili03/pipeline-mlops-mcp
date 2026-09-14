import os
import mlflow

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MLFLOW_DB_PATH = os.path.join(BASE_DIR, "mlflow.db")

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
_client = mlflow.MlflowClient()

MODEL_NAME = "telco_churn_model"


def get_model_info() -> dict:
    """Retourne la version du modèle actuellement en production (alias 'champion' dans MLflow)."""
    try:
        version = _client.get_model_version_by_alias(MODEL_NAME, "champion")
        return {
            "model_name": MODEL_NAME,
            "champion_version": version.version,
            "run_id": version.run_id,
            "status": version.status,
        }
    except Exception as e:
        return {"error": f"Impossible de récupérer le modèle champion : {str(e)}"}
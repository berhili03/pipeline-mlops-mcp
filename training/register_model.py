import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")

run_id = "c70f04aa68e645ed9ab9b23f405924f1"

model_uri = f"runs:/{run_id}/model"

result = mlflow.register_model(
    model_uri,
    "telco_churn_model"
)

print("Version créée :", result.version)
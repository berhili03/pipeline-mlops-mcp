import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow_old.db")

client = mlflow.MlflowClient()

client.set_registered_model_alias(
    "telco_churn_model",
    "champion",
    "3"
)

print("Version 3 est maintenant champion")

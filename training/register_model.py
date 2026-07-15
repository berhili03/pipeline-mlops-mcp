import mlflow

run_id = "ed33500992f5435ab3c3d20088e3611f"  # remplace par ton vrai run_id si différent

model_uri = f"runs:/{run_id}/model"
mlflow.register_model(model_uri, "telco_churn_model")
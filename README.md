# pipeline-mlops-mcp
Pipeline MLOps avec détection automatique de drift et agent conversationnel (MCP) pour interroger l'état du modèle en production

## Charger le modèle depuis le Model Registry

Le meilleur modèle entraîné (RandomForest, n_estimators=200, max_depth=13) est enregistré dans MLflow sous le nom `telco_churn_model`. L'alias `champion` pointe vers la version de référence à utiliser en priorité.

**Résultats de ce modèle :**
- Accuracy : 0.7918
- Precision : 0.6319
- Recall : 0.5187
- F1 Score : 0.5698

**Pour recharger ce modèle en Python :**

```python
import mlflow

model = mlflow.pyfunc.load_model("models:/telco_churn_model@champion")
predictions = model.predict(X_new)
```

**Prérequis :** le serveur de tracking MLflow doit être accessible (variable d'environnement `MLFLOW_TRACKING_URI`, ou exécution locale dans le dossier du projet où se trouve le dossier `mlruns/`).

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# 1. CHARGER LES DONNÉES
# ============================================================

df = pd.read_csv("data/telco_churn.csv")

print("Shape initiale :", df.shape)


# ============================================================
# 2. NETTOYAGE
# ============================================================

# customerID n'est pas utile pour la prédiction
df = df.drop(columns=["customerID"])

# TotalCharges contient quelques valeurs problématiques
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Supprimer les lignes contenant des valeurs manquantes
df = df.dropna()

print("Shape après nettoyage :", df.shape)


# ============================================================
# 3. X / y
# ============================================================

X = df.drop(columns=["Churn"])
y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})


# ============================================================
# 4. TRAIN / TEST
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# 5. IDENTIFIER LES COLONNES
# ============================================================

categorical_columns = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

numeric_columns = X_train.select_dtypes(
    exclude=["object"]
).columns.tolist()


print("Colonnes catégorielles :", categorical_columns)
print("Colonnes numériques :", numeric_columns)


# ============================================================
# 6. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
        (
            "numeric",
            "passthrough",
            numeric_columns
        )
    ]
)


# ============================================================
# 7. MODÈLE
# ============================================================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=13,
    random_state=42
)


# ============================================================
# 8. PIPELINE COMPLET
# ============================================================

pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", model)
    ]
)


# ============================================================
# 9. MLflow
# ============================================================

mlflow.set_experiment("telco_churn_prediction")

with mlflow.start_run() as run:

    # Entraînement
    pipeline.fit(X_train, y_train)

    # Prédictions
    y_pred = pipeline.predict(X_test)

    # Métriques
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Paramètres
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 13)

    # Métriques
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # Sauvegarder le pipeline COMPLET
    mlflow.sklearn.log_model(
        pipeline,
        name="model"
    )

    print("\n===== RESULTATS =====")
    print(f"Run ID    : {run.info.run_id}")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")
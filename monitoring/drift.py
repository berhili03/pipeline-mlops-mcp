import pandas as pd
from sqlalchemy import create_engine

from evidently import Report
from evidently.presets import DataDriftPreset


# ============================================================
# 1. Training data
# ============================================================

training_data = pd.read_csv(
    "data/telco_churn.csv"
)


# ============================================================
# 2. Production data depuis SQLite
# ============================================================

engine = create_engine(
    "sqlite:///prediction_logs.db"
)

production_data = pd.read_sql(
    "SELECT * FROM prediction_logs",
    engine
)


# ============================================================
# 3. Colonnes utilisées par le modèle
# ============================================================

features = [
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


# ============================================================
# 4. Garder uniquement les features
# ============================================================

reference_data = training_data[features].copy()

current_data = production_data[features].copy()


# ============================================================
# 5. Nettoyage des données
# ============================================================

reference_data["TotalCharges"] = pd.to_numeric(
    reference_data["TotalCharges"],
    errors="coerce"
)

current_data["TotalCharges"] = pd.to_numeric(
    current_data["TotalCharges"],
    errors="coerce"
)


reference_data = reference_data.dropna()
current_data = current_data.dropna()


# ============================================================
# 6. Vérification
# ============================================================

print("Training data :", reference_data.shape)
print("Production data :", current_data.shape)


# ============================================================
# 7. Vérifier qu'on a suffisamment de données
# ============================================================

if len(current_data) == 0:
    raise ValueError(
        "Aucune donnée de production disponible."
    )


# ============================================================
# 8. Générer le rapport Evidently
# ============================================================

report = Report(
    [
        DataDriftPreset()
    ]
)


result = report.run(
    reference_data=reference_data,
    current_data=current_data
)


# ============================================================
# 9. Sauvegarder le rapport
# ============================================================

result.save_html(
    "monitoring/drift_report.html"
)


print()
print("===================================")
print("DRIFT REPORT GENERATED")
print("===================================")
print("Report : monitoring/drift_report.html")

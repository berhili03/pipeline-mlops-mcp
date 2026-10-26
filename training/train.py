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

#on sépare les données en deux parties pour pouvoir entraîner
# le modèle puis tester s’il fonctionne bien.


#  X_train = informations des clients
#  y_train = leur vrai Churn

X_train, X_test, y_train, y_test = train_test_split(
    X,                 #les caractéristiques des clients
    y,                 #la réponse à prédire (0 ou 1)
    test_size=0.2,     #On garde 20 % des données pour le test
    random_state=42,   #Sans random_state, Python pourrait choisir 
                        # une séparation différente à chaque exécution et 42
                        #  n'a pas de signification particulière aleatoire
    stratify=y          #conserver approximativement la même proportion de 0 et de 1 dans les deux groupes.
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
            "categorical",                          # Un nom qu'on choisit (juste une étiquette)
            OneHotEncoder(handle_unknown="ignore"), # L'outil de transformation
            categorical_columns                     #Les colonnes ciblées (ex: ['ville', 'sexe'])
        ),
        (
            "numeric",          #Le nom qu'on choisit
            "passthrough",      #L'outil de transformation
            numeric_columns     #Les colonnes ciblées (ex: ['age', 'salaire'])
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
# 8. PIPELINE COMPLET : Son rôle est d'attacher les étapes ensemble pour qu'elles 
# s'exécutent automatiquement à la chaîne, toujours dans le bon ordre.
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
    pipeline.fit(X_train, y_train) #Le pipeline prépare les données et entraîne le RandomForest

    # Prédictions
    y_pred = pipeline.predict(X_test) # L'IA fait ses prédictions sur les 20% de clients de test

    # Métriques
    acc = accuracy_score(y_test, y_pred) # Taux global de bonnes réponses
    prec = precision_score(y_test, y_pred) # Sur tous ceux que l'IA a prédits "Partants", combien sont VRAIMENT partis ?
    rec = recall_score(y_test, y_pred) # Sur tous les VRAIS clients partants, combien l'IA en a-t-elle détectés ?
    f1 = f1_score(y_test, y_pred) # L'équilibre (moyenne) entre Précision et Rappel

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
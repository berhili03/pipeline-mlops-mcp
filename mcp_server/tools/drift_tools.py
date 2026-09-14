"""
Tools MCP pour le Data Drift et les données de production.

⚠️ IMPORTANT : ton script monitoring/drift.py génère actuellement un
drift_report.html (visuel, difficile à parser proprement). Pour que
l'agent puisse lire les résultats, ajoute dans drift.py, juste à côté
de `report.save_html(...)` :

    report.save_json(os.path.join("monitoring", "drift_report.json"))

(Evidently peut exporter le même rapport en JSON structuré en plus du HTML.)
"""

import json
import os

import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DRIFT_JSON_PATH = os.path.join(BASE_DIR, "monitoring", "drift_report.json")
PRODUCTION_DATA_PATH = os.path.join(BASE_DIR, "data", "production_data.csv")


def get_drift_report() -> dict:
    """Lit le rapport Evidently (format v2) et retourne un résumé exploitable par l'agent."""

    if not os.path.exists(DRIFT_JSON_PATH):
        return {
            "error": (
                f"Fichier introuvable: {DRIFT_JSON_PATH}. "
                "Génère d'abord le rapport Evidently."
            )
        }

    with open(DRIFT_JSON_PATH, "r") as f:
        data = json.load(f)

    try:
        metrics = data.get("metrics", [])

        dataset_drift_detected = None
        drift_share = None
        number_of_drifted_columns = None
        number_of_columns = 0
        drifted_columns = []
        all_columns_scores = []

        for metric in metrics:
            config = metric.get("config", {})
            metric_type = config.get("type", "")
            value = metric.get("value")

            # --- Metric global : DriftedColumnsCount ---
            if metric_type.endswith("DriftedColumnsCount"):
                if isinstance(value, dict):
                    number_of_drifted_columns = value.get("count")
                    drift_share = value.get("share")

                drift_share_threshold = config.get("drift_share", 0.5)
                if drift_share is not None:
                    dataset_drift_detected = drift_share > drift_share_threshold

            # --- Metric par colonne : ValueDrift ---
            elif metric_type.endswith("ValueDrift"):
                number_of_columns += 1

                column_name = config.get("column")
                threshold = config.get("threshold", 0.1)
                method = config.get("method")
                score = value if isinstance(value, (int, float)) else None

                is_drifted = score is not None and score > threshold

                column_info = {
                    "column": column_name,
                    "drift_score": round(score, 4) if score is not None else None,
                    "method": method,
                    "threshold": threshold,
                    "drift_detected": is_drifted,
                }
                all_columns_scores.append(column_info)

                if is_drifted:
                    drifted_columns.append(column_info)

        return {
            "dataset_drift_detected": dataset_drift_detected,
            "drift_share": round(drift_share, 4) if drift_share is not None else None,
            "number_of_drifted_columns": number_of_drifted_columns,
            "number_of_columns": number_of_columns,
            "drifted_columns_detail": drifted_columns,
            "all_columns_scores": all_columns_scores,
        }

    except Exception as e:
        return {
            "error": f"Erreur lors de l'analyse du rapport Evidently : {str(e)}",
            "raw_top_level_keys": list(data.keys()),
        }

def get_production_data_summary() -> dict:
    """Statistiques descriptives sur les données de production reçues par l'API."""
    if not os.path.exists(PRODUCTION_DATA_PATH):
        return {"error": f"Fichier introuvable: {PRODUCTION_DATA_PATH}"}

    df = pd.read_csv(PRODUCTION_DATA_PATH)
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    summary = {"row_count": len(df)}

    for col in numeric_cols:
        if col in df.columns:
            summary[col] = {
                "mean": round(float(df[col].mean()), 2),
                "min": round(float(df[col].min()), 2),
                "max": round(float(df[col].max()), 2),
                "std": round(float(df[col].std()), 2),
            }

    return summary
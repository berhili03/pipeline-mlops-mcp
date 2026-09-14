"""
Serveur MCP du pipeline MLOps.

Expose au(x) client(s) MCP (ex: mcp_client/client.py) des tools permettant
à un agent d'interroger le monitoring, les prédictions loggées et le
Data Drift, sans avoir à parcourir les fichiers ou la base à la main.

Lancement manuel (pour debug) :
    python mcp_server/server.py

En pratique, c'est mcp_client/client.py qui lance ce script automatiquement
via stdio — tu n'as pas besoin de le lancer toi-même.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from tools import monitoring_tools, drift_tools, mlflow_tool  # noqa: E402

mcp = FastMCP("pipeline-mlops-mcp")


@mcp.tool()
def get_monitoring_stats() -> dict:
    """Récupère les statistiques de monitoring de l'API : nombre total de requêtes,
    nombre de prédictions réussies, nombre d'erreurs, taux d'erreur, temps de
    réponse moyen, nombre de prédictions churn/non-churn et version du modèle
    actuellement utilisée."""
    return monitoring_tools.get_monitoring_stats()


@mcp.tool()
def query_predictions(limit: int = 10, only_errors: bool = False) -> list:
    """Récupère les dernières prédictions enregistrées dans predictionlogs.db,
    triées de la plus récente à la plus ancienne.

    Args:
        limit: nombre maximum de lignes à retourner (par défaut 10).
        only_errors: si True, ne retourne que les requêtes ayant échoué.
    """
    return monitoring_tools.query_predictions(limit=limit, only_errors=only_errors)


@mcp.tool()
def get_drift_report() -> dict:
    """Analyse le rapport de Data Drift généré par Evidently et retourne un résumé
    structuré : drift détecté au niveau du dataset ou non, part de colonnes en
    drift, et détail des colonnes concernées avec leur score de drift."""
    return drift_tools.get_drift_report()


@mcp.tool()
def get_production_data_summary() -> dict:
    """Retourne des statistiques descriptives (moyenne, min, max, écart-type) sur
    les colonnes numériques clés (tenure, MonthlyCharges, TotalCharges) des
    données de production reçues par l'API, ainsi que le nombre de lignes."""
    return drift_tools.get_production_data_summary()



@mcp.tool()
def get_model_info() -> dict:
    """Version du modèle actuellement en production (alias champion MLflow)."""
    return mlflow_tool.get_model_info()


if __name__ == "__main__":
    mcp.run(transport="stdio")
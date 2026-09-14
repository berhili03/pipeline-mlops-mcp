import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import os

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8001")

st.set_page_config(page_title="MLOps Churn Dashboard", layout="wide")


@st.cache_data(ttl=30)
def fetch(endpoint: str, params: dict | None = None):
    try:
        r = requests.get(f"{BACKEND_URL}{endpoint}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


kpis = fetch("/api/kpis")
drift = fetch("/api/drift")
production = fetch("/api/production")
model_info = fetch("/api/model-info")
health = fetch("/api/health-status")
predictions = fetch("/api/predictions", {"limit": 500})

# ---------- HEADER ----------
status_colors = {"Healthy": "🟢", "Warning": "🟠", "Critical": "🔴"}
status = health.get("status", "Unknown")

col_title, col_status = st.columns([3, 1])
with col_title:
    st.title("📊 MLOps Churn — Dashboard de monitoring")
with col_status:
    st.metric("État du système", f"{status_colors.get(status, '⚪')} {status}")

# st.caption(
#     f"Modèle en production (MLflow champion) : **v{model_info.get('champion_version', '?')}** "
#     f"— version utilisée par l'API : **v{kpis.get('model_version', '?')}**"
# )
# if str(model_info.get("champion_version")) != str(kpis.get("model_version")):
#     st.warning(
#         "⚠️ La version champion MLflow diffère de la version utilisée par l'API en production."
#     )

st.divider()

# ---------- KPIs ----------
st.subheader("Indicateurs clés")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Requêtes totales", kpis.get("total_requests", 0))
c2.metric("Taux d'erreur", f"{kpis.get('error_rate', 0) * 100:.2f}%")
c3.metric("Temps de réponse moyen", f"{kpis.get('average_response_time', 0) * 1000:.1f} ms")
c4.metric("Prédictions churn", kpis.get("churn_predictions", 0))
c5.metric("Prédictions non-churn", kpis.get("non_churn_predictions", 0))
c6.metric("Données production (n)", production.get("row_count", 0))

st.divider()

# ---------- DATA DRIFT ----------
st.subheader("Data Drift")
d1, d2, d3 = st.columns(3)
d1.metric("Drift détecté", "Oui" if drift.get("dataset_drift_detected") else "Non")
d2.metric("Part de colonnes en drift", f"{(drift.get('drift_share') or 0) * 100:.1f}%")
d3.metric(
    "Colonnes en drift",
    f"{int(drift.get('number_of_drifted_columns') or 0)} / {drift.get('number_of_columns', 0)}",
)

all_scores = drift.get("all_columns_scores", [])
if all_scores:
    df_drift = pd.DataFrame(all_scores)
    fig = px.bar(
        df_drift.sort_values("drift_score", ascending=False),
        x="column",
        y="drift_score",
        color="drift_detected",
        color_discrete_map={True: "#e74c3c", False: "#2ecc71"},
        title="Score de drift par colonne",
    )
    fig.add_hline(y=0.1, line_dash="dash", annotation_text="seuil (indicatif)")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------- GRAPHIQUES DE MONITORING ----------
st.subheader("Évolution du monitoring")

if isinstance(predictions, list) and predictions:
    df_pred = pd.DataFrame(predictions)
    df_pred["timestamp"] = pd.to_datetime(df_pred["timestamp"], errors="coerce")
    df_pred = df_pred.dropna(subset=["timestamp"]).sort_values("timestamp")
    df_pred["date"] = df_pred["timestamp"].dt.date

    g1, g2 = st.columns(2)

    with g1:
        req_by_day = df_pred.groupby("date").size().reset_index(name="requêtes")
        st.plotly_chart(
            px.line(req_by_day, x="date", y="requêtes", title="Requêtes par jour"),
            use_container_width=True,
        )

    with g2:
        err_by_day = df_pred.groupby("date")["error"].sum().reset_index()
        st.plotly_chart(
            px.line(err_by_day, x="date", y="error", title="Erreurs par jour"),
            use_container_width=True,
        )

    g3, g4 = st.columns(2)

    with g3:
        churn_counts = (
            df_pred[df_pred["error"] == 0]["prediction"]
            .map({0: "Non-churn", 1: "Churn"})
            .value_counts()
            .reindex(["Non-churn", "Churn"], fill_value=0)
            .reset_index()
        )
        churn_counts.columns = ["classe", "count"]
        st.plotly_chart(
            px.pie(
                    churn_counts, names="classe", values="count",
                    title="Distribution churn / non-churn",
                    color="classe",
                    color_discrete_map={"Non-churn": "#2ecc71", "Churn": "#e74c3c"},
                ),
                use_container_width=True,
        )



    with g4:
        st.plotly_chart(
            px.histogram(df_pred, x="response_time", nbins=30, title="Distribution du temps de réponse"),
            use_container_width=True,
        )
else:
    st.info("Pas encore assez de prédictions enregistrées pour tracer les graphiques.")

st.divider()

# ---------- DONNÉES DE PRODUCTION ----------
st.subheader("Statistiques des données de production")
prod_cols = st.columns(3)
for i, col_name in enumerate(["tenure", "MonthlyCharges", "TotalCharges"]):
    stats = production.get(col_name, {})
    with prod_cols[i]:
        st.markdown(f"**{col_name}**")
        st.write(stats)

st.divider()

# ---------- CHAT AGENT ----------
st.subheader("💬 Poser une question à l'agent")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for role, content in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(content)

question = st.chat_input("Ex: y a-t-il du drift ? Faut-il réentraîner le modèle ?")
if question:
    st.session_state.chat_history.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("L'agent réfléchit..."):
            resp = requests.post(f"{BACKEND_URL}/api/ask", json={"question": question}, timeout=60)
            answer = resp.json().get("answer", "Erreur.")
        st.markdown(answer)
    st.session_state.chat_history.append(("assistant", answer))
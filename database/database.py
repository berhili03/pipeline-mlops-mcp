from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime


# ============================================================
# 1. Connexion SQLite
# ============================================================

DATABASE_URL = "sqlite:///./prediction_logs.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)


# ============================================================
# 2. Base SQLAlchemy
# ============================================================

Base = declarative_base()


# ============================================================
# 3. Table prediction_logs
# ============================================================

class PredictionLog(Base):

    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)

    timestamp = Column(
        DateTime,
        default=datetime.utcnow
    )

    # Données du client
    gender = Column(String)
    SeniorCitizen = Column(Integer)
    Partner = Column(String)
    Dependents = Column(String)
    tenure = Column(Integer)
    PhoneService = Column(String)
    MultipleLines = Column(String)
    InternetService = Column(String)
    OnlineSecurity = Column(String)
    OnlineBackup = Column(String)
    DeviceProtection = Column(String)
    TechSupport = Column(String)
    StreamingTV = Column(String)
    StreamingMovies = Column(String)
    Contract = Column(String)
    PaperlessBilling = Column(String)
    PaymentMethod = Column(String)

    MonthlyCharges = Column(Float)
    TotalCharges = Column(Float)

    # Résultat de la prédiction
    prediction = Column(Integer)

    # Version du modèle utilisée
    model_version = Column(String)

    # Temps de réponse en secondes
    response_time = Column(Float)

    # 0 = succès, 1 = erreur
    error = Column(Integer, default=0)


# ============================================================
# 4. Créer la table
# ============================================================

Base.metadata.create_all(bind=engine)


# ============================================================
# 5. Session SQLite
# ============================================================

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)
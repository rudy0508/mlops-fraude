"""serve/app.py — API REST de detección de fraude en tarjeta de crédito."""
import logging
import os
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s | API-FRAUDE | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

V_COLS = [f"V{i}" for i in range(1, 29)]
FEATURES = V_COLS + ["Amount_log", "Time_sin", "Time_cos"]
MODEL_PATH = Path("artifacts/modelo_fraude.pkl")
UMBRAL_FRAUDE = float(os.getenv("UMBRAL_FRAUDE", "0.50"))
modelo = None


class TransaccionInput(BaseModel):
    """
    Input del modelo: features PCA-transformadas + monto y hora.
    En producción, la capa upstream realiza la transformación PCA antes de llamar esta API.
    Amount y Time se transforman aquí dentro.
    """
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float = Field(..., ge=0, description="Monto de la transacción en USD")
    Time: float   = Field(..., ge=0, description="Segundos desde inicio del período de monitoreo")

    model_config = {"json_schema_extra": {"example": {
        "V1": -1.36, "V2": 0.47, "V3": 0.24, "V4": 1.38, "V5": -0.34,
        "V6": 0.46, "V7": 0.24, "V8": 0.10, "V9": 0.36, "V10": 0.09,
        "V11": -0.55, "V12": -0.62, "V13": -0.99, "V14": -0.31, "V15": 1.47,
        "V16": -0.47, "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
        "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07, "V25": 0.13,
        "V26": -0.19, "V27": 0.13, "V28": -0.02,
        "Amount": 149.62, "Time": 0
    }}}


class FraudResponse(BaseModel):
    probabilidad_fraude: float
    decision:            str     # FRAUDE | LEGÍTIMA
    nivel_riesgo:        str     # ALTO | MEDIO | BAJO
    score:               float
    umbral_usado:        float
    modelo:              str


class HealthResponse(BaseModel):
    status:  str
    modelo:  str
    version: str
    umbral:  float


def _transformar_features(solicitud: TransaccionInput) -> pd.DataFrame:
    """Aplica las mismas transformaciones que preprocess/run.py."""
    data = solicitud.model_dump()
    v_values = {f"V{i}": data[f"V{i}"] for i in range(1, 29)}

    amount_log = np.log1p(data["Amount"])

    segundos_dia = 24 * 3600
    t_norm = (data["Time"] % segundos_dia) / segundos_dia
    time_sin = np.sin(2 * np.pi * t_norm)
    time_cos = np.cos(2 * np.pi * t_norm)

    row = {**v_values, "Amount_log": amount_log, "Time_sin": time_sin, "Time_cos": time_cos}
    return pd.DataFrame([row])


def _nivel_riesgo(prob: float) -> str:
    if prob >= 0.70:
        return "ALTO"
    elif prob >= 0.30:
        return "MEDIO"
    return "BAJO"


def cargar_modelo():
    global modelo
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            modelo = pickle.load(f)
        log.info("Modelo cargado: %s (%s)", MODEL_PATH, type(modelo).__name__)
    else:
        raise FileNotFoundError("Ejecuta: python random_forest/run.py")


@asynccontextmanager
async def lifespan(app: FastAPI):
    cargar_modelo()
    yield


app = FastAPI(
    title="API Detección de Fraude — MLOps Demo",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/", tags=["Info"])
def root():
    return {"api": "Detección de Fraude", "version": "1.0.0", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["Salud"])
def health():
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    return HealthResponse(status="ok", modelo=type(modelo).__name__,
                          version="1.0.0", umbral=UMBRAL_FRAUDE)


@app.post("/predict", response_model=FraudResponse, tags=["Prediccion"])
def predict(solicitud: TransaccionInput):
    """
    Evalúa si una transacción es fraudulenta.
    decision: FRAUDE | LEGÍTIMA
    nivel_riesgo: ALTO (≥70%) | MEDIO (30-70%) | BAJO (<30%)
    """
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    try:
        df = _transformar_features(solicitud)
        feats = [f for f in FEATURES if f in df.columns]
        prob = float(modelo.predict_proba(df[feats])[0][1])
        return FraudResponse(
            probabilidad_fraude=round(prob, 4),
            decision="FRAUDE" if prob >= UMBRAL_FRAUDE else "LEGÍTIMA",
            nivel_riesgo=_nivel_riesgo(prob),
            score=round(prob, 4),
            umbral_usado=UMBRAL_FRAUDE,
            modelo=type(modelo).__name__,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

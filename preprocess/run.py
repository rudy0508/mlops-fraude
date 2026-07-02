"""
preprocess/run.py — Etapa 2: Limpieza y normalización del dataset de fraude.

Operaciones específicas para Credit Card Fraud:
  - V1-V28: ya están PCA-transformadas — solo imputar nulos si los hay
  - Amount:  normalización log1p + MinMaxScaler (distribución muy sesgada a la derecha)
  - Time:    normalización cíclica (sen/cos del ciclo de 24h) para capturar patrones horarios
  - Validación de schema post-procesamiento

Nota sobre Amount: se usa log1p porque la mayoría de transacciones son < 100 USD
pero hay outliers de > 20,000 USD. La transformación log reduce el skewness sin perder info.

Ejecutar: python preprocess/run.py
"""
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | PREPROCESS | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

V_COLS = [f"V{i}" for i in range(1, 29)]
TARGET = "Class"
ALL_COLS = V_COLS + ["Amount", "Time", "Amount_log", "Time_sin", "Time_cos", TARGET]


def imputar_nulos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in V_COLS + ["Amount", "Time"]:
        if col in df.columns:
            n = df[col].isnull().sum()
            if n > 0:
                df[col].fillna(df[col].median(), inplace=True)
                log.info("  Imputados %d nulos en '%s' (mediana)", n, col)
    return df


def transformar_amount(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica log1p a Amount para reducir el skewness extremo.
    Luego normaliza a [0,1] con MinMaxScaler.
    """
    df = df.copy()
    df["Amount_log"] = np.log1p(df["Amount"])
    scaler = MinMaxScaler()
    df["Amount_log"] = scaler.fit_transform(df[["Amount_log"]])
    log.info("  Amount: log1p + MinMaxScaler aplicado (rango original: %.2f - %.2f)",
             df["Amount"].min(), df["Amount"].max())
    return df


def transformar_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforma Time en dos features cíclicas (seno y coseno) del ciclo de 24h.
    Captura que la hora 23:59 y 00:01 son adyacentes (periodicidad circular).
    """
    df = df.copy()
    segundos_dia = 24 * 3600
    t_norm = (df["Time"] % segundos_dia) / segundos_dia
    df["Time_sin"] = np.sin(2 * np.pi * t_norm)
    df["Time_cos"] = np.cos(2 * np.pi * t_norm)
    log.info("  Time: codificación cíclica (Time_sin, Time_cos) para periodicidad diaria")
    return df


def normalizar_v_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Las V1-V28 ya están PCA-transformadas. Verificamos que estén en rango razonable
    y aplicamos una normalización suave si hay outliers extremos.
    """
    df = df.copy()
    for col in V_COLS:
        if col in df.columns:
            # El dataset real tiene V1-V28 en rango aproximado [-30, 30]
            n_outliers = ((df[col] < -50) | (df[col] > 50)).sum()
            if n_outliers > 0:
                df[col] = df[col].clip(-50, 50)
                log.info("  Clipeados %d outliers extremos en '%s'", n_outliers, col)
    return df


def validar_schema(df: pd.DataFrame) -> None:
    errores = []
    nulos = df.isnull().sum().sum()
    if nulos > 0:
        errores.append(f"NULOS RESIDUALES: {nulos}")

    for col in ["Amount_log", "Time_sin", "Time_cos", TARGET]:
        if col not in df.columns:
            errores.append(f"COLUMNA FALTANTE: {col}")

    if df[TARGET].nunique() != 2:
        errores.append(f"TARGET inválido: valores únicos = {df[TARGET].unique()}")

    if errores:
        raise ValueError("VALIDACIÓN POST-PREPROCESS FALLIDA:\n" + "\n".join(errores))

    fraude_pct = df[TARGET].mean() * 100
    log.info("  Schema OK — nulos: 0 | Tasa fraude: %.4f%%", fraude_pct)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/fraud_raw.csv")
    parser.add_argument("--output", default="data/fraud_clean.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not Path(args.input).exists():
        raise FileNotFoundError(f"{args.input} no encontrado. Ejecuta: python data/run.py")

    log.info("Cargando: %s", args.input)
    df = pd.read_csv(args.input)
    log.info("Shape inicial: %d x %d | Nulos: %d", df.shape[0], df.shape[1], df.isnull().sum().sum())

    df = imputar_nulos(df)
    df = transformar_amount(df)
    df = transformar_time(df)
    df = normalizar_v_features(df)
    validar_schema(df)

    # Descartar columnas originales que ya fueron transformadas
    df.drop(columns=["Amount", "Time"], inplace=True)

    df.to_csv(args.output, index=False)
    log.info("Dataset limpio guardado: %s (%d filas, %d columnas)", args.output, df.shape[0], df.shape[1])
    log.info("Features finales: V1-V28 + Amount_log + Time_sin + Time_cos + Class")

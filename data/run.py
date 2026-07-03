"""
data/run.py — Etapa 1: Descarga del dataset Credit Card Fraud.

Fuente primaria : sklearn.datasets.fetch_openml(name='creditcard', version=1)
                  Espejo del dataset Kaggle — sin API key ni credenciales requeridas.
                  284,807 transacciones reales | 492 fraudes (0.17%)

Fuente fallback : datos sintéticos con estructura idéntica (flag --sintetico)
                  Útil para CI/CD, tests y desarrollo rápido.

Features:
  V1-V28  — variables PCA-transformadas (features originales confidenciales)
  Amount  — monto de la transacción en USD
  Time    — segundos desde la primera transacción del dataset
  Class   — target: 0=legítima, 1=fraude

Ejecutar:
  python data/run.py                   # descarga real desde OpenML
  python data/run.py --sintetico       # datos sintéticos (rápido)
"""
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | DOWNLOAD | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

OUTPUT = "data/fraud_raw.csv"
V_COLS = [f"V{i}" for i in range(1, 29)]
ALL_COLS = V_COLS + ["Amount", "Time", "Class"]


def descargar_openml() -> pd.DataFrame:
    from sklearn.datasets import fetch_openml
    log.info("Descargando desde OpenML (name='creditcard') — puede tardar 1-3 min la primera vez...")
    log.info("Los datos se cachean localmente en ~/scikit_learn_data/ para próximas ejecuciones.")
    data = fetch_openml(name="creditcard", version=1, as_frame=True, parser="auto")
    df = data.frame.copy()

    # OpenML a veces entrega el target como string '0'/'1'
    df["Class"] = df["Class"].astype(int)

    # Asegurar nombres de columna consistentes
    df.columns = [c.strip() for c in df.columns]

    log.info("Dataset descargado: %d filas x %d columnas", df.shape[0], df.shape[1])
    log.info("Tasa de fraude: %.4f%% (%d fraudes de %d transacciones)",
             df["Class"].mean() * 100, df["Class"].sum(), len(df))
    return df


def generar_sintetico(n: int = 10_000) -> pd.DataFrame:
    """
    Genera datos sintéticos con la misma estructura que el dataset real.
    Desbalanceo similar: ~0.17% fraudes.
    Útil para desarrollo, CI/CD y tests sin descarga.
    """
    from sklearn.datasets import make_classification
    log.warning("Modo SINTÉTICO — solo para desarrollo/tests, NO apto para producción.")

    rng = np.random.default_rng(42)

    # En dev/test usamos 5% de fraudes (no 0.17%) para que el modelo tenga
    # suficientes ejemplos positivos. El dataset real tiene la distribución correcta.
    X, y = make_classification(
        n_samples=n,
        n_features=28,
        n_informative=15,
        n_redundant=5,
        n_clusters_per_class=1,
        weights=[0.95, 0.05],
        flip_y=0.005,
        random_state=42,
    )

    df = pd.DataFrame(X, columns=V_COLS)

    # Amount: distribución exponencial similar al dataset real (media ~88 USD)
    df["Amount"] = rng.exponential(scale=88, size=n).clip(0.01, 25_000).round(2)

    # Time: segundos en ventana de 2 días (172,800 s)
    df["Time"] = np.sort(rng.uniform(0, 172_800, size=n)).round(0)

    df["Class"] = y

    fraudes = df["Class"].sum()
    log.info("Sintético generado: %d filas | Fraudes: %d (%.4f%%)", n, fraudes, fraudes / n * 100)
    return df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--sintetico", action="store_true",
                        help="Genera datos sintéticos en lugar de descargar de OpenML")
    parser.add_argument("--n-sintetico", type=int, default=10_000,
                        help="Número de filas para datos sintéticos (default: 10000)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    Path("data").mkdir(exist_ok=True)

    if args.sintetico:
        df = generar_sintetico(n=args.n_sintetico)
    else:
        try:
            df = descargar_openml()
        except Exception as e:
            log.warning("OpenML no disponible (%s). Usando datos sintéticos como fallback.", e)
            df = generar_sintetico()

    df.to_csv(args.output, index=False)
    log.info("Dataset guardado: %s (%d filas)", args.output, len(df))
    log.info("Siguiente etapa: python preprocess/run.py")

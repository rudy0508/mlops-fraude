"""
segregate/run.py — Etapa 3: Train/Test split estratificado para dataset de fraude.

El stratify en 'Class' es crítico aquí: con solo 0.17% de fraudes,
un split aleatorio sin stratify podría dejar el test set sin suficientes
fraudes para una evaluación estadísticamente válida.

Ejecutar: python segregate/run.py
"""
import argparse
import logging
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s | SEGREGATE | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

TARGET = "Class"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",     default="data/fraud_clean.csv")
    parser.add_argument("--train_out", default="data/fraud_train.csv")
    parser.add_argument("--test_out",  default="data/fraud_test.csv")
    parser.add_argument("--test_size", type=float, default=0.20)
    parser.add_argument("--seed",      type=int,   default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not Path(args.input).exists():
        raise FileNotFoundError(f"{args.input} no encontrado. Ejecuta: python preprocess/run.py")

    df = pd.read_csv(args.input)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    log.info("Total: %d filas | Fraudes: %d (%.4f%%)", len(df), y.sum(), y.mean() * 100)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,   # CRÍTICO: mantiene la proporción de fraudes en ambos splits
    )

    df_train = pd.concat([X_train, y_train], axis=1)
    df_test  = pd.concat([X_test,  y_test],  axis=1)

    df_train.to_csv(args.train_out, index=False)
    df_test.to_csv(args.test_out,   index=False)

    log.info("Train: %d filas | Fraudes: %d (%.4f%%)",
             len(df_train), df_train[TARGET].sum(), df_train[TARGET].mean() * 100)
    log.info("Test : %d filas | Fraudes: %d (%.4f%%)",
             len(df_test),  df_test[TARGET].sum(),  df_test[TARGET].mean() * 100)
    log.info("Guardado: %s y %s", args.train_out, args.test_out)

"""
random_forest/run.py — Etapa 5: Entrenamiento con GridSearchCV + MLflow.

Adaptaciones clave para detección de fraude vs. caso de crédito base:
  - class_weight='balanced': compensa el desbalanceo extremo (0.17% fraudes)
  - scoring='average_precision': PR-AUC como métrica de optimización
    (ROC-AUC es engañosamente alto en datasets desbalanceados)
  - Se registran tanto PR-AUC como ROC-AUC en MLflow para comparabilidad

Ejecutar: python random_forest/run.py
"""
import json
import logging
import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score, recall_score, precision_score
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | TRAIN | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

TARGET = "Class"
V_COLS = [f"V{i}" for i in range(1, 29)]
FEATURES = V_COLS + ["Amount_log", "Time_sin", "Time_cos"]
ARTIFACTS = Path("artifacts")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data",        default="data/fraud_train.csv")
    parser.add_argument("--experiment_name",   default="fraud_detection")
    parser.add_argument("--mlflow_uri",        default=None)
    parser.add_argument("--model_name",        default="FraudDetectionModel")
    parser.add_argument("--n_estimators",      default="100,200,300")
    parser.add_argument("--max_depth",         default="5,8,10")
    parser.add_argument("--min_samples_split", default="2,5,10")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ARTIFACTS.mkdir(exist_ok=True)

    if not Path(args.train_data).exists():
        raise FileNotFoundError(f"{args.train_data} no encontrado. Ejecuta los pasos anteriores.")

    log.info("Cargando datos: %s", args.train_data)
    df = pd.read_csv(args.train_data)
    X = df[FEATURES]
    y = df[TARGET]

    log.info("Train: %d filas | Fraudes: %d (%.4f%%)", len(df), y.sum(), y.mean() * 100)

    mlflow.set_tracking_uri(args.mlflow_uri or os.environ.get("MLFLOW_TRACKING_URI", "file:///app/mlruns"))
    mlflow.set_experiment(args.experiment_name)

    param_grid = {
        "n_estimators":      [int(x) for x in args.n_estimators.split(",")],
        "max_depth":         [int(x) for x in args.max_depth.split(",")],
        "min_samples_split": [int(x) for x in args.min_samples_split.split(",")],
    }
    n_comb = len(param_grid["n_estimators"]) * len(param_grid["max_depth"]) * len(param_grid["min_samples_split"])
    log.info("GridSearchCV: %d combinaciones x 5 folds (scoring: average_precision)", n_comb)

    gs = GridSearchCV(
        RandomForestClassifier(
            random_state=42,
            class_weight="balanced",   # crítico para desbalanceo extremo
        ),
        param_grid,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring="average_precision",   # PR-AUC: métrica correcta para fraude
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    gs.fit(X, y)

    log.info("Mejores parámetros: %s", gs.best_params_)
    log.info("Mejor PR-AUC CV:    %.4f", gs.best_score_)

    with mlflow.start_run(run_name="random_forest_fraud") as run:
        mlflow.log_params(gs.best_params_)
        mlflow.log_params({
            "n_combinaciones":  n_comb,
            "cv_folds":         5,
            "scoring":          "average_precision",
            "class_weight":     "balanced",
            "desbalanceo_pct":  round(y.mean() * 100, 4),
        })

        y_pred = gs.predict(X)
        y_proba = gs.predict_proba(X)[:, 1]
        mlflow.log_metrics({
            "cv_prauc_mean":    round(gs.best_score_, 4),
            "train_prauc":      round(average_precision_score(y, y_proba), 4),
            "train_rocauc":     round(roc_auc_score(y, y_proba), 4),
            "train_f1":         round(f1_score(y, y_pred), 4),
            "train_recall":     round(recall_score(y, y_pred), 4),
            "train_precision":  round(precision_score(y, y_pred, zero_division=0), 4),
        })

        mlflow.sklearn.log_model(
            gs.best_estimator_,
            artifact_path="fraud_model",
            registered_model_name=args.model_name,
        )
        run_id = run.info.run_id
        log.info("MLflow Run ID: %s", run_id)

    with open(ARTIFACTS / "modelo_fraude.pkl", "wb") as f:
        pickle.dump(gs.best_estimator_, f)

    with open(ARTIFACTS / "train_run_id.txt", "w") as f:
        f.write(run_id)

    log.info("Modelo guardado: artifacts/modelo_fraude.pkl")

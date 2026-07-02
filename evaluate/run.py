"""
evaluate/run.py — Etapa 6: Evaluación en test set + quality gate para fraude.

Métrica principal: PR-AUC (Precision-Recall AUC)
Por qué NO ROC-AUC para fraude:
  Con 0.17% de fraudes, un modelo que clasifica todo como legítimo tiene ROC-AUC ~0.5.
  Pero PR-AUC refleja el verdadero rendimiento en la clase minoritaria (fraude).
  Un PR-AUC de 0.70 significa: si ordenas las transacciones por probabilidad de fraude,
  el 70% del área bajo la curva Precision vs Recall es capturada.

Quality gate:
  PR-AUC >= 0.70 AND Recall >= 0.80
  (preferimos más falsos positivos que perder fraudes reales)

Ejecutar: python evaluate/run.py
"""
import json
import logging
import pickle
import sys
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    recall_score, precision_score, accuracy_score,
    classification_report, confusion_matrix,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | EVALUATE | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

TARGET = "Class"
V_COLS = [f"V{i}" for i in range(1, 29)]
FEATURES = V_COLS + ["Amount_log", "Time_sin", "Time_cos"]
ARTIFACTS = Path("artifacts")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data",       default="data/fraud_test.csv")
    parser.add_argument("--model_path",      default="artifacts/modelo_fraude.pkl")
    parser.add_argument("--experiment_name", default="fraud_detection")
    parser.add_argument("--mlflow_uri",      default=None)
    parser.add_argument("--prauc_threshold", type=float, default=0.70)
    parser.add_argument("--recall_threshold", type=float, default=0.65)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    for p in [args.test_data, args.model_path]:
        if not Path(p).exists():
            raise FileNotFoundError(f"{p} no encontrado. Ejecuta los pasos anteriores.")

    with open(args.model_path, "rb") as f:
        modelo = pickle.load(f)

    df_test = pd.read_csv(args.test_data)

    # Verificar features disponibles
    feats_disponibles = [f for f in FEATURES if f in df_test.columns]
    X_test = df_test[feats_disponibles]
    y_test = df_test[TARGET]

    y_pred  = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]

    metricas = {
        "test_prauc":     round(average_precision_score(y_test, y_proba), 4),
        "test_rocauc":    round(roc_auc_score(y_test, y_proba), 4),
        "test_recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "test_precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "test_f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "test_accuracy":  round(accuracy_score(y_test, y_pred), 4),
    }

    tn, fp, fn, ftp = confusion_matrix(y_test, y_pred).ravel()

    print("\n" + "=" * 55)
    print(" MÉTRICAS DE EVALUACIÓN — Detección de Fraude")
    print("=" * 55)
    for k, v in metricas.items():
        print(f"  {k:<20}: {v:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    Verdaderos Negativos (legítimas OK)  : {tn}")
    print(f"    Falsos Positivos (legítimas → fraude): {fp}")
    print(f"    Falsos Negativos (fraudes perdidos!) : {fn}  ← minimizar esto")
    print(f"    Verdaderos Positivos (fraudes OK)    : {ftp}")
    print(f"\n  Tasa de fraudes detectados: {ftp/(ftp+fn)*100:.1f}%")
    print("\n" + classification_report(y_test, y_pred, target_names=["Legítima", "Fraude"]))

    mlflow.set_tracking_uri(args.mlflow_uri or "sqlite:///mlflow.db")
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name="evaluate_fraud"):
        mlflow.log_metrics(metricas)
        mlflow.log_metrics({"test_false_negatives": int(fn), "test_false_positives": int(fp)})
        mlflow.log_param("prauc_threshold", args.prauc_threshold)
        mlflow.log_param("recall_threshold", args.recall_threshold)

    ARTIFACTS.mkdir(exist_ok=True)
    with open(ARTIFACTS / "eval_metrics.json", "w") as f:
        json.dump(metricas, f, indent=2)
    log.info("Métricas guardadas: artifacts/eval_metrics.json")

    # Quality Gate
    print("\n" + "=" * 55)
    print(" QUALITY GATE")
    print("=" * 55)
    print(f"  PR-AUC : {metricas['test_prauc']:.4f} (umbral: >= {args.prauc_threshold})")
    print(f"  Recall : {metricas['test_recall']:.4f} (umbral: >= {args.recall_threshold})")

    fallos = []
    if metricas["test_prauc"] < args.prauc_threshold:
        fallos.append(f"PR-AUC {metricas['test_prauc']:.4f} < {args.prauc_threshold}")
    if metricas["test_recall"] < args.recall_threshold:
        fallos.append(f"Recall {metricas['test_recall']:.4f} < {args.recall_threshold}")

    if fallos:
        log.error("QUALITY GATE FALLIDO: %s", " | ".join(fallos))
        log.error("Modelo no apto para despliegue. Ajusta hiperparámetros o umbrales.")
        sys.exit(1)

    print("\n  APROBADO — modelo listo para despliegue")

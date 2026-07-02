"""
drift/run.py — Etapa 7: Detección de drift — Fraude.

En fraude, el drift es adversarial: los defraudadores adaptan su comportamiento
constantemente para evadir los modelos. Es crítico monitorear cambios en:
  - Distribución de montos (Amount)
  - Patrones horarios (Time)
  - Features V1-V28 (que reflejan patrones de comportamiento)

Ejecutar: python drift/run.py
"""
import json
import logging
import sys
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | DRIFT | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

V_COLS = [f"V{i}" for i in range(1, 29)]
FEATURES = V_COLS + ["Amount_log", "Time_sin", "Time_cos"]
REPORTS = Path("reportes")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference",       default="data/fraud_train.csv")
    parser.add_argument("--current",         default="data/fraud_test.csv")
    parser.add_argument("--drift_threshold", type=float, default=0.30)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    REPORTS.mkdir(exist_ok=True)

    for p in [args.reference, args.current]:
        if not Path(p).exists():
            raise FileNotFoundError(f"{p} no encontrado. Ejecuta los pasos anteriores.")

    feats_ref = [f for f in FEATURES if f in pd.read_csv(args.reference, nrows=1).columns]
    df_ref  = pd.read_csv(args.reference)[feats_ref]
    df_prod = pd.read_csv(args.current)[feats_ref]

    log.info("Referencia (train): %d filas | Actual (test): %d filas", len(df_ref), len(df_prod))

    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=df_ref, current_data=df_prod)
        report.save_html(str(REPORTS / "drift_report.html"))
        resultado   = report.as_dict()
        drift_info  = resultado["metrics"][0]["result"]
        drift_det   = drift_info["dataset_drift"]
        drift_share = drift_info["share_of_drifted_columns"]
        drift_n     = drift_info["number_of_drifted_columns"]
        drift_total = drift_info["number_of_columns"]
        log.info("Reporte Evidently guardado: reportes/drift_report.html")
    except ImportError:
        from scipy.stats import ks_2samp
        drifted     = sum(1 for c in feats_ref if ks_2samp(df_ref[c], df_prod[c]).pvalue < 0.05)
        drift_share = drifted / len(feats_ref)
        drift_det   = drift_share > args.drift_threshold
        drift_n, drift_total = drifted, len(feats_ref)
        log.info("Usando KS test (Evidently no instalado)")

    resumen = {
        "drift_detectado":     drift_det,
        "features_con_drift":  drift_n,
        "total_features":      drift_total,
        "share_drifted":       round(drift_share, 4),
        "umbral":              args.drift_threshold,
        "nota":                "En fraude, drift adversarial es esperable — re-entrenar frecuentemente",
    }
    with open(REPORTS / "drift_summary.json", "w") as f:
        json.dump(resumen, f, indent=2)

    print(f"\n  Drift detectado    : {drift_det}")
    print(f"  Features con drift : {drift_n}/{drift_total} ({drift_share*100:.0f}%)")

    if drift_share > args.drift_threshold:
        log.warning("ALERTA DRIFT ADVERSARIAL: %.0f%% features con drift — re-entrenamiento urgente",
                    drift_share * 100)
    else:
        print("  Drift dentro de límites aceptables")

"""
main.py — Orquestador del pipeline MLOps: Detección de Fraude en Tarjeta de Crédito.

Dataset: Credit Card Fraud Detection (OpenML / sklearn.fetch_openml)
         284,807 transacciones reales | 492 fraudes (0.17%)
         V1-V28: features PCA-transformadas (confidenciales en el dataset original)
         Amount, Time: features en escala original

Uso:
    python main.py                              # todas las etapas
    python main.py --steps download preprocess  # etapas específicas
    python main.py --sintetico                  # datos sintéticos (dev rápido, sin descarga)
"""
import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | PIPELINE-FRAUDE | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline_run.log"),
    ],
)
log = logging.getLogger(__name__)

ALL_STEPS = ["download", "preprocess", "segregate", "check_data", "random_forest", "evaluate", "drift"]


def build_commands(sintetico: bool) -> dict:
    download_cmd = [sys.executable, "data/run.py"]
    if sintetico:
        download_cmd.append("--sintetico")
    return {
        "download":      download_cmd,
        "preprocess":    [sys.executable, "preprocess/run.py"],
        "segregate":     [sys.executable, "segregate/run.py"],
        "check_data":    [sys.executable, "-m", "pytest", "check_data/test_data.py", "-v", "--tb=short"],
        "random_forest": [sys.executable, "random_forest/run.py"],
        "evaluate":      [sys.executable, "evaluate/run.py"],
        "drift":         [sys.executable, "drift/run.py"],
    }


def ejecutar_paso(nombre: str, cmd: list) -> tuple[bool, float]:
    inicio = time.time()
    log.info(">>> Iniciando: %s", nombre)
    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"
    result = subprocess.run(cmd, capture_output=False, env=env)
    dur = round(time.time() - inicio, 2)
    ok = result.returncode == 0
    if ok:
        log.info("<<< Completado: %s (%.2f s)", nombre, dur)
    else:
        log.error("XXX FALLO: %s (código: %d)", nombre, result.returncode)
    return ok, dur


def main():
    parser = argparse.ArgumentParser(description="Pipeline MLOps — Detección de Fraude")
    parser.add_argument("--steps", nargs="+", default=ALL_STEPS, choices=ALL_STEPS)
    parser.add_argument("--sintetico", action="store_true",
                        help="Usa datos sintéticos en lugar de OpenML (desarrollo rápido)")
    args = parser.parse_args()

    for d in ["data", "artifacts", "reportes"]:
        Path(d).mkdir(exist_ok=True)

    if args.sintetico:
        log.warning("Modo SINTÉTICO activado — solo para desarrollo/tests, no para producción")

    step_commands = build_commands(args.sintetico)

    log.info("=" * 60)
    log.info(" PIPELINE MLOps — Detección de Fraude en Tarjeta de Crédito")
    log.info(" Etapas: %s", " → ".join(args.steps))
    log.info("=" * 60)

    resumen = []
    for paso in args.steps:
        ok, dur = ejecutar_paso(paso, step_commands[paso])
        resumen.append({"paso": paso, "estado": "OK" if ok else "FALLO", "duracion_s": dur})
        if not ok:
            log.error("Pipeline detenido en: %s", paso)
            sys.exit(1)

    dur_total = sum(r["duracion_s"] for r in resumen)
    log.info("=" * 60)
    log.info(" PIPELINE COMPLETADO EN %.2f segundos", dur_total)
    log.info("=" * 60)
    for r in resumen:
        log.info("  [%s] %s (%.2f s)", r["estado"], r["paso"], r["duracion_s"])
    log.info("")
    log.info("  Siguiente: uvicorn serve.app:app --host 0.0.0.0 --port 8001")


if __name__ == "__main__":
    main()

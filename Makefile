.PHONY: all install lint pipeline pipeline-sintetico serve docker smoke clean help

install:
	pip install -r requirements.txt

all: install lint pipeline serve

lint:
	flake8 . --max-line-length=100 --exclude=.git,__pycache__,mlruns,artifacts,data,reportes

# Pipeline completo con datos reales (OpenML, ~2min descarga primera vez)
pipeline:
	python main.py

# Pipeline con datos sintéticos — ideal para CI/CD y desarrollo rápido
pipeline-sintetico:
	python main.py --sintetico

# Etapas individuales
etapa1: ; python data/run.py
etapa1-sintetico: ; python data/run.py --sintetico
etapa2: ; python preprocess/run.py
etapa3: ; python segregate/run.py
etapa4: ; pytest check_data/test_data.py -v --tb=short
etapa5: ; python random_forest/run.py
etapa6: ; python evaluate/run.py
etapa7: ; python drift/run.py

serve:
	uvicorn serve.app:app --host 0.0.0.0 --port 8001 --reload

docker-build:
	docker build -t fraud-detection-api:local -f serve/Dockerfile .

docker-run:
	docker run -p 8001:8001 --name fraud-api fraud-detection-api:local

docker-stop:
	docker stop fraud-api && docker rm fraud-api

smoke:
	curl -sf http://localhost:8001/health | python3 -m json.tool
	@echo ""
	curl -X POST http://localhost:8001/predict \
	  -H 'Content-Type: application/json' \
	  -d '{"V1":-1.36,"V2":0.47,"V3":0.24,"V4":1.38,"V5":-0.34,"V6":0.46,"V7":0.24,"V8":0.10,"V9":0.36,"V10":0.09,"V11":-0.55,"V12":-0.62,"V13":-0.99,"V14":-0.31,"V15":1.47,"V16":-0.47,"V17":0.21,"V18":0.03,"V19":0.40,"V20":0.25,"V21":-0.02,"V22":0.28,"V23":-0.11,"V24":0.07,"V25":0.13,"V26":-0.19,"V27":0.13,"V28":-0.02,"Amount":149.62,"Time":0}' \
	  | python3 -m json.tool

mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5000

clean:
	rm -rf data/ artifacts/ reportes/ mlruns/ __pycache__ pipeline_run.log mlflow.db
	find . -name "*.pyc" -delete
	@echo "Limpieza completada"

help:
	@echo ""
	@echo "=== Pipeline MLOps — Detección de Fraude ==="
	@echo "  make install           — instalar dependencias"
	@echo "  make pipeline          — ejecutar con datos reales (OpenML)"
	@echo "  make pipeline-sintetico — ejecutar con datos sintéticos (rápido)"
	@echo "  make etapa1-7          — ejecutar una etapa específica"
	@echo "  make serve             — levantar API FastAPI en puerto 8001"
	@echo "  make docker-build      — construir imagen Docker"
	@echo "  make smoke             — test rápido de endpoints"
	@echo "  make mlflow-ui         — UI de MLflow en puerto 5000"
	@echo "  make clean             — limpiar artefactos"
	@echo ""

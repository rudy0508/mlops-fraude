FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serve/ serve/

# artifacts/ NO se copia en build — se monta como volumen en docker-compose.yml
# Así el modelo entrenado (pkl) está disponible sin reconstruir la imagen
RUN mkdir -p artifacts

ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db
ENV UMBRAL_FRAUDE=0.50

EXPOSE 8001
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8001"]

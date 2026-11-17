FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    EEG_CHECKPOINT_PATH=/app/model/final_fusion_checkpoint.pt

WORKDIR /app

COPY api/requirements-api.txt /app/api/requirements-api.txt
RUN python -m pip install --no-cache-dir -r /app/api/requirements-api.txt

COPY api/__init__.py /app/api/__init__.py
COPY api/app/__init__.py api/app/main.py api/app/schemas.py api/app/service.py api/app/settings.py /app/api/app/

COPY src/preprocessing.py src/epoching.py src/features.py src/deep_fusion.py src/deep_models.py src/inference.py src/explainability.py src/ui_service.py /app/src/
COPY validate_dataset.py /app/validate_dataset.py
COPY results/deep_learning/cell_01/final_fusion/run_20261116T092136Z/final_fusion_checkpoint.pt /app/model/final_fusion_checkpoint.pt

RUN useradd --create-home --uid 10001 appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os; from urllib.request import urlopen; urlopen('http://127.0.0.1:' + os.getenv('PORT', '8000') + '/ready', timeout=3).read()"

CMD ["sh", "-c", "uvicorn api.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

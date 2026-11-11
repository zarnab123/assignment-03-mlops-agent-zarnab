FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV APP_VERSION=1.0.0

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY api ./api
COPY training/export_model.py ./training/export_model.py

RUN python training/export_model.py

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

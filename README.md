# MLOps AI Agent

A production-style MLOps project that combines DVC, MLflow, Apache Airflow, FastAPI, Docker, Kubernetes, GitHub Actions, and a safe multi-tool MLOps agent.

The project trains a breast-cancer classifier and lets a user inspect datasets, ML experiments, Airflow pipeline status, and Kubernetes deployment health using natural-language requests.

## Project goals

- Version and inspect datasets with DVC
- Track experiments, metrics, artifacts, and models with MLflow
- Orchestrate training with Apache Airflow
- Serve predictions through FastAPI
- Package the service in Docker
- Deploy two API replicas to Kubernetes
- Route user requests to real MLOps tools
- Require confirmation before state-changing actions
- Test and build automatically with GitHub Actions

## Architecture

```mermaid
flowchart TD
    DVC[DVC data versioning] --> AIRFLOW[Airflow training pipeline]
    AIRFLOW --> MLFLOW[MLflow tracking and registry]
    MLFLOW --> MODEL[Trained model]
    MODEL --> API[FastAPI inference API]
    API --> DOCKER[Docker image]
    DOCKER --> K8S[Kubernetes deployment]
    AGENT[MLOps agent] --> DVC
    AGENT --> MLFLOW
    AGENT --> AIRFLOW
    AGENT --> K8S
    CI[GitHub Actions CI] --> TESTS[Pytest]
    CI --> DOCKER

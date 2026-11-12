# MLOps AI Agent

A production-style MLOps project that combines DVC, MLflow, Apache Airflow, FastAPI, Docker, Kubernetes, GitHub Actions, and a safe multi-tool AI agent.

The project trains a breast-cancer classifier and lets an agent inspect datasets, ML experiments, Airflow pipeline status, and Kubernetes deployment health through natural-language requests.

## Project goals

- Version and inspect datasets with DVC
- Track and compare ML experiments with MLflow
- Orchestrate model training with Apache Airflow
- Serve predictions through a FastAPI inference API
- Package the API in Docker
- Deploy two API replicas to Kubernetes
- Route natural-language requests to real MLOps tools
- Require explicit confirmation before state-changing operations
- Automatically test and build the project with GitHub Actions

## Architecture

```mermaid
flowchart TD
    DVC[DVC dataset versioning] --> AIRFLOW[Airflow training pipeline]
    AIRFLOW --> MLFLOW[MLflow tracking and registry]
    MLFLOW --> MODEL[Trained model]

    MODEL --> API[FastAPI inference API]
    API --> DOCKER[Docker image]
    DOCKER --> K8S[Kubernetes deployment]

    AGENT[Multi-tool MLOps agent] --> DVC
    AGENT --> MLFLOW
    AGENT --> AIRFLOW
    AGENT --> K8S

    CI[GitHub Actions CI] --> TESTS[Pytest]
    CI --> DOCKER

# Technical Report: Production-Ready MLOps AI Agent

**Student:** Zarnab Javed  
**Repository:** https://github.com/zarnab123/assignment-03-mlops-agent-zarnab  
**Course:** MLOps and Production AI Engineering  
**Assignment:** Building a Production-Ready MLOps AI Agent

## 1. Project objective

This project implements a working MLOps agent for a breast-cancer classification system. The agent accepts natural-language requests and uses real tools to inspect:

- DVC dataset versions and provenance
- MLflow experiments, metrics, artifacts, and registered models
- Apache Airflow training pipeline status
- Kubernetes deployment health, Services, endpoints, and Pods

The project also contains a FastAPI prediction service, a Docker image, a Kubernetes deployment with two replicas, automated tests, and GitHub Actions CI.

## 2. System architecture

```mermaid
flowchart TD
    USER[User request] --> AGENT[MLOps agent orchestrator]

    AGENT --> DVC[DVC tools]
    AGENT --> MLFLOW[MLflow tools]
    AGENT --> AIRFLOW[Airflow tools]
    AGENT --> K8S[Kubernetes tools]

    DVC --> DATA[Versioned dataset]
    AIRFLOW --> TRAIN[Training pipeline]
    TRAIN --> MLFLOW
    MLFLOW --> MODEL[Registered model]
    MODEL --> API[FastAPI inference service]
    API --> DOCKER[Docker image]
    DOCKER --> K8S

    CI[GitHub Actions] --> TESTS[Pytest]
    CI --> BUILD[Docker build]

# Customer Churn MLOps Pipeline


## Overview

Production-style machine learning pipeline with:

- Model training
- Experiment tracking
- Model versioning
- Monitoring


## Architecture


Data

↓

Training Pipeline

↓

ML Model

↓

Deployment

↓

Monitoring


## Tools

Python

XGBoost

MLflow

Git


## Run Project


Install:

pip install -r requirements.txt


Train model:

python src/train.py


Start MLflow:

mlflow ui


## Business Goal

Predict customers who may churn and continuously monitor model performance.

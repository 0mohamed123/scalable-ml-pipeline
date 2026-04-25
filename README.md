# Scalable ML Pipeline

![Language](https://img.shields.io/badge/Language-Python-blue)
![Tests](https://img.shields.io/badge/Tests-10%20passing-green)
![Accuracy](https://img.shields.io/badge/Accuracy-96.67%25-brightgreen)

End-to-end ML pipeline: data ingestion, preprocessing, multi-model training,
evaluation, and REST-ready serving with latency tracking.

## Pipeline Stages

    [1] Preprocessing  ->  [2] Training  ->  [3] Evaluation  ->  [4] Serving

## Demo Output

    [1] Preprocessing...
        Samples: 150, Features: 4, Missing: 0
        Train: 105, Val: 15, Test: 30

    [2] Training...
        logistic_regression   | acc=0.8667 | f1=0.8705 | time=0.16s
        random_forest         | acc=0.8667 | f1=0.8705 | time=0.12s
        gradient_boosting     | acc=0.8667 | f1=0.8705 | time=0.11s
        Best model: logistic_regression (acc=0.8667)

    [3] Evaluating...
        Test Accuracy: 0.9667
        Test F1-Score: 0.9664

    [4] Serving...
        Status: healthy | Model: LogisticRegression
        Predictions: [0, 0, 0] | Latency: 0.0ms

## Quick Start

    git clone https://github.com/0mohamed123/scalable-ml-pipeline.git
    cd scalable-ml-pipeline
    pip install scikit-learn numpy

    cd src
    python pipeline.py

    cd ../tests
    python -m pytest test_pipeline.py -v

## Usage

    from pipeline import MLPipeline
    from sklearn.datasets import load_iris

    data = load_iris()
    pipeline = MLPipeline()
    metrics = pipeline.run(data.data, data.target)

## Components

| Component | Description |
|-----------|-------------|
| DataPreprocessor | StandardScaler + LabelEncoder + train/val/test split |
| ModelTrainer | Trains LogisticRegression, RandomForest, GradientBoosting |
| ModelServer | Serves predictions with latency tracking + health check |
| MLPipeline | Orchestrates all stages end-to-end |

## Test Results

    10 passed | 0 failed

    Tests cover: fit/transform, scaler mean, not fitted error,
    split sizes, stats, single model training, all models,
    evaluation metrics, server predict, health check

## Technologies

- Python 3.12
- scikit-learn
- NumPy
- pytest (10 tests)
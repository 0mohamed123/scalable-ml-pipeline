import sys
sys.path.append('../src')

import numpy as np
import pytest
from sklearn.datasets import load_iris
from preprocessing import DataPreprocessor
from training import ModelTrainer
from serving import ModelServer


@pytest.fixture
def iris():
    data = load_iris()
    return data.data, data.target


@pytest.fixture
def preprocessor(iris):
    X, y = iris
    p = DataPreprocessor()
    p.fit(X, y)
    return p


# ===== Preprocessor Tests =====
def test_fit_transform(iris):
    X, y = iris
    p = DataPreprocessor()
    X_scaled, y_enc = p.fit_transform(X, y)
    assert X_scaled.shape == X.shape
    assert len(y_enc) == len(y)


def test_scaler_mean(iris):
    X, y = iris
    p = DataPreprocessor()
    X_scaled = p.fit_transform(X)
    assert abs(X_scaled.mean()) < 0.1


def test_not_fitted_raises():
    p = DataPreprocessor()
    with pytest.raises(RuntimeError):
        p.transform([[1, 2, 3, 4]])


def test_split(iris):
    X, y = iris
    p = DataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = p.split(X, y)
    assert len(X_train) + len(X_val) + len(X_test) == len(X)
    assert len(X_test) == 30


def test_get_stats(iris):
    X, y = iris
    p = DataPreprocessor()
    stats = p.get_stats(X)
    assert stats['n_samples'] == 150
    assert stats['n_features'] == 4
    assert stats['missing'] == 0


# ===== Trainer Tests =====
def test_train_single(iris):
    X, y = iris
    p = DataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = p.split(X, y)
    X_train, y_train = p.fit_transform(X_train, y_train)
    X_val = p.transform(X_val)

    trainer = ModelTrainer()
    from sklearn.linear_model import LogisticRegression
    result = trainer.train_single(
        'lr', LogisticRegression(max_iter=1000),
        X_train, y_train, X_val, y_val)
    assert result['accuracy'] > 0.5
    assert 'f1_score' in result
    assert 'train_time' in result


def test_train_all(iris):
    X, y = iris
    p = DataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = p.split(X, y)
    X_train, y_train = p.fit_transform(X_train, y_train)
    X_val = p.transform(X_val)

    trainer = ModelTrainer()
    results = trainer.train_all(X_train, y_train, X_val, y_val)
    assert len(results) == 3
    assert trainer.best_model is not None


def test_evaluate(iris):
    X, y = iris
    p = DataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = p.split(X, y)
    X_train, y_train = p.fit_transform(X_train, y_train)
    X_val = p.transform(X_val)
    X_test = p.transform(X_test)

    trainer = ModelTrainer()
    trainer.train_all(X_train, y_train, X_val, y_val)
    metrics = trainer.evaluate(X_test, y_test)
    assert metrics['accuracy'] > 0.8
    assert 'report' in metrics


# ===== Server Tests =====
def test_server_predict(iris):
    X, y = iris
    p = DataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = p.split(X, y)
    X_train, y_train = p.fit_transform(X_train, y_train)
    X_val = p.transform(X_val)

    trainer = ModelTrainer()
    trainer.train_all(X_train, y_train, X_val, y_val)

    server = ModelServer(trainer.best_model[1], p)
    result = server.predict(X[:5])
    assert len(result['predictions']) == 5
    assert 'latency_ms' in result


def test_server_health(iris):
    X, y = iris
    p = DataPreprocessor()
    p.fit(X, y)
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=1000)
    model.fit(p.transform(X), y)
    server = ModelServer(model, p)
    health = server.health_check()
    assert health['status'] == 'healthy'
    assert health['requests_served'] == 0
import numpy as np
import time
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report


class ModelTrainer:
    def __init__(self):
        self.models = {
            'logistic_regression': LogisticRegression(max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
        }
        self.results = {}
        self.best_model = None
        self.best_score = 0

    def train_single(self, name, model, X_train, y_train, X_val, y_val):
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start

        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')

        result = {
            'model': model,
            'accuracy': round(acc, 4),
            'f1_score': round(f1, 4),
            'train_time': round(train_time, 3)
        }
        print(f"  {name:25s} | acc={acc:.4f} | f1={f1:.4f} | time={train_time:.2f}s")
        return result

    def train_all(self, X_train, y_train, X_val, y_val):
        print("Training all models...")
        for name, model in self.models.items():
            result = self.train_single(name, model, X_train, y_train, X_val, y_val)
            self.results[name] = result
            if result['accuracy'] > self.best_score:
                self.best_score = result['accuracy']
                self.best_model = (name, model)
        print(f"\nBest model: {self.best_model[0]} (acc={self.best_score:.4f})")
        return self.results

    def evaluate(self, X_test, y_test):
        if not self.best_model:
            raise RuntimeError("No trained model. Call train_all() first.")
        model = self.best_model[1]
        y_pred = model.predict(X_test)
        return {
            'accuracy': round(accuracy_score(y_test, y_pred), 4),
            'f1_score': round(f1_score(y_test, y_pred, average='weighted'), 4),
            'report': classification_report(y_test, y_pred)
        }

    def save(self, path='models/best_model.pkl'):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.best_model[1], f)
        print(f"Model saved to {path}")

    def load(self, path='models/best_model.pkl'):
        with open(path, 'rb') as f:
            return pickle.load(f)
from preprocessing import DataPreprocessor
from training import ModelTrainer
from serving import ModelServer
import numpy as np


class MLPipeline:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.trainer = ModelTrainer()
        self.server = None

    def run(self, X, y, test_size=0.2):
        print("=" * 55)
        print("   Scalable ML Pipeline")
        print("=" * 55)

        print("\n[1] Preprocessing...")
        stats = self.preprocessor.get_stats(X)
        print(f"  Samples: {stats['n_samples']}, Features: {stats['n_features']}")
        print(f"  Missing values: {stats['missing']}")

        X_train, X_val, X_test, y_train, y_val, y_test = \
            self.preprocessor.split(X, y, test_size=test_size)

        X_train, y_train = self.preprocessor.fit_transform(X_train, y_train)
        X_val = self.preprocessor.transform(X_val)
        X_test = self.preprocessor.transform(X_test)

        print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        print("\n[2] Training...")
        self.trainer.train_all(X_train, y_train, X_val, y_val)

        print("\n[3] Evaluating on test set...")
        metrics = self.trainer.evaluate(X_test, y_test)
        print(f"  Test Accuracy: {metrics['accuracy']}")
        print(f"  Test F1-Score: {metrics['f1_score']}")

        print("\n[4] Starting server...")
        self.server = ModelServer(self.trainer.best_model[1], self.preprocessor)
        health = self.server.health_check()
        print(f"  Status: {health['status']}")
        print(f"  Model: {health['model']}")

        print("\n[5] Sample inference...")
        sample = X[:3]
        result = self.server.predict(sample)
        print(f"  Predictions: {result['predictions']}")
        print(f"  Latency: {result['latency_ms']}ms")

        print("\n" + "=" * 55)
        return metrics


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    data = load_iris()
    pipeline = MLPipeline()
    pipeline.run(data.data, data.target)
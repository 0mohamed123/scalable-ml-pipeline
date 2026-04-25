import numpy as np
import time


class ModelServer:
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor
        self.request_count = 0
        self.total_latency = 0

    def predict(self, X):
        start = time.time()
        X_processed = self.preprocessor.transform(np.array(X))
        prediction = self.model.predict(X_processed)
        latency = (time.time() - start) * 1000

        self.request_count += 1
        self.total_latency += latency

        return {
            'predictions': prediction.tolist(),
            'latency_ms': round(latency, 2)
        }

    def predict_proba(self, X):
        X_processed = self.preprocessor.transform(np.array(X))
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X_processed)
            return {'probabilities': proba.tolist()}
        return {'error': 'Model does not support probability estimates'}

    def health_check(self):
        return {
            'status': 'healthy',
            'model': type(self.model).__name__,
            'requests_served': self.request_count,
            'avg_latency_ms': round(
                self.total_latency / max(1, self.request_count), 2)
        }
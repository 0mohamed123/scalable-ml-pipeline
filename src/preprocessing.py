import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.is_fitted = False

    def fit(self, X, y=None):
        X = np.array(X)
        self.scaler.fit(X)
        if y is not None:
            self.label_encoder.fit(y)
        self.is_fitted = True
        return self

    def transform(self, X, y=None):
        if not self.is_fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")
        X_scaled = self.scaler.transform(np.array(X))
        if y is not None:
            y_encoded = self.label_encoder.transform(y)
            return X_scaled, y_encoded
        return X_scaled

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)

    def split(self, X, y, test_size=0.2, val_size=0.1, random_state=42):
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state)
        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_stats(self, X):
        X = np.array(X)
        return {
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'mean': X.mean(axis=0).tolist(),
            'std': X.std(axis=0).tolist(),
            'missing': int(np.isnan(X).sum())
        }
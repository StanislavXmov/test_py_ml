import numpy as np


class NaiveBayesClassifier:
    """Gaussian Naive Bayes for continuous features."""

    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing
        self.classes_: np.ndarray | None = None
        self.class_count_: np.ndarray | None = None
        self.class_prior_: np.ndarray | None = None
        self.theta_: np.ndarray | None = None
        self.var_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NaiveBayesClassifier":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features)")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("y must be a 1D array with length n_samples")

        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        self.class_count_ = np.zeros(n_classes, dtype=np.int64)
        self.theta_ = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var_ = np.zeros((n_classes, n_features), dtype=np.float64)

        for i, c in enumerate(self.classes_):
            mask = y == c
            X_c = X[mask]
            self.class_count_[i] = X_c.shape[0]
            self.theta_[i] = X_c.mean(axis=0)
            self.var_[i] = X_c.var(axis=0) + self.var_smoothing

        self.class_prior_ = self.class_count_ / n_samples
        return self

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("Classifier is not fitted. Call fit() first.")

        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        log_probs = np.zeros((n_samples, n_classes), dtype=np.float64)

        for i in range(n_classes):
            mean = self.theta_[i]
            var = self.var_[i]
            diff = X - mean
            log_probs[:, i] = -0.5 * np.sum(
                np.log(2.0 * np.pi * var) + (diff**2) / var, axis=1
            ) + np.log(self.class_prior_[i])

        return log_probs

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        log_probs = self._joint_log_likelihood(X)
        log_probs -= log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs)
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        log_probs = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(log_probs, axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))


# X_train = np.array([[0.1, -0.2], [0.3, 0.1], [2.9, 3.1], [3.2, 2.8]])
# y_train = np.array([0, 0, 1, 1])

# classifier = NaiveBayesClassifier()
# classifier.fit(X_train, y_train)

# X_test = np.array([[0.2, -0.1], [1.0, 0.8]])
# y_test = np.array([0, 1])

# print(classifier.predict(X_test))
# print(classifier.predict_proba(X_test))
# print(classifier.score(X_test, y_test))

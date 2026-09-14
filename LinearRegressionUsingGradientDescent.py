import numpy as np

class LinearRegressionUsingGradientDescent:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

        self.mseHistory = [] #for tuning

    def fit(self, X, y):
        num_samples, num_features = X.shape

        self.weights = np.zeros(num_features)
        self.bias = 0.0

        for i in range(self.num_iterations):
            y_pred = X @ self.weights + self.bias
            error = y_pred - y

            dw = (2/num_samples)*(X.T @ error)
            db = (2/num_samples)*np.sum(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            mse = np.mean(np.square(error))
            self.mseHistory.append(mse)

    def mse(self, X, y):
        return np.mean((self.predict(X) - y) ** 2)

    def predict(self, X):
        return X @ self.weights + self.bias





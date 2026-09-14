import itertools
import os
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
)

from LinearRegressionUsingGradientDescent import LinearRegressionUsingGradientDescent
from preprocess import X_train, y_train, y_test, X_test

os.makedirs("logs", exist_ok=True)

FEATURE_NAMES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol"
]

def report_metrics(model, X_train, y_train, X_test, y_test, feature_names=None):
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    print("\n" + "=" * 50)
    print("EVALUATION METRICS")
    print("=" * 50)

    print("\n-- Error metrics --")
    print(f"{'':25}{'Train':>12}{'Test':>12}")
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse  = mean_squared_error(y_test,  y_pred_test)
    print(f"{'MSE':25}{train_mse:>12.4f}{test_mse:>12.4f}")
    print(f"{'RMSE':25}{np.sqrt(train_mse):>12.4f}{np.sqrt(test_mse):>12.4f}")
    print(f"{'MAE':25}"
          f"{mean_absolute_error(y_train, y_pred_train):>12.4f}"
          f"{mean_absolute_error(y_test,  y_pred_test):>12.4f}")

    print("\n-- Goodness of fit --")
    print(f"{'R^2':25}"
          f"{r2_score(y_train, y_pred_train):>12.4f}"
          f"{r2_score(y_test,  y_pred_test):>12.4f}")
    print(f"{'Explained Variance':25}"
          f"{explained_variance_score(y_train, y_pred_train):>12.4f}"
          f"{explained_variance_score(y_test,  y_pred_test):>12.4f}")

    print("\n-- Model parameters --")
    weights = getattr(model, "weights", None)
    if weights is None:
        weights = model.coef_
    bias = getattr(model, "bias", None)
    if bias is None:
        bias = float(np.atleast_1d(model.intercept_)[0])

    print(f"Bias (intercept): {bias:.4f}")
    print("Weight coefficients:")
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(len(weights))]
    for name, w in zip(feature_names, weights):
        print(f"  {name:25} {w:+.4f}")


learning_rates = [0.0001, 0.001, 0.01, 0.1, 0.5]
iteration_counts = [500, 1000, 5000]

best = {"test_mse": float("inf")}

with open("logs/trials.log", "w") as f:
    f.write("lr,n_iters,train_mse,test_mse\n")

    for lr, n_iters in itertools.product(learning_rates, iteration_counts):
        model = LinearRegressionUsingGradientDescent(learning_rate=lr, num_iterations=n_iters)
        model.fit(X_train, y_train)

        train_mse = model.mse(X_train, y_train)
        test_mse  = model.mse(X_test, y_test)

        line = f"{lr},{n_iters},{train_mse:.6f},{test_mse:.6f}\n"
        f.write(line)
        print(line.strip())

        if np.isfinite(test_mse) and test_mse < best["test_mse"]:
            best = {
                "lr": lr,
                "n_iters": n_iters,
                "train_mse": train_mse,
                "test_mse": test_mse,
                "model": model,
            }

    f.write(f"\nBest: lr={best['lr']}, n_iters={best['n_iters']}, "
            f"test_mse={best['test_mse']:.6f}\n")

print(f"\n\nBEST PARAMETERS: lr={best['lr']}, n_iters={best['n_iters']}")
report_metrics(best["model"], X_train, y_train, X_test, y_test, FEATURE_NAMES)

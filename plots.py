import os
import numpy as np
import matplotlib.pyplot as plt

from LinearRegressionUsingGradientDescent import LinearRegressionUsingGradientDescent
from part1 import X_train, X_test, y_train, y_test

os.makedirs("plots", exist_ok=True)

BEST_LR = 0.1
BEST_ITERS = 500

FEATURE_NAMES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol"
]

model = LinearRegressionUsingGradientDescent(learning_rate=BEST_LR, num_iterations=BEST_ITERS)
model.fit(X_train, y_train)

#MSE vs iterations (linear)
plt.figure(figsize=(8, 5))
plt.plot(model.mseHistory)
plt.xlabel("Iteration")
plt.ylabel("MSE (training)")
plt.title(f"MSE vs. iterations (lr={BEST_LR}, iters={BEST_ITERS})")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/mse_vs_iterations.png", dpi=150)
plt.close()

#MSE vs iterations (log)
plt.figure(figsize=(8, 5))
plt.plot(model.mseHistory)
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("MSE (log scale)")
plt.title(f"MSE vs. iterations, log scale (lr={BEST_LR}, iters={BEST_ITERS})")
plt.grid(True, alpha=0.3, which="both")
plt.tight_layout()
plt.savefig("plots/mse_vs_iterations_log.png", dpi=150)
plt.close()

#Learning rate comparison
plt.figure(figsize=(9, 5))
for lr in [0.0001, 0.001, 0.01, 0.1]:
    m = LinearRegressionUsingGradientDescent(learning_rate=lr, num_iterations=1000)
    m.fit(X_train, y_train)
    plt.plot(m.mseHistory, label=f"lr = {lr}")
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("MSE (log scale)")
plt.title("Convergence for different learning rates")
plt.legend()
plt.grid(True, alpha=0.3, which="both")
plt.tight_layout()
plt.savefig("plots/lr_comparison.png", dpi=150)
plt.close()

#Predicted vs actual
y_pred = model.predict(X_test)
plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred, alpha=0.4, edgecolor="k", linewidth=0.3)
lo, hi = y_test.min() - 0.5, y_test.max() + 0.5
plt.plot([lo, hi], [lo, hi], "r--", label="Perfect prediction")
plt.xlabel("Actual quality")
plt.ylabel("Predicted quality")
plt.title("Predicted vs. actual (test set)")
plt.xlim(lo, hi)
plt.ylim(lo, hi)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/predicted_vs_actual.png", dpi=150)
plt.close()

#Feature weights
weights = model.weights
order = np.argsort(np.abs(weights))
sorted_names = [FEATURE_NAMES[i] for i in order]
sorted_weights = weights[order]
colors = ["tab:red" if w < 0 else "tab:blue" for w in sorted_weights]

plt.figure(figsize=(9, 6))
plt.barh(sorted_names, sorted_weights, color=colors)
plt.xlabel("Weight")
plt.title("Feature weights (blue = increases quality, red = decreases)")
plt.axvline(0, color="black", linewidth=0.8)
plt.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("plots/feature_weights.png", dpi=150)
plt.close()


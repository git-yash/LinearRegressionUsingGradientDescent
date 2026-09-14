from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

from preprocess import X_train, y_train, y_test, X_test

learning_rates = [0.0001, 0.001, 0.01, 0.1]
max_iters = [500, 1000, 5000]

best = {"test_mse": float("inf")}

with open("logs/trials_part2.log", "w") as f:
    f.write("eta0,max_iter,train_mse,test_mse\n")

    for eta0 in learning_rates:
        for max_iter in max_iters:
            model = SGDRegressor(
                loss="squared_error",
                learning_rate="constant",
                eta0=eta0,
                max_iter=max_iter,
                tol=None,
                random_state=42,
            )
            model.fit(X_train, y_train)

            train_mse = mean_squared_error(y_train, model.predict(X_train))
            test_mse  = mean_squared_error(y_test,  model.predict(X_test))

            line = f"{eta0},{max_iter},{train_mse:.6f},{test_mse:.6f}\n"
            f.write(line)
            print(line.strip())

            if test_mse < best["test_mse"]:
                best = {
                    "eta0": eta0,
                    "max_iter": max_iter,
                    "train_mse": train_mse,
                    "test_mse": test_mse,
                    "model": model,
                }

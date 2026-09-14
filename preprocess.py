import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_URL_RED = "https://raw.githubusercontent.com/git-yash/LinearRegressionUsingGradientDescent/main/wine%2Bquality/winequality-red.csv"
df_red = pd.read_csv(DATA_URL_RED, sep=";")

df_red = df_red.drop_duplicates().reset_index(drop=True)

X = df_red.drop("quality", axis=1).values # Our features (X)
Y = df_red["quality"].values # trying to predict this value (target)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

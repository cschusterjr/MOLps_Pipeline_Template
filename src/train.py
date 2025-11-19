
import numpy as np, joblib, pathlib
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

OUT = pathlib.Path("artifacts"); OUT.mkdir(exist_ok=True, parents=True)

X, y = make_regression(n_samples=1000, n_features=8, noise=8.0, random_state=7)
model = LinearRegression().fit(X, y)
joblib.dump(model, OUT/"model.pkl")
print("Saved artifacts/model.pkl")

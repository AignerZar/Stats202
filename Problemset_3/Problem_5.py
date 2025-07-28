"""
The following code is used to solve the Problem 5, from the Problemset 3
@author: Zarah Aigner
date: 07.24.2025
"""
# importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
import seaborn as sns

"""
Loading the data
"""
df = pd.read_csv('/Users/zarahaigner/Documents/Stanford/STATS202/Codes_HW_3/Data_Problem_5.csv')
df = df.replace("?", np.nan)
df = df.dropna()

df["horsepower"] = df["horsepower"].astype(float)

X = df[["horsepower"]]
y = df["mpg"]

"""
Fitting the model: linear quadratic and cubic
"""
# linear model
lm = LinearRegression().fit(X, y)

# quadratic model
poly2 = PolynomialFeatures(degree=2)
X_poly2 = poly2.fit_transform(X)
lm2 = LinearRegression().fit(X_poly2, y)

# cubic model
poly3 = PolynomialFeatures(degree=3)
X_poly3 = poly3.fit_transform(X)
lm3 = LinearRegression().fit(X_poly3, y)


"""
Visualization, producing a plot
"""
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="lightgray", label="Data")

x_range = np.linspace(X.min(), X.max(), 100).reshape(-1,1)

plt.plot(x_range, lm.predict(x_range), label="Linear", color="blue")
plt.plot(x_range, lm2.predict(poly2.transform(x_range)), label="Quadratic", color="red")
plt.plot(x_range, lm3.predict(poly3.transform(x_range)), label="Cubic", color="green")

plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.title("Nononlinear models in comparision")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Problem_5.pdf")
plt.show()


"""
Comparision via cross-validation
"""
def rmse_cv(model, X, y):
    mse_scores = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=10)
    return np.mean(-mse_scores)

cv_linear = rmse_cv(LinearRegression().fit(X, y), X, y)
cv_quad = rmse_cv(LinearRegression().fit(X_poly2, y), X_poly2, y)
cv_cubic = rmse_cv(LinearRegression().fit(X_poly3, y), X_poly3, y)

print(f"Linear model RMSE (CV): {cv_linear:.2f}")
print(f"Quadratic model RMSE (CV): {cv_quad:.2f}")
print(f"Cubic model RMSE (CV): {cv_cubic:.2f}")
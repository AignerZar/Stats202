"""
Code to solve problem 9, from the second assignment
@author: Zarah Aigner
date: 12 July, 2025
"""
"""
Importing libraries--------------------------------------------------------------------------------------
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import LeaveOneOut, cross_val_score
import statsmodels.api as sm

#######################################################################################################################
# Exercise a
#########################################################################################################################
"""
Generating the data---------------------------------------------------------------------------------------------
"""
np.random.seed(1)
x = np.random.normal(0, 1, 100)
y = x - 2 * x**2 + np.random.normal(0, 1, 100)

df = pd.DataFrame({"x": x, "y": y})

# n = 100, p = 1
print(f"Number of observations n = {len(x)}")
print(f"Number of predictors p = 1")

# Model in form of an equation:
print("Model: y = x - 2*x^2 + ε, with ε ~ N(0,1)")

#######################################################################################################################
# Exercise b
#########################################################################################################################
plt.scatter(x, y, color='blue')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Scatterplot of X vs Y")
plt.savefig("Problem_9_b.pdf")
plt.show()

#######################################################################################################################
# Exercise c
#########################################################################################################################
def compute_loocv_mse(x, y, degree):
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(x.reshape(-1,1))
    model = LinearRegression()
    loo = LeaveOneOut()
    neg_mse_scores = cross_val_score(model, X_poly, y, cv=loo, scoring='neg_mean_squared_error')
    mse = -np.mean(neg_mse_scores)
    return mse

np.random.seed(2)  # for comparision
degrees = [1, 2, 3, 4]
errors = []

for d in degrees:
    mse = compute_loocv_mse(x, y, d)
    errors.append(mse)
    print(f"LOOCV MSE for degree {d}: {mse:.4f}")

#######################################################################################################################
# Exercise d
#########################################################################################################################
"""
Choosing another seed and doing the whole computations again
"""
np.random.seed(3) 
errors2 = []
for d in degrees:
    mse = compute_loocv_mse(x, y, d)
    errors2.append(mse)
    print(f"(new seed) LOOCV MSE for degree {d}: {mse:.4f}")

#######################################################################################################################
# Exercise d
#########################################################################################################################
"""
Determining which model is the best------------------------------------------------------------------
"""
best_degree = degrees[np.argmin(errors)]
print(f"\nModel with degree {best_degree} has the smallest LOOCV error")

#######################################################################################################################
# Exercise f
#########################################################################################################################
"""
Significance of the coefficients-----------------------------------------------------------------------------------------
"""
for d in degrees:
    poly = PolynomialFeatures(d)
    X_poly = poly.fit_transform(x.reshape(-1,1))
    model_sm = sm.OLS(y, X_poly).fit()
    print(f"\nSignifikanz für Modell mit Grad {d}:")
    print(model_sm.summary())
    print("-" * 60)

"""
The following code is used to solve the Problem 3, from the Problemset 3
@author: Zarah Aigner
date: 07.24.2025
"""
# importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# loading the data
df = pd.read_csv('/Users/zarahaigner/Documents/Stanford/STATS202/Codes_HW_3/Data_Proble_3.csv') 

# converting 'Private' in a boolean value
df['Private'] = df['Private'].map({'Yes': 1, 'No': 0})

X = df.drop("Apps", axis=1)
y = df["Apps"]

# scaling the data, due to the fact ridge and lasso needs standarsized data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

"""
Exercise a: Splitting the data in train and test split (70-30)
"""
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)


"""
Exercise b: fitting a linear model using least squares on the training set
"""
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred_lin = linreg.predict(X_test)
mse_lin = mean_squared_error(y_test, y_pred_lin)
print(f"(b) Test-MSE (OLS): {mse_lin:.2f}")


"""
Exercise c: fitting a ridge regression model on the training set, with lambda chosen by cross validation
"""
alphas = np.logspace(-3, 3, 100)  # λ-values testing
ridge = RidgeCV(alphas=alphas, cv=5)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
print(f"(c) Test-MSE (Ridge): {mse_ridge:.2f}, best λ: {ridge.alpha_:.4f}")

"""
Exercise d: fitting a lasso model ont the training set with lambda chosen by cross validation
"""
lasso = LassoCV(cv=5, max_iter=10000)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
nonzero = np.sum(lasso.coef_ != 0)

print(f"(d) Test-MSE (Lasso): {mse_lasso:.2f}, best λ: {lasso.alpha_:.4f}")
print(f"    Number of non-zero coefficients: {nonzero} of {X.shape[1]}")
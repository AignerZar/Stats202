"""
Code to solve Problem 3, from assignment 4
@author: Zarah Aigner
date: 02 August 2025
"""

# importing necessary libraries
import pandas as pd
import numpy as np
from sklearn. model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


# loading and reading the data
hitters = pd.read_csv("Problem_3_Data.csv", index_col=0)

"""
Exercise a: Removing observations and log transforming salaries
"""
# Drop rows with missing Salary
hitters = hitters.dropna(subset=["Salary"])
# Log-transform Salary
hitters["Salary"] = np.log(hitters["Salary"])

"""
Exercise b: Creating a training and test data set
"""
# Convert categorical variables to dummies
X = pd.get_dummies(hitters.drop("Salary", axis=1), drop_first=True)
y = hitters["Salary"]

X_train, X_test = X.iloc[:200], X.iloc[200:]
y_train, y_test = y.iloc[:200], y.iloc[200:]

"""
Exercise c: Perform boosting with 1000 trees and producing a plot for different shrinkage values
"""
shrinkage_values = np.logspace(-4, 0, 20)
train_mse = []
test_mse = []

for eta in shrinkage_values:
    model = xgb.XGBRegressor(n_estimators=1000, learning_rate=eta, verbosity=0)
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_mse.append(mean_squared_error(y_train, y_pred_train))
    test_mse.append(mean_squared_error(y_test, y_pred_test))

plt.figure(figsize=(10,8))

plt.plot()
plt.plot(shrinkage_values, train_mse, marker='o')
plt.xscale('log')
plt.xlabel('Shrinkage (learning rate)')
plt.ylabel('Training MSE')
plt.title('Training MSE vs Shrinkage')
plt.savefig("Problem_3_c.pdf")
plt.show()

"""
Exercise d: Producign the plot on the test set
"""
plt.plot()
plt.plot(shrinkage_values, test_mse, marker='o', color='red')
plt.xscale('log')
plt.xlabel('Shrinkage (learning rate)')
plt.ylabel('Test MSE')
plt.title('Test MSE vs Shrinkage')
plt.savefig("Problem_3_d.pdf")
plt.show()

"""
Exercise e: Comparing the test MSE of boosting to the test MSE from two regression approaches
"""
# Linear regression
lm = LinearRegression()
lm.fit(X_train, y_train)
lm_mse = mean_squared_error(y_test, lm.predict(X_test))

# Ridge regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ridge = Ridge(alpha=1)
ridge.fit(X_train_scaled, y_train)
ridge_mse = mean_squared_error(y_test, ridge.predict(X_test_scaled))

# best boosting MSE
best_eta = shrinkage_values[np.argmin(test_mse)]
best_boost_mse = min(test_mse)

print(f"Linear Regression MSE: {lm_mse:.3f}")
print(f"Ridge Regression MSE: {ridge_mse:.3f}")
print(f"Boosting Best MSE (eta={best_eta:.4f}): {best_boost_mse:.3f}")

"""
Exercise f: which variables appear most important
"""
best_model = xgb.XGBRegressor(n_estimators=1000, learning_rate=best_eta)
best_model.fit(X_train, y_train)

xgb.plot_importance(best_model, max_num_features=10, importance_type='gain')
plt.title("Top 10 Feature Importances")
plt.savefig("Problem_3_f.pdf")
plt.show()

"""
Exercise g: applying bagging to training set
"""
bag_model = RandomForestRegressor(n_estimators=1000, random_state=1)
bag_model.fit(X_train, y_train)
bag_mse = mean_squared_error(y_test, bag_model.predict(X_test))

print(f"Bagging (Random Forest) Test MSE: {bag_mse:.3f}")


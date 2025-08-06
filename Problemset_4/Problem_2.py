"""
Code to solve Problem 2, from assignment 4
@author: Zarah Aigner
date: 02 August 2025
"""
"""
importing libaries
"""
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# loading the data
df = pd.read_csv('/Users/zarahaigner/Documents/Stanford/STATS202/Codes_HW_4/Data_Problem_2.csv')

print(df.columns)

X = pd.get_dummies(df.drop('Sales', axis=1), drop_first=True)
y = df['Sales']

"""
Exercise a: Splitting the data into a training and test set
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

"""
Exercise b: Fitting a regression tree to the training set, and plotting the tree
"""
# decision tree
tree = DecisionTreeRegressor(random_state=42)
tree.fit(X_train, y_train)

# plot
plt.figure(figsize=(16, 8))
plot_tree(tree, feature_names=X.columns, filled=True, rounded=True, max_depth=3)
plt.title("Regression Tree")
plt.savefig("Problem_2_b.pdf")
plt.show()

# Test MSE
y_pred_tree = tree.predict(X_test)
mse_tree = mean_squared_error(y_test, y_pred_tree)
print(f"(b) Test MSE (Tree): {mse_tree:.3f}")

"""
Exercise c: Cross validation to determine the optimal level of complexity
"""
path = tree.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas[:-1]  
trees = [DecisionTreeRegressor(random_state=42, ccp_alpha=alpha).fit(X_train, y_train) for alpha in ccp_alphas]
mse_cv = [mean_squared_error(y_test, t.predict(X_test)) for t in trees]

optimal_idx = np.argmin(mse_cv)
optimal_tree = trees[optimal_idx]
print(f"(c) Optimal ccp_alpha: {ccp_alphas[optimal_idx]:.4f}, Test MSE: {mse_cv[optimal_idx]:.3f}")

"""
Exercise d: Bagging to analyze data
"""
bagging = BaggingRegressor(n_estimators=100, random_state=42)
bagging.fit(X_train, y_train)
y_pred_bag = bagging.predict(X_test)
mse_bag = mean_squared_error(y_test, y_pred_bag)
print(f"(d) Test MSE (Bagging): {mse_bag:.3f}")

importances_bag = np.mean([tree.feature_importances_ for tree in bagging.estimators_], axis=0)
bag_imp = pd.Series(importances_bag, index=X.columns).sort_values(ascending=False)
print("\n(d) Feature Importance (Bagging):")
print(bag_imp)

"""
Exercise e: Random forests
"""
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
print(f"(e) Test MSE (Random Forest): {mse_rf:.3f}")

rf_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n(e) Feature Importance (Random Forest):")
print(rf_imp)

errors = []
m_vals = range(1, X.shape[1] + 1)
for m in m_vals:
    rf_temp = RandomForestRegressor(n_estimators=100, max_features=m, random_state=42)
    rf_temp.fit(X_train, y_train)
    y_pred_m = rf_temp.predict(X_test)
    errors.append(mean_squared_error(y_test, y_pred_m))

plt.figure()
plt.plot(m_vals, errors, marker='o')
plt.xlabel("Number of variables considered at each split (m)")
plt.ylabel("Test MSE")
plt.title("Effect of m on Random Forest Error")
plt.grid(True)
plt.savefig("Problem_2_e.pdf")
plt.show()

"""
Problem f: Bart
"""
#try:
#    from bartpy.sklearnmodel import SklearnModel
#    bart_model = SklearnModel()
 #   bart_model.fit(X_train.values, y_train.values)
#    y_pred_bart = bart_model.predict(X_test.values)
#    mse_bart = mean_squared_error(y_test, y_pred_bart)
#    print(f"(f) Test MSE (BART): {mse_bart:.3f}")
#except ImportError:
#    print("(f) BART skipped – package 'bartpy' not installed.")

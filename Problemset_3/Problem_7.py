"""
Code to solve the Problem 7, from the third Problemset.
@author: Zarah Aigner
Date: 27 July, 2025
"""
# importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# loading the data
OJ = pd.read_csv('/Users/zarahaigner/Documents/Stanford/STATS202/Codes_HW_3/Data_Problem_7.csv')

X_raw = OJ.drop(columns='Purchase')
X = pd.get_dummies(X_raw, drop_first=True)  

y = OJ['Purchase']

y = y.map({'CH': 1, 'MM': 0})

"""
Exercise a: Creating a training set of 800 random sample observations, and test set the remaining
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=800, random_state=123, stratify=y)


"""
Exercise b: fitting a support vector classifier to training data
"""
svc_linear = SVC(kernel='linear', C=0.01)
svc_linear.fit(X_train, y_train)

print("Training classification report:")
print(classification_report(y_train, svc_linear.predict(X_train)))


"""
Exercise c: Computing the training and test error rates
"""
train_error = 1 - accuracy_score(y_train, svc_linear.predict(X_train))
test_error = 1 - accuracy_score(y_test, svc_linear.predict(X_test))

print(f"Training error: {train_error:.3f}")
print(f"Test error: {test_error:.3f}")


"""
Exercise d: using the tune function to select an optimal cost
"""
param_grid = {'C': [0.01, 0.1, 1, 10]}
grid_linear = GridSearchCV(SVC(kernel='linear'), param_grid, cv=5)
grid_linear.fit(X_train, y_train)

print(f"Best cost (C): {grid_linear.best_params_['C']}")


"""
Exercise e: comparing the training and test error rates using the new value for cost
"""
best_linear = grid_linear.best_estimator_

train_error_best = 1 - accuracy_score(y_train, best_linear.predict(X_train))
test_error_best = 1 - accuracy_score(y_test, best_linear.predict(X_test))

print(f"Best linear SVM training error: {train_error_best:.3f}")
print(f"Best linear SVM test error: {test_error_best:.3f}")


"""
Exercise f: Repeating for a support vector machine with a radial kernel
"""
svc_rbf = SVC(kernel='rbf', C=0.01)
svc_rbf.fit(X_train, y_train)

train_error_rbf = 1 - accuracy_score(y_train, svc_rbf.predict(X_train))
test_error_rbf = 1 - accuracy_score(y_test, svc_rbf.predict(X_test))

print(f"RBF SVM (C=0.01) training error: {train_error_rbf:.3f}")
print(f"RBF SVM (C=0.01) test error: {test_error_rbf:.3f}")

# Tune C for radial kernel
grid_rbf = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
grid_rbf.fit(X_train, y_train)

best_rbf = grid_rbf.best_estimator_
train_error_rbf_best = 1 - accuracy_score(y_train, best_rbf.predict(X_train))
test_error_rbf_best = 1 - accuracy_score(y_test, best_rbf.predict(X_test))

print(f"Best RBF SVM training error: {train_error_rbf_best:.3f}")
print(f"Best RBF SVM test error: {test_error_rbf_best:.3f}")


"""
Exercise g: Repeating with a support vector machine with a polynomial kernel
"""
svc_poly = SVC(kernel='poly', degree=2, C=0.01)
svc_poly.fit(X_train, y_train)

train_error_poly = 1 - accuracy_score(y_train, svc_poly.predict(X_train))
test_error_poly = 1 - accuracy_score(y_test, svc_poly.predict(X_test))

print(f"Poly SVM (C=0.01) training error: {train_error_poly:.3f}")
print(f"Poly SVM (C=0.01) test error: {test_error_poly:.3f}")

# Tune C for poly kernel
grid_poly = GridSearchCV(SVC(kernel='poly', degree=2), param_grid, cv=5)
grid_poly.fit(X_train, y_train)

best_poly = grid_poly.best_estimator_
train_error_poly_best = 1 - accuracy_score(y_train, best_poly.predict(X_train))
test_error_poly_best = 1 - accuracy_score(y_test, best_poly.predict(X_test))

print(f"Best Poly SVM training error: {train_error_poly_best:.3f}")
print(f"Best Poly SVM test error: {test_error_poly_best:.3f}")


"""
Exercise h: Observing which approach seems to give the best result on the data
"""
print("\n--- Final Test Error Comparison ---")
print(f"Linear SVM:       {test_error_best:.3f}")
print(f"Radial SVM:       {test_error_rbf_best:.3f}")
print(f"Polynomial SVM:   {test_error_poly_best:.3f}")

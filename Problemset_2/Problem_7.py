"""
Code for problem 7 of the 2nd assignment
@author: Zarah Aigner
date: 12 July 2025
"""
# importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import accuracy_score

"""
Setting a random seed -> the reproduce the results--------------------------------------------------------------
"""
np.random.seed(42)

"""
loading the data----------------------------------------------------------------
"""
df = pd.read_csv('/Users/zarahaigner/Documents/Stanford/STATS202/Codes_HW_2/Data_Problem_7.csv',  index_col=0)
print(df.head())


"""
converting the data--------------------------------------------------------------------------------------------
"""
df['default'] = (df['default'] == 'Yes').astype(int)


#######################################################################################################################
# Exercise a
##########################################################################################################################
"""
fitting the logistic regression-----------------------------------------------------------------------------------------------
"""
X = df[['income', 'balance']]
X = sm.add_constant(X)
y = df['default']

logit_model_a = sm.Logit(y, X)
result_a = logit_model_a.fit(disp=True)

print("Exercise a) Logistic regression model fitted on the entire dataset:")
print(result_a.summary())

#######################################################################################################################
# Exercise b
##########################################################################################################################
"""
i. Splitting the data in train and val
"""
train, val = train_test_split(df, test_size=0.5, random_state=42)

"""
ii. Fitting logistic regression on training data
"""
X_train = train[['income', 'balance']]
X_train = sm.add_constant(X_train)
y_train = train['default']

logit_model_b = sm.Logit(y_train, X_train)
result_b = logit_model_b.fit(disp=True)

"""
iii. Prediction on the val dataset
"""
X_val = val[['income', 'balance']]
X_val = sm.add_constant(X_val)
y_val = val['default']

y_pred_prob = result_b.predict(X_val)
y_pred = (y_pred_prob > 0.5).astype(int)

"""
iv. Clalculating the val error rate
"""
error_rate = np.mean(y_pred != y_val)
print(f"\nExercise b) Validation error rate: {error_rate:.4f}")


#######################################################################################################################
# Exercise c
##########################################################################################################################
print("\nExercise c) Repeating for 3 different val/train splits:\n")

error_rates = []
for i, seed in enumerate([101, 202, 303], 1):
    train_c, val_c = train_test_split(df, test_size=0.5, random_state=seed)

    X_train_c = train_c[['income', 'balance']]
    X_train_c = sm.add_constant(X_train_c)
    y_train_c = train_c['default']

    logit_model_c = sm.Logit(y_train_c, X_train_c)
    result_c = logit_model_c.fit(disp=False)  # suppress output for brevity

    X_val_c = val_c[['income', 'balance']]
    X_val_c = sm.add_constant(X_val_c)
    y_val_c = val_c['default']

    y_pred_prob_c = result_c.predict(X_val_c)
    y_pred_c = (y_pred_prob_c > 0.5).astype(int)

    error_rate_c = np.mean(y_pred_c != y_val_c)
    error_rates.append(error_rate_c)
    print(f"  Split {i} (random_state={seed}): Validation error rate = {error_rate_c:.4f}")


#######################################################################################################################
# Exercise d
##########################################################################################################################
print("\nExercise d) Adding a dummy variable:\n")

# convert student to dummy (0/1)
df['student'] = (df['student'] == 'Yes').astype(int)

"""
Using same split as in Exercise b (random_state=42)
"""
train_d, val_d = train_test_split(df, test_size=0.5, random_state=42)

X_train_d = train_d[['income', 'balance', 'student']]
X_train_d = sm.add_constant(X_train_d)
y_train_d = train_d['default']

logit_model_d = sm.Logit(y_train_d, X_train_d)
result_d = logit_model_d.fit(disp=True)

"""
Prediction and validation error
"""
X_val_d = val_d[['income', 'balance', 'student']]
X_val_d = sm.add_constant(X_val_d)
y_val_d = val_d['default']

y_pred_prob_d = result_d.predict(X_val_d)
y_pred_d = (y_pred_prob_d > 0.5).astype(int)

error_rate_d = np.mean(y_pred_d != y_val_d)
print(f"\nExercise d) Validation error rate with student dummy: {error_rate_d:.4f}")

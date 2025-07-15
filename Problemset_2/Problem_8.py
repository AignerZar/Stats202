"""
Code for the Problem 8, of the assignment 2.
@author: Zarah Aigner
date: 12 July 2025
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.utils import resample

np.random.seed(123)

Default = pd.read_csv('/Users/zarahaigner/Documents/Stanford/STATS202/Codes_HW_2/Data_Problem_7.csv')
Default['default_bin'] = Default['default'].map({'No': 0, 'Yes': 1})

"""
Preparing the predictors and add the intercept--------------------------------------------------------------------------------------
"""
X = Default[['income', 'balance']]
X = sm.add_constant(X)
y = Default['default_bin']

#######################################################################################################################
# Exercise a
##########################################################################################################################
model = sm.Logit(y, X)
result = model.fit(disp=0)
print("\na) Standard errors from glm (statsmodels):")
print(result.bse[['income', 'balance']])
print()


#######################################################################################################################
# Exercise b
##########################################################################################################################
def boot_fn(data, n_boot=1000):
    coefs = np.zeros((n_boot, 2))  # income and balance coef
    n = len(data)
    for i in range(n_boot):
        sample = resample(data, n_samples=n, replace=True)
        y_sample = sample['default_bin']
        X_sample = sample[['income', 'balance']]
        X_sample = sm.add_constant(X_sample)
        model_sample = sm.Logit(y_sample, X_sample)
        res_sample = model_sample.fit(disp=0)
        coefs[i, :] = res_sample.params[['income', 'balance']]
    return coefs#

#######################################################################################################################
# Exercise c
#########################################################################################################################
boot_coefs = boot_fn(Default, n_boot=1000)
boot_se = boot_coefs.std(axis=0)

print("\nc) Standard errors from bootstrap:")
print(f"Income: {boot_se[0]}")
print(f"Balance: {boot_se[1]}")
print()


#######################################################################################################################
# Exercise d
#########################################################################################################################
print("\nd) Comparison:")
print(pd.DataFrame({
    'Method': ['glm (statsmodels)', 'bootstrap'],
    'SE_Income': [result.bse['income'], boot_se[0]],
    'SE_Balance': [result.bse['balance'], boot_se[1]]
}))
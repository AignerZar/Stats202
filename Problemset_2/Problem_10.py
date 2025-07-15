"""
Code for Problem 10 of the second assignment
@author: Zarah Aigner
date: 13 July 2025
"""
"""
Importing libaries---------------------------------------------------------------------------
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# BOOTSTRAP
def bootstrap_statistic(data, func, n_bootstrap=1000):
    estimates = []
    n = len(data)
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        estimates.append(func(sample))
    return np.array(estimates)

"""
Loading the data--------------------------------------------------------------------------------------------
"""
df = pd.read_csv('/Users/zarahaigner/Documents/Stanford/STATS202/Codes_HW_2/Data_Problem_10.csv')
print(df.columns)
medv = df["medv"]


#######################################################################################################################
# Exercise a
#########################################################################################################################
"""
Computing the population mean
"""
mu_hat = np.mean(medv)
print(f"(a) Estimated mean μ̂: {mu_hat:.4f}")

#######################################################################################################################
# Exercise b
#########################################################################################################################
"""
FInding an estimated standard error
"""
std_error_mu = np.std(medv, ddof=1) / np.sqrt(len(medv))
print(f"(b) SE(μ̂) from formula: {std_error_mu:.4f}")

#######################################################################################################################
# Exercise c
#########################################################################################################################
"""
Bootstrap estimate of SE
"""
boot_means = bootstrap_statistic(medv, np.mean, n_bootstrap=1000)
boot_se_mu = np.std(boot_means, ddof=1)
print(f"(c) Bootstrap SE(μ̂): {boot_se_mu:.4f}")

#######################################################################################################################
# Exercise d
#########################################################################################################################
ci_lower = mu_hat - 2 * boot_se_mu
ci_upper = mu_hat + 2 * boot_se_mu
print(f"(d) 95% CI using bootstrap: [{ci_lower:.4f}, {ci_upper:.4f}]")

# comparision to t-test CI
import scipy.stats as stats
t_ci = stats.t.interval(0.95, df=len(medv)-1, loc=mu_hat, scale=std_error_mu)
print(f"95% CI using t-test: [{t_ci[0]:.4f}, {t_ci[1]:.4f}]")

#######################################################################################################################
# Exercise e
#########################################################################################################################
"""
estimating the median
"""
mu_median_hat = np.median(medv)
print(f"(e) Estimated median μ̂_med: {mu_median_hat:.4f}")


#######################################################################################################################
# Exercise f
#########################################################################################################################
"""
bootstrap SE of the median
"""
boot_medians = bootstrap_statistic(medv, np.median, n_bootstrap=1000)
boot_se_median = np.std(boot_medians, ddof=1)
print(f"(f) Bootstrap SE(μ̂_med): {boot_se_median:.4f}")

#######################################################################################################################
# Exercise g
#########################################################################################################################
"""
estimating the 10th percentile
"""
mu_0_1_hat = np.quantile(medv, 0.1)
print(f"(g) Estimated 10th percentile μ̂_0.1: {mu_0_1_hat:.4f}")


#######################################################################################################################
# Exercise h
#########################################################################################################################
"""
bootstrap SE of the 10th percentile
"""
boot_percentiles = bootstrap_statistic(medv, lambda x: np.quantile(x, 0.1), n_bootstrap=1000)
boot_se_percentile = np.std(boot_percentiles, ddof=1)
print(f"(h) Bootstrap SE(μ̂_0.1): {boot_se_percentile:.4f}")
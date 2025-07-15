"""
Code for Problem 6, from the second assignment
@author: Zarah Aigner
Date: 12 July, 2025
"""
import numpy as np
import matplotlib.pyplot as plt

n_values = np.arange(1, 100001)
prob_in_bootstrap = 1 - (1 - 1 / n_values) ** n_values

plt.figure(figsize=(10,6))
plt.plot(n_values, prob_in_bootstrap, color='blue')
plt.xlabel('Sample size n')
plt.ylabel('P(j is in bootstrap sample)')
plt.title('Probability that the jth observation is in the bootstrap sample')
plt.grid(True)
plt.ylim(0, 1)
plt.savefig("Problem_6_g.pdf")
plt.show()
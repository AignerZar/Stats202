"""
Code to solve Problem 5 (Bonus task), from assignment 4
@author: Zarah Aigner
date: 02 August 2025
"""

# importing libraries
import numpy as np
import matplotlib.pyplot as plt

# defining the sigmoid function+
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# defining the tanh function
def tanh(x):
    return np.tanh(x)

# creating a range of the x values
x = np.linspace(-10, 10, 400)

# computing the function values
sigmoid_y = sigmoid(x)
tanh_y = tanh(x)

# creating the plot
plt.figure(figsize=(8, 5))
plt.plot(x, sigmoid_y, label='Sigmoid', linewidth=2)
plt.plot(x, tanh_y, label='Tanh', linewidth=2)
plt.title('Activation Functions: Sigmoid and Tanh')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.legend()
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig("Problem_5.pdf")
plt.show()
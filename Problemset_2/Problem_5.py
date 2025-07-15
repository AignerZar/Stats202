"""
Code to solve Problem 5, from assignment 2
@author: Zarah Aigner
date: 14 July 2025
"""
"""
importing libaries
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

"""
Loading the data
"""
data = pd.read_csv('/Users/zarahaigner/Documents/Stanford/STATS202/Codes_HW_2/Data_Problem_5.csv')

#######################################################################################################################
# Exercise a
##########################################################################################################################
"""
Plotting the numerical and graphical summaries of the data
"""
print("\n(a) Numerical summary:") #numerical summary
print(data.describe())

sns.pairplot(data, hue='Direction') #pairplot -> looking for patterns
plt.savefig("Problem_5_a1.pdf")
plt.show()

plt.figure(figsize=(10,6)) # Volumen over time
plt.plot(data['Year'], data['Volume'])
plt.title('Volume over time')
plt.savefig("Problem_5_a2.pdf")
plt.show()

#######################################################################################################################
# Exercise b
##########################################################################################################################
"""
COmputing logistic regression with DIrection as the response
"""
X_full = data[['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume']]
y = data['Direction'].apply(lambda x: 1 if x=='Up' else 0)

logreg = LogisticRegression()
logreg.fit(X_full, y)

print("\n(b) Logistic Regression Coefficients")
print(pd.DataFrame({'Feature': X_full.columns, 'Coef': logreg.coef_[0]}))

# Check p-values using statsmodels for significance
import statsmodels.api as sm
X_sm = sm.add_constant(X_full)
model = sm.Logit(y, X_sm).fit()
print(model.summary())

#######################################################################################################################
# Exercise c
##########################################################################################################################
"""
Computing a confusion matrix 
"""
y_pred = logreg.predict(X_full)
cm = confusion_matrix(y, y_pred)
acc = accuracy_score(y, y_pred)

print("\n(c) Confusion Matrix")
print(cm)
print(f"Accuracy: {acc:.4f}")

#######################################################################################################################
# Exercise d
#######################################################################################################################
"""
logistic regression during certain data periods, with Lag2 is the only predictor
"""
train = data[data['Year'] < 2009]
test = data[data['Year'] >= 2009]

X_train = train[['Lag2']]
y_train = train['Direction'].apply(lambda x: 1 if x=='Up' else 0)
X_test = test[['Lag2']]
y_test = test['Direction'].apply(lambda x: 1 if x=='Up' else 0)

logreg2 = LogisticRegression()
logreg2.fit(X_train, y_train)
y_pred_test = logreg2.predict(X_test)

cm2 = confusion_matrix(y_test, y_pred_test)
acc2 = accuracy_score(y_test, y_pred_test)

print("\n(d) Logistic Regression on test data")
print(cm2)
print(f"Accuracy: {acc2:.4f}")


#######################################################################################################################
# Exercise e
#######################################################################################################################
"""
Same as in exercise d but using LDA
"""
lda = LDA()
lda.fit(X_train, y_train)
y_pred_lda = lda.predict(X_test)

cm_lda = confusion_matrix(y_test, y_pred_lda)
acc_lda = accuracy_score(y_test, y_pred_lda)

print("\n(e) LDA on test data")
print(cm_lda)
print(f"Accuracy: {acc_lda:.4f}")


#######################################################################################################################
# Exercise f
#######################################################################################################################
"""
Same as in exercise d but using QDA
"""
qda = QDA()
qda.fit(X_train, y_train)
y_pred_qda = qda.predict(X_test)

cm_qda = confusion_matrix(y_test, y_pred_qda)
acc_qda = accuracy_score(y_test, y_pred_qda)

print("\n(f) QDA on test data")
print(cm_qda)
print(f"Accuracy: {acc_qda:.4f}")

########################################################################################################################
# Exercise g
#######################################################################################################################
"""
Same as in exercise d but using KNN with K=1
"""
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

cm_knn = confusion_matrix(y_test, y_pred_knn)
acc_knn = accuracy_score(y_test, y_pred_knn)

print("\n(g) KNN (K=1) on test data")
print(cm_knn)
print(f"Accuracy: {acc_knn:.4f}")


########################################################################################################################
# Exercise h
#######################################################################################################################
"""
Same as in exercise d but using Naive Bayes
"""
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

cm_nb = confusion_matrix(y_test, y_pred_nb)
acc_nb = accuracy_score(y_test, y_pred_nb)

print("\n(h) Naive Bayes on test data")
print(cm_nb)
print(f"Accuracy: {acc_nb:.4f}")

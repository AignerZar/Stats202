"""
Code for the Final Project, instruction can be found on the course website and the data is provided by the kaggle website
This code is additional to the Code with the name "Code_Finalproject.py", whereas for this code i used a neural network, to be more
precisely a MLP model, I used Pytorch, however Tensorflow could also be used
@author: Zarah Aigner
date: 08-09-2025
"""
# importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

"""
Loading the data and preparing it
"""
train = pd.read_csv("/Users/zarahaigner/Documents/Stanford/STATS202/Finalproject/training.csv")
test = pd.read_csv("/Users/zarahaigner/Documents/Stanford/STATS202/Finalproject/test.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print(train.head())


"""
Exploratory ata Analysis (EDA)
"""
print(train.info())
print(train.describe())

# identifying the target variable
target_col = "relevance"

# distribution of the target variable
sns.countplot(x=target_col, data=train)
plt.title("Target-distribution")
plt.savefig("Target_distribution.pdf")
plt.show()


"""
Split features and target
"""
X = train.drop(columns=[target_col])
y = train[target_col]


"""
Feature scaling
"""
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

"""
Split into train and test split 80-20 split, random splitting
"""
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

"""
Definition of the MLP model and training
"""
MLP_model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32), # 2 hidden layers
    activation='relu', #redctified linear unit (ReLU) as an activation function
    solver='adam', # using adam optimizer
    alpha=1e-4, # L2 regularization
    learning_rate_init=0.0005, # learning rate
    max_iter=300, #maximum of iterations
    random_state=42 # introduce randomness
)

MLP_model.fit(X_train, y_train)

"""
Evaluating the model
"""
# validation accuracy
y_val_pred = MLP_model.predict(X_val)
val_acc = accuracy_score(y_val, y_val_pred)

# training accuracy
y_train_pred = MLP_model.predict(X_train)
train_acc = accuracy_score(y_train, y_train_pred)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")

# Confusion matrix
cm = confusion_matrix(y_val, y_val_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - MLP")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("Confusion_Matrix_MLP.pdf")
plt.show()


"""
retraining on the full data and predict for kaggle test set
"""
X_full_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test.drop(columns=["id"]))

MLP_model.fit(X_full_scaled, y)
test_pred = MLP_model.predict(test_scaled)

# Create submission file
submission = pd.DataFrame({
    'id': test['id'],
    'relevance': test_pred
})

submission.to_csv("submission_MLP.csv", index=False)
print("Kaggle submission saved as submission.csv")



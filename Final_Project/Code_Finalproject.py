"""
Code for the Final Project, instruction can be found on the course website and the data is provided by the kaggle website
@author: Zarah Aigner
date: 08-07-2025
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

"""
Loading and preparing the data 
"""
# loading necessary files
train = pd.read_csv("/Users/zarahaigner/Documents/Stanford/STATS202/Finalproject/training.csv")
test = pd.read_csv("/Users/zarahaigner/Documents/Stanford/STATS202/Finalproject/test.csv")

print("Train shape:", train.shape) #shape of the training data
print("Test shape:", test.shape) #shape of the test data
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
Split into features and targets
"""
X = train.drop(columns=[target_col])
y = train[target_col]


"""
Split into train and test split 80-20 split, random splitting
"""
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


"""
Defining different models -> for better comparision
"""
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "SVM": SVC(kernel='rbf', random_state=42)
}

results = {}


"""
Training and evaluating the different models
"""
for name, model in models.items():
    print(f"\n=== {name} ===")
    model.fit(X_train, y_train)

    # Validation accuracy
    y_val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)

    # Train accuracy
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)

    results[name] = val_acc

    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_val, y_val_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"Confusion_Matrix_{name.replace(' ', '_')}.pdf")
    plt.show()


"""
Determing the best model
"""
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"\nBest model: {best_model_name} with Validation Accuracy = {results[best_model_name]:.4f}")


"""
Prediction for kaggle test data
"""
best_model.fit(X, y)
test_pred = best_model.predict(test)

submission = pd.DataFrame({
    'id': test['id'],       
    'relevance': test_pred  
})

submission.to_csv("submission.csv", index=False)
print("Kaggle submission saved as submission.csv")

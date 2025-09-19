import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Read in dataset as a dataframe
df = pd.read_csv('winequality-red.csv')

# Distribution of the 'quality' variable
plt.figure(figsize=(10, 6))
sns.countplot(x='quality', data=df)
plt.title('Distribution of Wine Quality Ratings')
plt.xlabel('Quality Rating')
plt.ylabel('Frequency')
plt.show()

# Correlation matrix
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Wine Features')
plt.show()


# Convert to binary classification
df['quality'] = (df['quality'] > 5).astype(int)

# Features and target
X = df.drop('quality', axis=1)
y = df['quality']

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

############################################ Model training and hyperparameter tuning #####################################################################

# Instantiate the models
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVC": SVC(random_state=42)
}

# Parameters for Grid Search
param_grid = {
    "Logistic Regression": {'C': np.logspace(-4, 4, 20)},
    "Random Forest": {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, 30, None]},
    "SVC": {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001]}
}

# Grid Search to find the best parameters
for name, model in models.items():
    print(f"Running GridSearchCV for {name}.")
    clf = GridSearchCV(model, param_grid[name], cv=5, scoring='accuracy')
    clf.fit(X_train_scaled, y_train)
    print(f"Best parameters for {name}: {clf.best_params_}")
    models[name] = clf.best_estimator_

##############################Performance Evaluation and Visualization #############################################

# Function to plot ROC Curve
def plot_roc_curve(fpr, tpr, model_name, auc_score):
    plt.plot(fpr, tpr, label=f'{model_name} (area = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")


# Evaluate each model
plt.figure(figsize=(10, 8))
for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    print(f"Classification report for {name}:")
    print(classification_report(y_test, y_pred))

    # Compute the ROC curve
    y_probs = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(
        X_test_scaled)
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    plot_roc_curve(fpr, tpr, name, roc_auc)

plt.show()


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


precision, recall, _ = precision_recall_curve(y_test, y_probs)
pr_auc = auc(recall, precision)

plt.plot(recall, precision, label=f'PR curve (area = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()
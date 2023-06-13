import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

# Load the data from CSV
data = pd.read_csv('data_unificada.csv')

# Splitting the data into features (X) and labels (y)
X = data.drop('expression', axis=1)  # Assuming 'expression' is the column name for labels
y = data['expression']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an SVM classifier
classifier = SVC()

# Train the classifier
classifier.fit(X_train, y_train)

# Predict the labels for test data
y_pred = classifier.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
accuracy_df = pd.DataFrame({'Accuracy': [accuracy]})
accuracy_df.to_csv('accuracySVM.csv', index=False)

# Calculate feature importances
perm_importances = []
for column in X.columns:
    X_test_perm = X_test.copy()
    X_test_perm[column] = np.random.permutation(X_test_perm[column])
    y_pred_perm = classifier.predict(X_test_perm)
    perm_accuracy = accuracy_score(y_test, y_pred_perm)
    importance = accuracy - perm_accuracy
    perm_importances.append(importance)

# Sort and print the feature importances
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': perm_importances})
importance_df = importance_df.sort_values('Importance', ascending=False)
importance_df.to_csv('feature_importancesSVM.csv', index=False)

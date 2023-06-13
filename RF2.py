import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the data from CSV
data = pd.read_csv('data_unificada.csv')

# Splitting the data into features (X) and labels (y)
X = data.drop('expression', axis=1)  # Assuming 'expression' is the column name for labels
y = data['expression']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest classifier with optimized hyperparameters
classifier = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2, min_samples_leaf=1)

# Train the classifier
classifier.fit(X_train, y_train)

# Predict the labels for test data
y_pred = classifier.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Get feature importances
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': classifier.feature_importances_})
importance_df = importance_df.sort_values('Importance', ascending=False)
importance_df.to_csv('feature_importancesRF2.csv', index=False)

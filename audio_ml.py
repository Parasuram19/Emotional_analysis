import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib  # Import joblib for saving the model

# Load the CSV file with features and labels
data = pd.read_csv('emodb_features.csv')

# Separate features and labels
X = data.drop(columns=['label'])
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Print the classification report for more details
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the model to a .pkl file using joblib
model_filename = 'random_forest_emodb.pkl'
joblib.dump(clf, model_filename)
print(f"Model saved to {model_filename}")

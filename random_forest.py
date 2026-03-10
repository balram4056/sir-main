import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("./anti-bully.csv", encoding="latin-1")

# Split the data into training and testing sets (80:20)
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Convert the messages into numerical feature vectors using TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Initialize Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Perform 10-fold cross-validation
cv_scores = cross_val_score(rf_model, X_train_vectors, y_train, cv=10, scoring='f1')

# Train the model on the full training set
rf_model.fit(X_train_vectors, y_train)

# Predict the labels of the test data
y_pred = rf_model.predict(X_test_vectors)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print results
print("Random Forest Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Cross-Validation F1 Scores: {cv_scores}")
print(f"Mean Cross-Validation F1 Score: {np.mean(cv_scores):.4f}")

# Plot the cross-validation F1 scores
plt.figure(figsize=(10,6))
plt.plot(range(1, 11), cv_scores, marker='o')
plt.title('10-Fold Cross-Validation F1 Scores for Random Forest')
plt.xlabel('Fold')
plt.ylabel('F1 Score')
plt.ylim(0, 1)
plt.grid(True)
plt.show()

# Plot the evaluation metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values = [accuracy, precision, recall, f1]
plt.figure(figsize=(8,6))
plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
plt.title('Evaluation Metrics for Random Forest')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.show()

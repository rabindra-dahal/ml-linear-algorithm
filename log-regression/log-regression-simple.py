import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load the wine dataset
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target

# 2. Filter dataset for Binary Classification (Keep only Class 0 and Class 1)
binary_df = df[df['target'].isin([0, 1])]


# 3. Designate Predictor (X) and Binary Target (y)
# Let's use 2 distinct chemical features to predict if it's Class 0 or Class 1
features = ['alcohol', 'color_intensity']
X = binary_df[features]
y = binary_df['target']

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Feature Scaling (Highly recommended for Logistic Regression convergence)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Initialize and Train Logistic Regression
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# 7. Make Classification Predictions
y_pred = model.predict(X_test_scaled)
# You can also extract raw probabilities for each class!
y_prob = model.predict_proba(X_test_scaled)

# 8. Evaluate Classification Statistics
print("--- Classification Performance ---")
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# 9. Predict on an existing row sample from your test dataset
sample_row = X_test.head(1)
sample_scaled = scaler.transform(sample_row)

predicted_class = model.predict(sample_scaled)[0]
predicted_probabilities = model.predict_proba(sample_scaled)[0]

print("\n--- Inference on a Test Sample Row ---")
print(f"Features passed: Alcohol={sample_row['alcohol'].values[0]}, Color Intensity={sample_row['color_intensity'].values[0]}")
print(f"Predicted Class: Class {predicted_class}")
print(f"Probabilities -> Class 0: {predicted_probabilities[0]:.4f} | Class 1: {predicted_probabilities[1]:.4f}")

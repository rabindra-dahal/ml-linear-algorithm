import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Fetch the data using your exact requested URL format
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'
# The dataset has no header row, so we read it raw using header=None
df = pd.read_csv(url, header=None)

# 2. Extract features (Columns 0 to 59) and target label string (Column 60)
X = df.iloc[:, 0:60]
y_raw = df.iloc[:, 60]

# 3. Encode categorical string labels ('M' and 'R') into clean numeric binaries (1 and 0)
# 'M' stands for Mine, 'R' stands for Rock
y = y_raw.map({'M': 1, 'R': 0})

# 4. Split into Training set (80%) and Test set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Feature Scaling (Ensures multi-dimensional weights converge smoothly)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Initialize and Train the Perceptron Model
# max_iter handles epochs; eta0 represents the constant learning rate scale
perceptron_model = Perceptron(max_iter=1000, eta0=0.1, random_state=42)
perceptron_model.fit(X_train_scaled, y_train)

# 7. Make Hard Binary Predictions on Test Set
y_pred = perceptron_model.predict(X_test_scaled)

# 8. Output Model Metrics
print("--- Perceptron Performance on Sonar Dataset ---")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Rock (0)', 'Mine (1)']))

# 9. Safely Predict on an Existing Row from X_test to verify live behavior
existing_sample = X_test.head(1)
original_index = existing_sample.index

# Process and execute single-row prediction
sample_scaled = scaler.transform(existing_sample)
pred_class = perceptron_model.predict(sample_scaled)

print("\n--- Live Prediction on Existing Test Sample ---")
print(f"Sample Original Index: {original_index}")
print(f"True Identity Label:    {y_raw.loc[original_index]} (Encoded: {y_test.loc[original_index]})")
print(f"Predicted Class Label: {pred_class} ({'Mine' if pred_class == 1 else 'Rock'})")

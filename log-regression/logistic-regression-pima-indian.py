import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
"""
If you treat this classification task as a diagnostic screening test, 
your outcomes map directly to Null (H_{0}) and Alternative (H_{1}) hypotheses:
Null Hypothesis (H_{0}): Statement: The patient is healthy and does not have diabetes.
Implication: There is no significant presence of diabetic indicators in the patient's 
chemical/clinical features.

Alternative Hypothesis (H_{1}):Statement: The patient has diabetes.
Implication: The patient's clinical markers (glucose, BMI, age) deviate significantly 
from the healthy baseline cohort.

Evaluation of the model's prediction can be interpreted in this context:
Because your model calculated an 82.42% probability for Class 0, 
you fail to reject the Null Hypothesis (H_{0}). 
Statistically, you conclude that there is insufficient evidence 
to suggest the patient has diabetes.

"""
# 1. Fetch the data using your exact requested URL
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
    'Insulin', 'BMI', 'DiabetesPedigree', 'Age', 'Outcome'
]
df = pd.read_csv(url, names=column_names)

# 2. Split features (X) and target outcome (y)
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# 3. Train/Test Split (80% training, 20% validation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Feature Scaling (Crucial for gradient descent convergence)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Initialize SGDClassifier for Logistic Regression
# loss='log_loss' forces the gradient engine to use log-likelihood loss mapping
sgd_log_reg = SGDClassifier(
    loss='log_loss', 
    max_iter=1000, 
    tol=1e-3, 
    penalty='l2', 
    random_state=42
)
sgd_log_reg.fit(X_train_scaled, y_train)

# 6. Predict classifications and probabilities
y_pred = sgd_log_reg.predict(X_test_scaled)
y_prob = sgd_log_reg.predict_proba(X_test_scaled)

# 7. Print Performance Matrices
print("--- SGD Logistic Regression Performance ---")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. Safely Predict on an Existing Row from X_test 
existing_sample = X_test.head(1)
original_index = existing_sample.index

# Transform and infer on the single-row dataframe structure
sample_scaled = scaler.transform(existing_sample)
pred_class = sgd_log_reg.predict(sample_scaled)
pred_probs = sgd_log_reg.predict_proba(sample_scaled)

print("\n--- Live Prediction on Existing Test Sample ---")
print(f"Sample Original Index: {original_index[0]}")
print(f"True Outcome Label:    {y_test.loc[original_index[0]]}")
print(f"Predicted Class Label: {pred_class[0]}")
# Access specific index points directly to prevent array string formatting errors
print(f"Confidence -> No Diabetes (0): {pred_probs[0][0]:.4f} | Diabetes (1): {pred_probs[0][1]:.4f}")

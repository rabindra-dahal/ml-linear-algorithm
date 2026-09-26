import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load the dataset from a live remote URL
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/auto-insurance.csv"
df = pd.read_csv(url, header=None, names=['Claims', 'TotalPayment'])

# 2. Split into features (X) and target variable (y)
# Sklearn expects a 2D array for features, so we use [['Claims']]
X = df[['Claims']] 
y = df['TotalPayment']

# 3. Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and fit the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predict on the test data
y_pred = model.predict(X_test)
print("Predictions on test data:", y_pred)

# 6. Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Intercept (Base Payment): {model.intercept_:.2f}")
print(f"Model Coefficient (Weight per Claim): {model.coef_[0]:.2f}")
print("--------------------------------------------------")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2 Score): {r2:.4f}")

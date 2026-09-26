import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Generate sample data (Predicting House Price based on Size and Number of Bedrooms)
# In real use cases, you would load your data using: df = pd.read_csv('your_file.csv')
data = {
    'Size_SqFt': [1500, 1800, 2400, 3000, 1200, 1700, 2200, 2800, 1600, 2500],
    'Bedrooms': [3, 3, 4, 4, 2, 3, 3, 5, 3, 4],
    'Price': [300000, 360000, 470000, 590000, 250000, 350000, 430000, 560000, 320000, 500000]
}
df = pd.DataFrame(data)

# 2. Separate independent variables (X) and dependent variable (y)
X = df[['Size_SqFt', 'Bedrooms']]
y = df['Price']

# 3. Split the data into Training set (80%) and Test set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and train the Multiple Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Make predictions on the test data
y_pred = model.predict(X_test)

# 6. Evaluate the model performance
print("--- Model Coefficients ---")
print(f"Intercept (b0): {model.intercept_:.2f}")
print(f"Coefficients (b1, b2): {model.coef_}")

print("\n--- Evaluation Metrics ---")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.2f}")
print(f"R-squared (R²) Score: {r2_score(y_test, y_pred):.4f}")

# 7. Predict for a custom new house (e.g., 2000 sq ft, 3 bedrooms)
new_house = np.array([[2000, 3]])
predicted_price = model.predict(new_house)
print(f"\nPredicted price for a 2000 sq ft house with 3 bedrooms: ${predicted_price[0]:,.2f}")

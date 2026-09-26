"""
This script demonstrates multivariate linear regression using Python's scikit-learn library.
In this example, we predict two target variables (Price and Days_on_Market) based on 
two features (Size_SqFt and Bedrooms) of a house. The model learns the relationship between 
the features and the targets, allowing us to make predictions for new data points.
The dataset is generated within the script for demonstration purposes, but in real-world scenarios,
you would typically load your data from a CSV file or a database.

Multivariate regression is often confused with multiple linear regression, 
but they are fundamentally different.While multiple regression uses 
multiple independent variables (X₁, X₂) to predict one single dependent variable (y), 
multivariate regression predicts two or more dependent variables (y₁, y₂) simultaneously 
from the same set of inputs.For example, you might want to predict both a house's Price (y₁) 
and Days on Market (y₂) at the same time using its size and number of bedrooms. Fortunately, 
scikit-learn's LinearRegression handles multivariate targets natively if you pass a 2D array 
or multiple columns for y.

"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 1. Generate sample data
# Predictors (X): Size, Bedrooms
# Targets (y): Price, Days_on_Market
data = {
    'Size_SqFt': [1500, 1800, 2400, 3000, 1200, 1700, 2200, 2800, 1600, 2500],
    'Bedrooms': [3, 3, 4, 4, 2, 3, 3, 5, 3, 4],
    'Price': [300000, 360000, 470000, 590000, 250000, 350000, 430000, 560000, 320000, 500000],
    'Days_on_Market': [35, 20, 45, 15, 30, 10, 50, 25, 40, 18],
}
df = pd.DataFrame(data)

# 2. Separate multiple features (X) and multiple targets (y)
X = df[['Size_SqFt', 'Bedrooms']]
y = df[['Price', 'Days_on_Market']]  # Note the double brackets for multiple targets

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Multivariate model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predict on test data
y_pred = model.predict(X_test)

# 6. Evaluate the model
# The model outputs a matrix of coefficients and intercepts (one set for each target)
print("--- Model Intercepts ---")
print(f"Intercepts (Price, Days): {model.intercept_}")

print("\n--- Model Coefficients ---")
print("Rows = Targets (Price, Days), Columns = Features (Size, Bedrooms)")
print(model.coef_)

# R² score evaluates how well the model predicts both targets
print("\n--- Evaluation (R² Scores) ---")
print(f"R² Scores for [Price, Days]: {r2_score(y_test, y_pred, multioutput='raw_values')}")

# Assuming y_test and y_pred are your true and predicted 2D arrays from the model
residuals = y_test - y_pred

# Calculate the Residual Covariance Matrix
cov_matrix = np.cov(residuals, rowvar=False)
print("Residual Covariance Matrix:\n", cov_matrix)

# 7. Predict for a new custom house (Fixing the feature name warning as well)
new_house = pd.DataFrame([[2000, 3]], columns=['Size_SqFt', 'Bedrooms'])
predicted_metrics = model.predict(new_house)

print(f"\nPredictions for a 2000 sq ft, 3-bedroom house:")
print(f"Predicted Price: ${predicted_metrics[0][0]:,.2f}")
print(f"Predicted Days on Market: {predicted_metrics[0][1]:.1f} days")



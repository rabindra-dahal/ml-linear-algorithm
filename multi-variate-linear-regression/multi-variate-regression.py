import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error

# 1. Load the built-in wine dataset
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)

# 2. Designate Multiple Predictors (X) and Multiple Targets (y)
features = ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'color_intensity']
targets = ['total_phenols', 'flavanoids']

X = df[features]
y = df[targets]

# 3. Train/Test Split (80% training, 20% validation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Feature Scaling (Crucial for Stochastic Gradient Descent)
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# 5. Initialize SGD and wrap it in MultiOutputRegressor
# max_iter controls epochs; tol is the stopping criteria; learning_rate handles step sizes
base_sgd = SGDRegressor(max_iter=1000, tol=1e-3, learning_rate='invscaling', random_state=42)
multivariate_sgd = MultiOutputRegressor(base_sgd)

# 6. Train the model
multivariate_sgd.fit(X_train_scaled, y_train)

# 7. Predict on Test Set
y_pred = multivariate_sgd.predict(X_test_scaled)

# 8. Extract Model Statistics
print("--- Individual Target Fit ($R^2$ Scores) ---")
r2_scores = r2_score(y_test, y_pred, multioutput='raw_values')
for target, score in zip(targets, r2_scores):
    print(f"{target}: {score:.4f}")

print("\n--- Model Coefficients per Estimator ---")
for idx, target in enumerate(targets):
    # Access individual underlying SGD estimators
    estimator = multivariate_sgd.estimators_[idx]
    print(f"\n{target} Intercept: {estimator.intercept_[0]:.4f}")
    print(f"{target} Coefficients:")
    coef_df = pd.Series(estimator.coef_, index=features)
    print(coef_df.to_string())

# 9. Global Statistic: Residual Covariance Matrix (Σ)
residuals = y_test - y_pred
residual_covariance = np.cov(residuals, rowvar=False)

print("\n--- Residual Covariance Matrix (Σ) ---")
print(pd.DataFrame(residual_covariance, index=targets, columns=targets))

# 1. Define a brand new wine sample with custom chemical measurements
# Features: ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'color_intensity']
custom_wine_values = [[13.5, 2.3, 2.4, 19.0, 100.0, 5.5]] 
new_wine_data = pd.DataFrame(custom_wine_values, columns=features)

# 2. Scale the new input data (CRUCIAL: use .transform(), NOT .fit_transform())
new_wine_scaled = scaler_X.transform(new_wine_data)

# 3. Generate predictions for both targets simultaneously
predicted_outputs = multivariate_sgd.predict(new_wine_scaled)

# 4. Display the results by slicing the array indices [row, column]
print("--- Custom Wine Predictions ---")
# [0, 0] accesses row 0, column 0 (total_phenols)
print(f"Predicted Total Phenols: {predicted_outputs[0, 0]:.4f}") 
# [0, 1] accesses row 0, column 1 (flavanoids)
print(f"Predicted Flavanoids:    {predicted_outputs[0, 1]:.4f}") 

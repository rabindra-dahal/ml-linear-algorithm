import numpy as np
from sklearn import linear_model

# 1. Initialize the estimator
reg = linear_model.LinearRegression()

# 2. Define features (X) and target variable (y)
X = [[0, 0], [1, 1], [2, 2]]
y = [0, 1, 2]

# 3. Fit the model to the data
reg.fit(X, y)

# 4. Inspect learned parameters
print("Coefficients:", reg.coef_)     # Weights for each feature
print("Intercept:", reg.intercept_)   # Bias term

# 5. Predict values for new, unseen data
# Pass a 2D array-like structure with the same number of features (2 columns)
X_new = [[3, 3], [0, 1]]
predictions = reg.predict(X_new)

print("Predictions:", predictions)
# Output will be close to: [3. , 0.5]
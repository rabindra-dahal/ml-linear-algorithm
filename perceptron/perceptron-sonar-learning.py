import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Download and load the Sonar dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'

df = pd.read_csv(url, header=None)

# 2. Preprocess data
# The last column (60) contains the labels 'R' or 'M'
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Map string labels to binary integers: Mine ('M') -> 1, Rock ('R') -> 0
y = np.where(y == 'M', 1, 0)

# Split dataset into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Define the Perceptron Class from scratch
class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=500):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape
        # Initialize weights to zeros (or small random numbers)
        self.weights = np.zeros(num_features)
        self.bias = 0.0

        # Training loop
        for epoch in range(self.epochs):
            for idx, x_i in enumerate(X):
                # Calculate the linear activation
                activation = np.dot(x_i, self.weights) + self.bias
                
                # Apply step function
                y_predicted = 1.0 if activation >= 0.0 else 0.0
                
                # Update rule if there is a misclassification
                error = y[idx] - y_predicted
                if error != 0:
                    self.weights += self.lr * error * x_i
                    self.bias += self.lr * error

    def predict(self, X):
        # Compute predictions for an entire array of inputs
        activation = np.dot(X, self.weights) + self.bias
        return np.where(activation >= 0.0, 1, 0)

# 4. Train the Perceptron Model
model = Perceptron(learning_rate=0.05, epochs=200)
model.fit(X_train, y_train)

# 5. Evaluate the model
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

train_accuracy = accuracy_score(y_train, train_preds)
test_accuracy = accuracy_score(y_test, test_preds)

print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")

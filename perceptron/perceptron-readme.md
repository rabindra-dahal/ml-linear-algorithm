# 🎂 Simple Python Perceptron (The Cake Judge)

A clean, beginner-friendly implementation of the **Perceptron Algorithm** built from scratch using only raw Python and NumPy. No complex machine learning libraries required!

---

## 🎨 How it Works: The Cake Judge Analogy

Think of this Perceptron as a **Cake Judge** trying to decide if a recipe is **Good (1)** or **Bad (0)** based on two ingredients: **Sugar** and **Salt**.

*   **Inputs:** The ingredients (How much Sugar and Salt are in the cake).
*   **Weights:** The judge's opinion on an ingredient (e.g., High Sugar = positive weight, High Salt = negative weight).
*   **Learning Rate (`lr`):** How drastically the judge changes their mind when you tell them they made a mistake.
*   **Epochs:** How many times the judge practices on the exact same recipe book.

---

## 🛠️ Code Structure

```python
import numpy as np

class SimplePerceptron:
    def __init__(self, learning_rate=0.1, epochs=10):
        self.lr = learning_rate       # How fast it learns
        self.epochs = epochs          # Practice rounds
        self.weights = None           # Ingredient importance
        self.bias = 0.0               # Default mood of the judge

    def fit(self, X, y):
        # Starts with zero knowledge
        self.weights = np.zeros(X.shape[1])
        
        for epoch in range(1, self.epochs + 1):
            errors = 0
            for idx in range(len(X)):
                # 1. Taste the cake (Calculate score)
                score = np.dot(X[idx], self.weights) + self.bias
                
                # 2. Make a guess (Yes/No)
                prediction = 1.0 if score >= 0.0 else 0.0
                
                # 3. Check if wrong
                error = y[idx] - prediction
                
                # 4. Adjust the rule if a mistake happened
                if error != 0:
                    errors += 1
                    self.weights += self.lr * error * X[idx]
                    self.bias += self.lr * error
            
            print(f"Epoch {epoch}: Errors Made = {errors}")
            if errors == 0:
                print("🎉 Perfect rules learned! Stopping early.")
                break
```

---

## 🚀 How to Run

1. Make sure you have **NumPy** installed:
   ```bash
   pip install numpy
   ```

2. Run the script:
   ```bash
   python perceptron.py
   ```

## 📊 Expected Output

The judge starts out guessing randomly, makes errors, adjusts its rules, and eventually gets everything right:

```text
--- Starting Training Loop ---
Epoch 1: Errors Made = 3
Epoch 2: Errors Made = 2
Epoch 3: Errors Made = 1
Epoch 4: Errors Made = 0
🎉 Perfect rules learned! Stopping early.
```

## 📄 License
MIT License. Feel free to use and remix this code!

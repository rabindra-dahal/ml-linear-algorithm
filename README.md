# 📈 Machine Learning Foundations: Regression & Classification

This repository contains foundational Machine Learning concepts, grouping our journey from continuous mathematical predictions to discrete binary decisions.

---

## 🧠 Core Architectural Concepts

### 1. Feature Scaling (The Balance Principle)
Machine learning models only see raw numbers; they do not understand real-world context. If you feed a model a patient's **Age** (numbers like 20 to 80) alongside their **Insulin Level** (numbers like 15 to 400), the model will assume Insulin is vastly more important simply because the numbers are larger. 

We use a `StandardScaler` to fix this by leveling the playing field. It shifts all features to center around a baseline average of `0`, with normal variations spanning between `-3` and `+3`.

*   **Training Phase (`fit_transform`):** The scaler reviews your training data to learn its specific average (mean) and spread (variance). It immediately uses those patterns to rewrite the training data onto the new scaled level.
*   **Testing Phase (`transform`):** The scaler locks its memory. When looking at brand-new test data or custom patient entries, it **never calculates a new average**. It forces the new data to be evaluated against the exact same rules learned during the training phase. This prevents data leakage and ensures consistent predictions.
*   **Why Numbers Turn Positive and Negative:** The scaling math subtracts the group average from each row. If a patient's blood pressure is **below average**, their scaled score becomes a **negative** number. If their blood pressure is **above average**, their scaled score becomes a **positive** number. A score of exactly `0` means they are perfectly average.

### 2. Standard Regression vs. Multivariate Regression
*   **Multiple Regression:** Uses multiple data points (like house size, number of bedrooms, and local school ratings) to predict **one single continuous result** (the final house price).
*   **Multivariate Regression:** Predicts **two or more continuous results simultaneously** (like predicting both the house price and how many days it will sit on the market before selling). It uses a special error matrix called a **Residual Covariance Matrix** to track whether its calculation mistakes on one target are linked to its mistakes on the other target.

---

## ⚙️ Model Variations Explored

### 1. Continuous Regressions (Predicting Quantities)
*   **Simple Linear Regression:** Fits a single, perfectly straight line to the data to find a basic trend (e.g., tracking how a house's price rises purely based on its square footage).
*   **Multivariate Stochastic Gradient Descent (SGD):** Instead of calculating the perfect rule in one massive math step, this model takes a journey. It reviews the wine dataset, takes small steps down an error hill, and constantly tweaks its rules to accurately predict multiple chemical properties at the exact same time.

### 2. Binary Classifications (Predicting Classes)
*   **SGD Logistic Regression:** Tested using the real **Pima Indians Diabetes Dataset**. This algorithm doesn't just guess "yes" or "no"—it calculates continuous probabilities. For example, it can evaluate a patient's metrics and state with 82.42% certainty that the patient belongs to the healthy cohort (Class 0), failing to reject the baseline hypothesis that they are free of diabetes.
*   **The Perceptron Algorithm:** Tested using the **Sonar Dataset** to distinguish between rocks and explosive metal mines. Unlike Logistic Regression, a standard Perceptron provides **no confidence percentages**. It uses a strict, hard boundary. It is a strictly **linear model**, meaning it can only successfully separate data if a perfectly flat line or plane can cleanly divide the two categories.

---

## 🎂 The Perceptron Analogy (For Beginners)

If the algorithm behavior feels complex, visualize our **Simple Perceptron** implementation using the **Cake Judge Analogy**:

*   **The Inputs:** The raw evidence (How many spoons of Sugar and Salt are baked into the cake).
*   **The Weights:** The judge's internal preference rules (High sugar creates a positive weight; high salt creates a heavy negative penalty weight).
*   **The Learning Rate:** A strictness dial controlling how drastically the judge changes their mind and rewrites their entire preference handbook whenever you yell *"Wrong!"* after a bad guess.
*   **The Epochs:** Complete practice rounds. The judge tastes every single cake recipe in the kitchen from start to finish. That is one Epoch. If they made mistakes, you clear the table, bring out the exact same cakes, and force them to do another pass (Epoch 2) until they can clear the entire table with **zero errors**.

---

## 📊 Project Datasets Applied
*   **Wine Dataset (UCI):** Continuous chemical attributes used to predict multiple target properties at once.
*   **Pima Indians Diabetes Dataset (GitHub/UCI):** Real-world clinical indicators used to predict a binary healthy vs. diabetic outcome.
*   **Sonar Dataset (GitHub/UCI):** 60 distinct continuous sonar energy bounce frequencies used to classify metal cylinders vs. ocean rocks.

---

## 📄 License
This project collection is completely open-source and distributed under the terms of the MIT License.

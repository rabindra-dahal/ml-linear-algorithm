# Linear Regression Implementations in Python

A clean, production-ready reference implementation of various linear regression techniques using **Python 3**, **scikit-learn**, and **pandas**. This project demonstrates how to structure, train, and evaluate models ranging from single-variable predictions to multi-output systems.

## 📌 Regression Types Included

- **Simple Linear Regression:** Models the relationship between 1 Independent Variable (X) and 1 Dependent Variable (y).
- **Multiple Linear Regression:** Models the relationship between 2+ Independent Variables (X₁, X₂) and 1 Dependent Variable (y).
- **Multivariate Linear Regression:** Models the relationship between 2+ Independent Variables (X₁, X₂) and **2+ Dependent Variables** (y₁, y₂) simultaneously.

---

## 🛠️ Installation & Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com
   cd linear-regression-python
   ```

2. **Create a virtual environment (Optional but recommended):**

   ```bash
   python -bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install numpy pandas scikit-learn matplotlib
   ```

---

## 📊 Key Evaluation Metrics

The scripts display vital statistical summaries upon completion:

- **$R^2$ Score (Coefficient of Determination):** Evaluates variance coverage for individual outputs.
- **Mean Squared Error (MSE):** Measures overall variance penalty.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

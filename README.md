# Predictive Modeling on Synthetic Data

***This project focuses on building and evaluating predictive models using a synthetic dataset. The goal is to develop a regression model that accurately predicts target values from a given test dataset using various machine learning techniques.***  

---

## **Project Overview**  
This project, originally designed as a **data science take-home challenge**, involves developing a **regression model** using a synthetic dataset. The dataset contains **5,000 training records and 1,000 test records**, each with **254 features plus a target variable**. The primary evaluation metric for model performance is **Mean Squared Error (MSE)**.

The task involves:
- **Data Preprocessing**: Handling missing values, feature scaling, and encoding categorical variables.
- **Feature Selection & Engineering**: Identifying key predictors to optimize model accuracy.
- **Model Training**: Implementing and evaluating multiple regression models.
- **Performance Comparison**: Assessing models using **Root Mean Squared Error (RMSE)** and **R-Squared ($R^2$)**.

---

## **Technical Skills Demonstrated**  
- **Data Cleaning & Preprocessing**: Standardizing features, handling missing values, and encoding categorical variables.
- **Machine Learning Models**: Training **Linear Regression, Random Forest, Lasso, and Ridge Regression**.
- **Model Evaluation**: Using **MSE, RMSE, and R-Squared ($R^2$)** to assess performance.
- **Data Visualization**: Generating **comparative bar plots** for performance metrics.

---

## **Files in This Repository**  
| File | Description |
|------|------------|
| `Predictive Modeling on Synthetic Data.ipynb` | Jupyter Notebook containing model implementation and evaluation. |
| `codetest_train.txt` | Training dataset with 5,000 records and 254 features. |
| `codetest_test.txt` | Test dataset with 1,000 records and 254 features. |

---

## **How to Run This Project**  
1. **Clone the repository**  
   ```bash
   git clone https://github.com/JordanConallLuthaisWright/Predictive-Modeling-on-Synthetic-Data.git
   cd Predictive-Modeling-Synthetic-Data
2. **Install dependencies**
3. **Open the Jupyter Notebook**
   ```bash
   jupyter notebook
4. Run **Predictive Modeling on Synthetic Data.ipynb** step-by-step.

---

Business Context & Research Questions
-------------------------------------
**Objective:**
This project evaluates different regression models to answer the following questions:
1. How do different regression models compare in terms of RMSE and R-Squared?
2. Which model provides the most accurate target predictions?
3. What trade-offs exist between interpretability and predictive performance?
"""

# Import necessary libraries
- import matplotlib.pyplot as plt
- import numpy as np
- import pandas as pd
- from sklearn.linear_model import LinearRegression, Lasso, Ridge
- from sklearn.ensemble import RandomForestRegressor
- from sklearn.preprocessing import StandardScaler, LabelEncoder
- from sklearn.model_selection import train_test_split
- from sklearn.metrics import mean_squared_error, r2_score

---

Methodology & Approach
----------------------
**1. Data Cleaning & Preprocessing**
   - Loaded training and test datasets.
   - Standardized numerical features using StandardScaler.
   - Encoded categorical variables using Label Encoding where applicable.
   - Split data into training and validation sets for model evaluation.

      **Load datasets (update file paths as needed)**
      - train_data = pd.read_csv("codetest_train.txt", delimiter="\t")
      - test_data = pd.read_csv("codetest_test.txt", delimiter="\t")

      **Standardizing numerical features**
      - scaler = StandardScaler()
      - X_train_scaled = scaler.fit_transform(train_data.iloc[:, :-1])
      - X_test_scaled = scaler.transform(test_data.iloc[:, :-1])

      **Target variable**
      - y_train = train_data.iloc[:, -1]

      **Splitting into training and validation sets**
      - X_train, X_val, y_train, y_val = train_test_split(
      - X_train_scaled, y_train, test_size=0.2, random_state=42
)


**2. Model Training & Evaluation**
   The following regression models were implemented and evaluated:
   - Linear Regression – Simple, interpretable, and fast.
   - Random Forest Regression – Captures non-linearity and interactions.
   - Lasso Regression – Feature selection via L1 regularization.
   - Ridge Regression – Controls overfitting via L2 regularization.


      **Initialize models**
      - lr_model = LinearRegression()
      - rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
      - lasso_model = Lasso(alpha=0.1)
      - ridge_model = Ridge(alpha=1.0)

      **Train models**
      - models = {"Linear Regression": lr_model, "Random Forest": rf_model, "Lasso": lasso_model, "Ridge": ridge_model}
      - for name, model in models.items():
      - model.fit(X_train, y_train)

**3. Performance Metrics**
   - Mean Squared Error (MSE) – Measures average squared prediction error.
   - Root Mean Squared Error (RMSE) – Square root of MSE for easier interpretation.
   - R-Squared (R²) – Explains variance captured by the model.

      **Evaluate models**
      - rmse_values = []
      - r2_values = []

  
**4. Data Visualization**
   - Bar plots comparing RMSE and R-Squared across models using Matplotlib.

      **Define color palette**
      - color1 = '#98C5C0'  # Light teal
      - color2 = '#A880DC'  # Light purple

      **Creating the bar plot for RMSE and R-Squared**
      - fig, ax = plt.subplots(1, 2, figsize=(15, 6))

        ax[0].barh(list(models.keys()), rmse_values, color=color1)
        ax[0].set_xlabel("Root Mean Squared Error (RMSE)")
        ax[0].set_title("Comparison of RMSE among Four Models")

        ax[1].barh(list(models.keys()), r2_values, color=color2)
        ax[1].set_xlabel("R-Squared ($R^2$)")
        ax[1].set_title("Comparison of $R^2$ among Four Models")

        plt.tight_layout()
        plt.show()

--- 

Key Findings & Conclusion
-------------------------
- Random Forest achieved the lowest RMSE, making it the most accurate model.
- Linear Regression had the highest R-Squared (R²), indicating strong interpretability.
- Regularized models (Lasso & Ridge) reduced overfitting, balancing accuracy and generalization.
- Visualizing RMSE and R-Squared helped select the best model for real-world predictions.

Next Steps & Future Improvements
---------------------------------
- Hyperparameter tuning to further optimize model performance.
- Feature selection using importance scores from Random Forest.
- Try advanced models like Gradient Boosting (XGBoost) for better predictions.

---





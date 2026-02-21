# Customer Churn Analysis & Prediction in Telecom.

## Overview
This project aims to predict customer turnover (churn) in the telecommunications industry using machine learning. By analyzing customer demographics, service usage, and account information, we identify key drivers of churn and build predictive models to help the business improve retention strategies.

## Repository Structure
- **`data/`**: Contains the raw (`Customer churn dataset.csv`) and cleaned (`customer_churn_cleaned.csv`) datasets.
- **`scripts/`**:
    - `audit_churn_data.py`: Audits the raw data for missing values and data types.
    - `clean_churn_data.py`: Preprocesses the data (imputation, encoding, cleaning).
    - `train_models.py`: Trains and evaluates Logistic Regression and Random Forest models.
    - `analyze_coefficients.py`: Analyzes the Logistic Regression model to find top churn drivers.
- **`results/`**: Contains performance metrics (`model_results.csv`) and visualization plots.
- **`technical_report.md`**: A detailed report on the methodology, findings, and recommendations.

## How to reproduce findings
1.  **Data Audit**:
    Inspect the raw data to understand its structure and quality issues.
    python3 audit_churn_data.py

2.  **Data Cleaning**:
    Process the raw data (handle missing `TotalCharges`, drop `customerID`, encode categoricals).
    python3 clean_churn_data.py

    *Output*: `Dataset/customer_churn_cleaned.csv`

3.  **Model Training**:
    Train Logistic Regression and Random Forest models.
    python3 train_models.py

    *Output*: `model_results.csv`, `logistic_regression.pkl`, `random_forest.pkl`

4.  **Feature Importance Analysis**:
    Extract coefficients from the best model to identify churn drivers.
    python3 analyze_coefficients.py
    
    *Output*: `key findings`

## Key Findings

- **Best Model**: Logistic Regression (Accuracy: 82.2%, F1-Score: 0.64).
  
- **Top Churn Drivers (+)**:
    1.  **Fiber Optic Internet**: Customers with this service are significantly more likely to leave.
    2.  **Paperless Billing**: Correlated with higher churn.
    3.  **Electronic Check**: The most churn-prone payment method.
    
- **Top Retention Factors (-)**:
    1.  **Two-Year Contracts**: The strongest predictor of retention.
    2.  **One-Year Contracts**: Also very effective.
    3.  **Online Security**: Value-added services increase customer stickiness.

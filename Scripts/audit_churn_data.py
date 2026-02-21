import pandas as pd
import numpy as np

"""
Script: audit_churn_data.py
Description: 
    This script performs an initial audit of the Customer Churn dataset.
    It inspects the data structure, identifies missing values (specifically in 'TotalCharges'),
    and categorizes columns into numerical and categorical types.
    This step is crucial before any data cleaning or modeling can take place.

Usage:
    python3 audit_churn_data.py
"""

def audit_dataset(file_path):
    """
    Audits the dataset for missing values, data types, and structural issues.

    Args:
        file_path (str): The absolute path to the raw CSV dataset.
    """
    print(f"Auditing dataset: {file_path}")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return

    print(f"Dataset Shape: {df.shape}")
    print("-" * 20)

    # 1. Check for missing values in 'TotalCharges'
    # 'TotalCharges' is often loaded as an object (string) because it contains empty strings " " 
    # for customers with 0 tenure. We need to identify these.
    print("Checking 'TotalCharges'...")
    
    # Attempt to convert to numeric, coercing errors (strings) to NaN
    total_charges_numeric = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Count missing values (NaNs from coercion + original NaNs)
    missing_total_charges = total_charges_numeric.isna().sum()
    print(f"Missing values in 'TotalCharges' (after coercion): {missing_total_charges}")

    # Inspect the non-numeric values if any exist
    if missing_total_charges > 0:
         non_numeric_mask = pd.to_numeric(df['TotalCharges'], errors='coerce').isna()
         print(f"Example of non-numeric 'TotalCharges':")
         print(df[non_numeric_mask]['TotalCharges'].head())
    
    print("-" * 20)

    # 2. Identify Categorical vs Numerical columns
    # This helps in planning the encoding strategy (e.g., One-Hot Encoding for categoricals).
    print("Column Types:")
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    print(f"Numerical Columns ({len(numerical_cols)}):")
    print(numerical_cols)
    print(f"Categorical Columns ({len(categorical_cols)}):")
    print(categorical_cols)

    # 3. Special Check for 'SeniorCitizen'
    # 'SeniorCitizen' is typically 0 or 1, which pandas reads as int, but it is semantically a category.
    if 'SeniorCitizen' in numerical_cols:
         print("Note: 'SeniorCitizen' is numeric but represents a category (0/1).")

    print("-" * 20)
    print("Audit Complete.")

if __name__ == "__main__":
    # Define the path to the raw dataset and I changed the path to run this code and locate the CSV file.
    file_path = "Machine-Learning-Problem-set-1-code/Dataset/Customer churn dataset.csv"
    audit_dataset(file_path)

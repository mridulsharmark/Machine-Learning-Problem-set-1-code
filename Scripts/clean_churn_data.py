import pandas as pd
import numpy as np

"""
Script: clean_churn_data.py
Description:
    This script performs data cleaning and preprocessing on the Customer Churn dataset.
    
    Key Operations:
    1. Removes the non-predictive 'customerID' column.
    2. Converts 'TotalCharges' to numeric and handles missing values via median imputation.
    3. Encodes categorical variables using One-Hot Encoding (drop_first=True) to prepare for ML models.
    4. Saves the processed dataset to a new CSV file.

Usage:
    python3 clean_churn_data.py
"""

def clean_data(input_path, output_path):
    """
    Reads the raw data, applies cleaning/encoding steps, and saves the result.

    Args:
        input_path (str): Path to the raw CSV file.
        output_path (str): Path where the cleaned CSV will be saved.
    """
    print(f"Reading data from {input_path}...")
    df = pd.read_csv(input_path)

    # 1. Drop 'customerID'
    # This column is a unique identifier and has no predictive power for churn.
    if 'customerID' in df.columns:
        print("Dropping 'customerID' column...")
        df.drop(columns=['customerID'], inplace=True)

    # 2. Handle 'TotalCharges' (Convert to Numeric & Impute Missing)
    print("Converting 'TotalCharges' to numeric...")
    # Coerce errors to NaN (handles empty strings)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    missing_count = df['TotalCharges'].isna().sum()
    print(f"Found {missing_count} missing values in TotalCharges.")
    
    # Fill missing values with the median (robust to outliers)
    median_val = df['TotalCharges'].median()
    print(f"Filling missing values with median: {median_val}")
    df['TotalCharges'].fillna(median_val, inplace=True)

    # 3. Encode Categorical Columns
    # We use get_dummies with drop_first=True. This creates k-1 dummy variables for k categories.
    # It avoids the 'dummy variable trap' (multicollinearity) which is important for linear models like Logistic Regression.
    print("Encoding categorical columns...")
    df_cleaned = pd.get_dummies(df, drop_first=True)
    
    # 4. Save the Cleaned Dataset
    print(f"Saving cleaned data to {output_path}...")
    df_cleaned.to_csv(output_path, index=False)
    print("Data cleaning complete.")
    print(f"New shape: {df_cleaned.shape}")

if __name__ == "__main__":
    # Define input and output paths for the code execution error free.
    input_file = "Dataset/Customer churn dataset.csv"
    output_file = "Dataset/customer_churn_cleaned.csv"
    clean_data(input_file, output_file)

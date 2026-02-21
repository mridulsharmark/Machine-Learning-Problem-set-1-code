import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

"""
Script: analyze_coefficients.py
Description:
    This script interprets the trained Logistic Regression model by extracting its coefficients.
    It identifies which features have the strongest positive (churn-increasing) and 
    negative (churn-decreasing/retention) impact on the target variable.
    
    Outputs:
    1. A printed list of top 5 positive and negative influencers.
    2. A horizontal bar chart ('feature_importance_logreg.png') visualizing these key features.

Usage:
    python3 analyze_coefficients.py
"""

def analyze_coefficients():
    """
    Loads the trained model, extracts coefficients, and visualizes feature importance.
    """
    # 1. Load Model and Data
    print("Loading model and data...")
    try:
        # Load the trained Logistic Regression model
        model = joblib.load('logistic_regression.pkl')
        # Load the dataset to retrieve feature names (columns)
        # Note: Must align with the data structure used during training
        df = pd.read_csv('Dataset/customer_churn_cleaned.csv')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 2. Reconstruct Feature Names
    # We need to know which coefficient corresponds to which feature.
    # The target column must be excluded to match the X matrix shape.
    target_col = 'Churn_Yes'
    if target_col not in df.columns:
        print(f"Target '{target_col}' not found. Using last column as target.")
        X_cols = df.columns[:-1]
    else:
        X_cols = df.drop(columns=[target_col]).columns

    # 3. Extract Coefficients
    # Logistic Regression coefficients represent the change in log-odds of churn for a unit change in the feature.
    if hasattr(model, 'coef_'):
        coeffs = model.coef_[0]
    else:
        print("Model does not have coefficients (not a linear model?).")
        return

    # Create a wrapper DataFrame for easier manipulation
    feature_importance = pd.DataFrame({
        'Feature': X_cols,
        'Coefficient': coeffs
    })

    # 4. Identify Top Influencers
    # Sort by coefficient magnitude to find the most impactful features.
    feature_importance = feature_importance.sort_values(by='Coefficient', ascending=False)
    
    top_5_positive = feature_importance.head(5) # Top drivers of Churn
    top_5_negative = feature_importance.tail(5) # Top drivers of Retention
    
    print("\nTop 5 Positive Factors (Increase Churn):")
    print(top_5_positive)
    print("\nTop 5 Negative Factors (Decrease Churn):")
    print(top_5_negative)

    # 5. Visualize Results
    # Combine top positive and negative features for a comprehensive chart
    top_features = pd.concat([top_5_positive, top_5_negative])
    
    # Sort again to ensure the bar chart is ordered logically
    top_features = top_features.sort_values(by='Coefficient', ascending=True)

    plt.figure(figsize=(10, 6))
    # Color code: Red for Churn (Positive), Green for Retention (Negative)
    colors = ['green' if x < 0 else 'red' for x in top_features['Coefficient']]
    
    plt.barh(top_features['Feature'], top_features['Coefficient'], color=colors)
    plt.xlabel('Coefficient Value (Log-Odds)')
    plt.title('Top 10 Features Impacting Churn (Logistic Regression)')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    output_file = 'feature_importance_logreg.png'
    plt.savefig(output_file)
    print(f"\nFeature importance plot saved to {output_file}")

if __name__ == "__main__":
    analyze_coefficients()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib

"""
Script: train_models.py
Description:
    This script trains two machine learning models (Logistic Regression and Random Forest) 
    to predict customer churn.
    
    Steps:
    1. Loads the cleaned dataset.
    2. Identifies the target variable ('Churn_Yes').
    3. Splits the data into Training (80%) and Testing (20%) sets.
    4. Trains both baseline (Logistic Regression) and complex (Random Forest) models.
    5. Evaluates performance using Accuracy, Precision, Recall, and F1-Score.
    6. Saves the trained models (.pkl) and a performance comparison CSV.

Usage:
    python3 train_models.py
"""

def train_and_evaluate():
    """
    Main function to load data, train models, evaluate performance, and save artifacts.
    """
    print("Loading cleaned data...")
    try:
        # Load the preprocessed dataset
        df = pd.read_csv('/Users/mridulsharma/Desktop/Term 2 Assessments/Machine learning/Dataset/customer_churn_cleaned.csv')
    except FileNotFoundError:
        print("Error: Cleaned data file not found.")
        return

    # 1. Identify Target Variable
    # The cleaning process (One-Hot Encoding) converts 'Churn' (Yes/No) into 'Churn_Yes' (1/0).
    target_col = 'Churn_Yes'
    if target_col not in df.columns:
        # Fallback logic in case of unexpected column naming
        print(f"Target column '{target_col}' not found. Checking columns...")
        print(df.columns)
        if 'Churn_1' in df.columns: target_col = 'Churn_1'
        elif 'Churn' in df.columns: target_col = 'Churn'
        else:
             print("Could not identify target column. Aborting.")
             return

    print(f"Using target variable: {target_col}")

    # 2. Define Features (X) and Target (y)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 3. Split Data into Training and Testing Sets
    # 80% for training, 20% for testing. Random_state ensures reproducibility.
    print("Splitting data (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Initialize Models
    # Logistic Regression: Baseline linear model. Max_iter increased to ensure convergence.
    # Random Forest: Complex ensemble model for capturing non-linear relationships.
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42)
    }

    results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        print(f"Evaluating {name}...")
        y_pred = model.predict(X_test)
        
        # Calculate Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1
        })

        # Save the trained model for future use
        filename = name.lower().replace(" ", "_") + ".pkl"
        joblib.dump(model, filename)
        print(f"Saved model to {filename}")

    # 5. Save Results
    results_df = pd.DataFrame(results)
    results_df.to_csv("model_results.csv", index=False)
    print("\nSaved comparison results to 'model_results.csv'")
    print(results_df)

if __name__ == "__main__":
    train_and_evaluate()

# main.py
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib
import pandas as pd
import numpy as np

if __name__ == "__main__":
    # Load the Breast Cancer dataset
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train an XGBoost classifier
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save the trained model
    joblib.dump(model, "breast_cancer_model.pkl")
    print("\nModel saved as 'breast_cancer_model.pkl'")
# File: src/model_development.py
# Modified by: Nikhil Yellapragada - Northeastern University
# Changed ML model from Logistic Regression to Support Vector Machine (SVM)

import os
import pickle
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

WORKING_DIR = "/opt/airflow/working_data"
MODEL_DIR = "/opt/airflow/model"

os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data() -> str:
    """
    Load CSV and persist raw dataframe to a pickle file.
    Returns path to saved file.
    """
    csv_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "advertising.csv",
    )
    df = pd.read_csv(csv_path)
    out_path = os.path.join(WORKING_DIR, "raw.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(df, f)
    return out_path

def data_preprocessing(file_path: str) -> str:
    """
    Load dataframe, split, scale, and save (X_train, X_test, y_train, y_test) to pickle.
    Returns path to saved file.
    """
    with open(file_path, "rb") as f:
        df = pickle.load(f)
    
    X = df.drop(
        ["Timestamp", "Clicked on Ad", "Ad Topic Line", "Country", "City"],
        axis=1,
    )
    y = df["Clicked on Ad"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    num_columns = [
        "Daily Time Spent on Site",
        "Age",
        "Area Income",
        "Daily Internet Usage",
        "Male",
    ]
    
    ct = make_column_transformer(
        (MinMaxScaler(), num_columns),
        (StandardScaler(), num_columns),
        remainder="passthrough",
    )
    
    X_train_tr = ct.fit_transform(X_train)
    X_test_tr = ct.transform(X_test)
    
    out_path = os.path.join(WORKING_DIR, "preprocessed.pkl")
    with open(out_path, "wb") as f:
        pickle.dump((X_train_tr, X_test_tr, y_train.values, y_test.values), f)
    
    return out_path

def separate_data_outputs(file_path: str) -> str:
    """
    Passthrough; kept so the DAG composes cleanly.
    """
    return file_path

def build_model(file_path: str, filename: str) -> str:
    """
    Train SVM model and save to MODEL_DIR/filename. Returns model path.
    Modified by: Nikhil Yellapragada
    Changed from Logistic Regression to Support Vector Machine for improved classification.
    """
    with open(file_path, "rb") as f:
        X_train, X_test, y_train, y_test = pickle.load(f)
    
    # Using Support Vector Machine with RBF kernel instead of Logistic Regression
    # RBF kernel handles non-linear relationships better for advertising data
    model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    model.fit(X_train, y_train)
    
    model_path = os.path.join(MODEL_DIR, filename)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    print(f"SVM model trained and saved to {model_path}")
    return model_path

def load_model(file_path: str, filename: str) -> int:
    """
    Load saved SVM model and test set, print score, and return first prediction as int.
    Modified by: Nikhil Yellapragada
    """
    with open(file_path, "rb") as f:
        X_train, X_test, y_train, y_test = pickle.load(f)
    
    model_path = os.path.join(MODEL_DIR, filename)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    score = model.score(X_test, y_test)
    print(f"SVM Model accuracy on test data: {score:.4f}")
    
    pred = model.predict(X_test)
    print(f"First prediction: {pred[0]}")
    
    return int(pred[0])
import time
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import mlflow.xgboost
import xgboost as xgb
import optuna

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

optuna.logging.set_verbosity(optuna.logging.WARNING)


# Load Dataset

print("Loading CoverType dataset...")
covtype = fetch_covtype()
data = pd.DataFrame(covtype.data, columns=covtype.feature_names)
data['Cover_Type'] = covtype.target.astype(int)
print(f"Shape: {data.shape}")
print(data.head())


# EDA

print("\nMissing values:", data.isna().any().any())

plt.figure(figsize=(8, 5))
sns.countplot(x='Cover_Type', data=data, palette='viridis')
plt.title("Forest Cover Type - Class Distribution")
plt.tight_layout()
plt.savefig("class_distribution.png")
plt.close()


# Train / Val / Test Split

X = data.drop("Cover_Type", axis=1)
y = data["Cover_Type"]

X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.6, random_state=123, stratify=y)
X_val, X_test, y_val, y_test   = train_test_split(X_rem, y_rem, test_size=0.5, random_state=123, stratify=y_rem)

print(f"\nTrain: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")


# Helper: Log Artifacts

def log_artifacts(y_true, y_pred, prefix):
    report = classification_report(y_true, y_pred)
    report_path = f"{prefix}_classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    mlflow.log_artifact(report_path)

    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, colorbar=False)
    ax.set_title(f"{prefix} - Confusion Matrix")
    cm_path = f"{prefix}_confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path)


# Baseline Model: Random Forest

print("\nTraining Random Forest...")

with mlflow.start_run(run_name="random_forest_covtype"):
    rf = RandomForestClassifier(n_estimators=100, random_state=123, n_jobs=-1)
    rf.fit(X_train, y_train)
    preds_rf = rf.predict(X_test)

    acc_rf = accuracy_score(y_test, preds_rf)
    f1_rf  = f1_score(y_test, preds_rf, average='weighted')

    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", acc_rf)
    mlflow.log_metric("f1_weighted", f1_rf)
    log_artifacts(y_test, preds_rf, "random_forest")

    feat_imp = pd.Series(rf.feature_importances_, index=X_train.columns).nlargest(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    feat_imp.sort_values().plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title("Random Forest - Top 15 Feature Importances")
    plt.tight_layout()
    plt.savefig("rf_feature_importance.png")
    mlflow.log_artifact("rf_feature_importance.png")
    plt.close()

    sig = infer_signature(X_train, rf.predict(X_train))
    mlflow.sklearn.log_model(rf, "random_forest_model", signature=sig)
    rf_run_id = mlflow.active_run().info.run_id

print(f"RF — Accuracy: {acc_rf:.4f} | F1: {f1_rf:.4f}")


# Second Model: XGBoost + Optuna

print("\nTuning XGBoost with Optuna (5 trials)...")

y_train_xgb = y_train - 1
y_val_xgb   = y_val - 1
y_test_xgb  = y_test - 1

dtrain = xgb.DMatrix(X_train, label=y_train_xgb)
dval   = xgb.DMatrix(X_val,   label=y_val_xgb)
dtest  = xgb.DMatrix(X_test)

def objective(trial):
    params = {
        'max_depth':        trial.suggest_int('max_depth', 4, 12),
        'learning_rate':    trial.suggest_float('learning_rate', 1e-3, 1.0, log=True),
        'reg_alpha':        trial.suggest_float('reg_alpha', 1e-5, 0.1, log=True),
        'reg_lambda':       trial.suggest_float('reg_lambda', 1e-6, 0.1, log=True),
        'min_child_weight': trial.suggest_float('min_child_weight', 0.5, 20, log=True),
        'objective':        'multi:softmax',
        'num_class':        7,
        'eval_metric':      'mlogloss',
        'seed':             123,
    }
    with mlflow.start_run(run_name=f"xgb_trial_{trial.number}", nested=True):
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=50,
            evals=[(dval, "validation")],
            early_stopping_rounds=10,
            verbose_eval=False
        )
        preds_val = booster.predict(dval).astype(int)
        f1 = f1_score(y_val_xgb, preds_val, average='weighted')

        mlflow.log_params(params)
        mlflow.log_metric("f1_weighted", f1)
        sig = infer_signature(X_train, booster.predict(dtrain))
        mlflow.xgboost.log_model(booster, "model", signature=sig)

    return f1

with mlflow.start_run(run_name="xgboost_covtype"):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5)
    xgb_parent_run_id = mlflow.active_run().info.run_id

    best_trial = study.best_trial
    print(f"Best trial F1: {best_trial.value:.4f}")
    print(f"Best params: {best_trial.params}")


# Evaluate Best XGBoost on Test Set

best_xgb_run = mlflow.search_runs(
    filter_string='tags.mlflow.runName LIKE "xgb_trial_%"',
    order_by=["metrics.f1_weighted DESC"]
).iloc[0]
best_xgb_run_id = best_xgb_run.run_id

loaded_xgb = mlflow.xgboost.load_model(f"runs:/{best_xgb_run_id}/model")
preds_xgb  = loaded_xgb.predict(dtest).astype(int)
acc_xgb    = accuracy_score(y_test_xgb, preds_xgb)
f1_xgb     = f1_score(y_test_xgb, preds_xgb, average='weighted')

with mlflow.start_run(run_id=best_xgb_run_id):
    mlflow.log_metric("test_accuracy", acc_xgb)
    mlflow.log_metric("test_f1_weighted", f1_xgb)
    log_artifacts(y_test_xgb, preds_xgb, "xgboost")

print(f"XGBoost — Accuracy: {acc_xgb:.4f} | F1: {f1_xgb:.4f}")


# Register the Better Model

model_name = "cover_type_classifier"

if f1_rf >= f1_xgb:
    print(f"\nRandom Forest wins (F1: {f1_rf:.4f}). Registering RF.")
    model_version = mlflow.register_model(f"runs:/{rf_run_id}/random_forest_model", model_name)
else:
    print(f"\nXGBoost wins (F1: {f1_xgb:.4f}). Registering XGBoost.")
    model_version = mlflow.register_model(f"runs:/{best_xgb_run_id}/model", model_name)

time.sleep(15)


# Transition to Production

client = MlflowClient()
client.transition_model_version_stage(
    name=model_name,
    version=model_version.version,
    stage="Production"
)
print(f"Model '{model_name}' v{model_version.version} is now in Production")


# Load Production Model & Evaluate

prod_model = mlflow.pyfunc.load_model(f"models:/{model_name}/production")
prod_preds = prod_model.predict(X_test)

prod_preds_arr = np.array(prod_preds).astype(int)
if prod_preds_arr.min() == 0:
    final_f1 = f1_score(y_test_xgb, prod_preds_arr, average='weighted')
else:
    final_f1 = f1_score(y_test, prod_preds_arr, average='weighted')

print(f"Production model F1 (weighted): {final_f1:.4f}")


# Batch Inference

print("\nRunning batch inference on 500 samples...")
batch = X_test.sample(n=500, random_state=42).reset_index(drop=True)
batch_preds = prod_model.predict(batch)
batch['predicted_cover_type'] = batch_preds
print(batch['predicted_cover_type'].value_counts())
batch.to_csv("batch_inference_results.csv", index=False)
print("Saved batch_inference_results.csv")


# Real-Time Inference
# Start the server first in a separate terminal:
# mlflow models serve --env-manager=local -m models:/cover_type_classifier/production -h 0.0.0.0 -p 5001

url = 'http://localhost:5001/invocations'
payload = {"dataframe_split": X_test.head(5).to_dict(orient='split')}

try:
    response = requests.post(url, json=payload, timeout=5)
    print("Real-time predictions:", response.json())
except requests.exceptions.ConnectionError:
    print("\nModel server not running. Start it with:")
    print("  mlflow models serve --env-manager=local -m models:/cover_type_classifier/production -h 0.0.0.0 -p 5001")
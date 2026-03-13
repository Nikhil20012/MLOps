# Lab 2: MLflow Experiment Tracking with Forest Cover Type Classification

**Author:** Nikhil Yellapragada
**Student ID:** 002567331
**Course:** Data Analytics Engineering
**University:** Northeastern University
**Semester:** Spring 2026
**GitHub:** https://github.com/Nikhil20012/MLOps

---

## Project Overview

This lab demonstrates the end-to-end MLflow experiment tracking lifecycle — from data loading and model training to model registration, batch inference, and real-time serving. In this updated version, significant improvements and changes were made compared to the original lab.

---

## What This Does

The system loads the Forest Cover Type dataset, trains a Random Forest baseline and a tuned XGBoost model tracked via MLflow, evaluates both using accuracy and weighted F1 score, registers the better model to the MLflow Model Registry, runs batch inference on new data, and serves the model as a REST API for real-time predictions.

---

## Technologies Used

- Python 3.13
- MLflow
- scikit-learn
- XGBoost
- Optuna
- pandas
- numpy
- seaborn
- matplotlib

---

## My Modifications

### Dataset Update
- Replaced the original Wine Quality CSV dataset with the **Forest Cover Type dataset** from `scikit-learn` (`fetch_covtype`) — a large-scale dataset with 581,012 samples, 54 features, and 7 cover type classes.

### Task Update
- Replaced binary classification with **multi-class classification** (7 forest cover types).

### Metrics Update
- Replaced AUC with **Accuracy** and **Weighted F1 Score** as evaluation metrics, more appropriate for multi-class classification.

### Model Update
- Kept Random Forest as the baseline model.
- Replaced XGBoost with manual hyperparameter tuning using **Optuna** for automated hyperparameter optimization across 5 trials.

### Artifact Logging
- Added **classification report** (text file) and **confusion matrix** (PNG) logged as MLflow artifacts for both models.
- Added **feature importance plot** (top 15 features) logged as an MLflow artifact for the Random Forest model.

### Model Registration
- Automatically registers the **better performing model** (by F1 score) to the MLflow Model Registry and promotes it to Production.

### Inference
- Replaced Spark-based batch inference with **local pandas batch inference**.
- Added real-time serving via MLflow's REST API.

### Dependencies Update
- Updated `requirements.txt` to remove `pyspark` and add `optuna`, `cloudpickle`, and `requests`.

---

## Folder Structure

```
Lab2/
├── cover_type_mlflow.py        
├── starter.ipynb               
├── requirements.txt            
├── README.md                   
└── data/
    ├── winequality-red.csv
    ├── winequality-white.csv
    └── winequality.names
```

| File | Description |
|---|---|
| `cover_type_mlflow.py` | Main script — training, tracking, registration, inference |
| `starter.ipynb` | Original lab notebook (unchanged) |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

---

## Setup Instructions

### Step 1: Clone the Repository

```bash
git clone https://github.com/Nikhil20012/MLOps.git
cd MLOps/Labs/Experiment_Tracking_Labs/Mlflow_Labs/Lab2
```

### Step 2: Create and Activate a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Script

```bash
python cover_type_mlflow.py
```

This will:
- Load the Forest Cover Type dataset (581,012 samples)
- Train a Random Forest classifier and log it to MLflow
- Tune an XGBoost model using Optuna (5 trials) and log each trial to MLflow
- Register the better model to the MLflow Model Registry
- Promote the registered model to Production
- Run batch inference on 500 samples and save results to `batch_inference_results.csv`

### Step 5: View Runs in MLflow UI

In a separate terminal:

```bash
source .venv/bin/activate
mlflow ui
```

Open `http://127.0.0.1:5000` in your browser. Navigate to the **Default** experiment under **Training Runs** to see all logged runs, metrics, and artifacts.

### Step 6: Serve the Model for Real-Time Inference

In a separate terminal:

```bash
source .venv/bin/activate
mlflow models serve --env-manager=local -m models:/cover_type_classifier/production -h 0.0.0.0 -p 5001
```

### Step 7: Test Real-Time Inference

```bash
python -c "
import requests, pandas as pd
from sklearn.datasets import fetch_covtype
covtype = fetch_covtype()
X = pd.DataFrame(covtype.data[:5], columns=covtype.feature_names)
payload = {'dataframe_split': X.to_dict(orient='split')}
r = requests.post('http://localhost:5001/invocations', json=payload)
print(r.json())
"
```

---

## How the Pipeline Works

1. Load the Forest Cover Type dataset from `scikit-learn`
2. Perform EDA — check for missing values, plot class distribution
3. Split data into train (60%), validation (20%), and test (20%) sets using stratified sampling
4. Train Random Forest (100 estimators) and log metrics and artifacts to MLflow
5. Tune XGBoost using Optuna with 5 trials, logging each trial as a nested MLflow run
6. Evaluate the best XGBoost model on the test set
7. Compare RF vs XGBoost by F1 score and register the better model
8. Transition the registered model to Production in the MLflow Model Registry
9. Load the production model and run batch inference on 500 samples
10. Serve the model as a REST API and test with real-time predictions

---

## Results

| Model | Accuracy | F1 (Weighted) |
|---|---|---|
| Random Forest | 94.90% | 94.87% |
| XGBoost (Optuna) | 96.16% | 96.15% |

XGBoost won and was registered as the production model.

---

## Troubleshooting

**Script runs slowly**
- The dataset has 581k rows. Random Forest takes ~2-3 mins, XGBoost tuning takes ~5-8 mins. This is expected.

**MLflow UI shows no runs**
- Make sure you run `mlflow ui` from the same `Lab2/` directory where the script was run.

**Model server not starting**
- Ensure the script has completed successfully and the model is registered before running the serve command.

**Port 5001 already in use**
- Change `-p 5001` to `-p 5002` in both the serve command and the inference test.

---

## Files Modified

- `cover_type_mlflow.py` — Written from scratch with CoverType dataset, RF + XGBoost + Optuna, MLflow tracking, artifact logging, batch and real-time inference
- `requirements.txt` — Removed `pyspark`, added `optuna`, `cloudpickle`, `requests`
- `README.md` — Written from scratch with comprehensive documentation
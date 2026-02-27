# Lab 1: Containerized ML Application using Docker

This lab demonstrates how to containerize a Python machine learning application using **Docker**.

In this updated version, significant improvements and changes were made compared to the original lab.

---

## Objective

1. Containerize a Python ML application so it can run consistently across any environment.
2. Train a machine learning model in an isolated environment and save the trained model.
3. Demonstrate the process of updating Docker images, running containers, and managing dependencies.

---

## Changes Made

| Area | Update |
|---|---|
| **Dataset** | Replaced the Iris dataset with the **Breast Cancer dataset** from `scikit-learn` |
| **Model** | Replaced Random Forest with **XGBoost** for improved performance on structured data |
| **Dependencies** | Updated `requirements.txt` to include `xgboost`, `scikit-learn`, `pandas`, `numpy` |
| **Dockerfile** | Used `python:3.10-slim` as base image; optimized layer caching |
| **Code** | Added Accuracy, Classification Report, and Confusion Matrix; saves model as `breast_cancer_model.pkl` |

---

## Folder Structure
```
Lab1/
├── Dockerfile
├── README.md
└── src/
    ├── main.py
    └── requirements.txt
```

| File | Description |
|---|---|
| `Dockerfile` | Instructions to build the Docker image |
| `src/requirements.txt` | Python dependencies |
| `src/main.py` | Script to train the ML model |
| `README.md` | This file |

---

## How to Run

### 1. Build the Docker Image

From the `Lab1/` directory:
```bash
docker build -t lab1:v2 .
```

This creates a Docker image named `lab1:v2` containing:
- Python 3.10
- All required dependencies (`xgboost`, `scikit-learn`, `pandas`, `numpy`)
- The ML training script (`main.py`)

---

### 2. Run the Container
```bash
docker run --name lab1-container lab1:v2
```

This will:
- Start a container from the `lab1:v2` image
- Execute `main.py` automatically
- Train the XGBoost model on the Breast Cancer dataset
- Print evaluation metrics (accuracy, classification report, confusion matrix)
- Save the trained model as `breast_cancer_model.pkl` inside the container

---

### 3. View Container Logs
```bash
docker logs lab1-container
```

Displays all printed outputs from the script, including model evaluation results.

---

### 4. Access the Container Shell *(Optional)*
```bash
docker exec -it lab1-container /bin/bash
```

Once inside, verify the saved model exists:
```bash
ls
```

You should see `breast_cancer_model.pkl` in the output.

---

### 5. Save the Docker Image *(Optional)*
```bash
docker save lab1:v2 > lab1_image.tar
```

Creates a portable `.tar` file of the image that can be shared or loaded on another machine.
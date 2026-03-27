# Lab 1: Containerized ML Application using Docker

**Author:** Nikhil Yellapragada
**Student ID:** 002567331
**Course:** Data Analytics Engineering
**University:** Northeastern University
**Semester:** Spring 2026
**GitHub:** https://github.com/Nikhil20012/MLOps

---

## Project Overview

This lab demonstrates how to containerize a Python machine learning application using **Docker**. In this updated version, significant improvements and changes were made compared to the original lab.

---

## What This Does

The system loads the Breast Cancer dataset, trains an XGBoost classification model inside a Docker container, evaluates it using accuracy, classification report, and confusion matrix, and saves the trained model as a `.pkl` file.

---

## Technologies Used

- Docker
- Python 3.10
- XGBoost
- scikit-learn
- pandas
- numpy

---

## My Modifications

**Dataset Update**
- Replaced the original Iris dataset with the **Breast Cancer dataset** from `scikit-learn` to make the lab more realistic and challenging.

**Model Update**
- Replaced the Random Forest classifier with **XGBoost**, a powerful model for structured data.

**Dependencies Update**
- Updated `requirements.txt` to include `xgboost`, `scikit-learn`, `pandas`, and `numpy`.

**Dockerfile Update**
- Used `python:3.10-slim` as the base image.
- Optimized caching by copying `requirements.txt` first and installing dependencies before copying source code.
- Ensured the container runs `main.py` automatically.

**Code Improvements**
- Added evaluation metrics: Accuracy, Classification Report, Confusion Matrix.
- Saves the trained model as `breast_cancer_model.pkl` inside the container.

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

## Setup Instructions

### Step 1: Clone the Repository
```bash
git clone https://github.com/Nikhil20012/MLOps.git
cd MLOps/Labs/Docker_Labs/Lab_1
```

### Step 2: Build the Docker Image

From the `Lab1/` directory:
```bash
docker build -t lab1:v2 .
```

This creates a Docker image named `lab1:v2` containing:
- Python 3.10
- All required dependencies (`xgboost`, `scikit-learn`, `pandas`, `numpy`)
- The ML training script (`main.py`)

### Step 3: Run the Container
```bash
docker run --name lab1-container lab1:v2
```

This will:
- Start a container from the `lab1:v2` image
- Execute `main.py` automatically
- Train the XGBoost model on the Breast Cancer dataset
- Print evaluation metrics (accuracy, classification report, confusion matrix)
- Save the trained model as `breast_cancer_model.pkl` inside the container

### Step 4: View Container Logs
```bash
docker logs lab1-container
```

Displays all printed outputs from the script, including model evaluation results.

### Step 5: Access the Container Shell *(Optional)*
```bash
docker exec -it lab1-container /bin/bash
```

Once inside, verify the saved model exists:
```bash
ls
```

You should see `breast_cancer_model.pkl` in the output.

### Step 6: Save the Docker Image *(Optional)*
```bash
docker save lab1:v2 > lab1_image.tar
```

Creates a portable `.tar` file of the image that can be shared or loaded on another machine.

---

## How the Pipeline Works

The container runs the following steps in order:

1. Load the Breast Cancer dataset from `scikit-learn`
2. Preprocess and split data into training and test sets
3. Train an XGBoost classifier
4. Evaluate the model (accuracy, classification report, confusion matrix)
5. Save the trained model as `breast_cancer_model.pkl`

---

## Troubleshooting

**Docker image not building**
- Ensure Docker Desktop is running and you are in the `Lab1/` directory when running the build command.

**Container exits immediately**
- Run `docker logs lab1-container` to check for errors in the script.

**Model file not found inside container**
- Make sure the container ran successfully before checking for the `.pkl` file.

---

## Stopping the Lab

To remove the container after use:
```bash
docker rm lab1-container
```

---

## Files Modified

- `main.py` — Changed dataset to Breast Cancer, changed model to XGBoost, added evaluation metrics
- `src/requirements.txt` — Updated dependencies
- `Dockerfile` — Updated base image and optimized layer caching
- `README.md` — Written from scratch with comprehensive documentation
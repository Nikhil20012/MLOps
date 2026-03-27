# Lab 2: Wine Classifier REST API deployed on Google Cloud Run

**Author:** Nikhil Yellapragada
**Student ID:** 002567331
**Course:** Data Analytics Engineering
**University:** Northeastern University
**Semester:** Spring 2026
**GitHub:** https://github.com/Nikhil20012/MLOps

---

## 🌐 Live Demo
https://wine-classifier-269900465405.us-central1.run.app

---

## Project Overview

This lab demonstrates how to build and deploy a containerized Machine Learning REST API on **Google Cloud Run**. In this updated version, significant improvements and changes were made compared to the original lab, which only deployed a simple "Hello World" Flask app.

---

## What This Does

The system loads the Wine dataset from scikit-learn, trains a Random Forest classification model on startup, and exposes a REST API with multiple endpoints to inspect the model and make real-time predictions. The app is deployed serverlessly on Google Cloud Run and is publicly accessible via a URL.

---

## Technologies Used

- Google Cloud Run
- Docker
- Python 3.9
- Flask
- scikit-learn
- numpy

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | App info and model accuracy |
| GET | `/health` | Health check |
| GET | `/features` | Lists all 13 required input features |
| POST | `/predict` | Predicts wine class from feature values |

---

## My Modifications

### Application Update
- Replaced the original "Hello World" single-route Flask app with a **fully functional ML REST API** with 4 meaningful endpoints.

### Dataset
- Used the **Wine Dataset** from scikit-learn (178 samples, 13 features, 3 wine classes) — a more realistic and challenging dataset than a basic placeholder.

### Model
- Trained a **Random Forest Classifier** with `StandardScaler` preprocessing directly inside the container on startup — no manual training step required.

### API Design
- Added `/health`, `/features`, and `/predict` endpoints with structured JSON responses including confidence scores and class probabilities.

### Dependencies Update
- Added a `requirements.txt` with pinned versions of `flask`, `scikit-learn`, and `numpy` instead of ad-hoc pip installs.

### Dockerfile Update
- Used `python:3.9-slim` as the base image.
- Optimized layer caching by copying `requirements.txt` first and installing dependencies before copying source code.
- Fixed platform compatibility issue for Apple Silicon Macs by building with `--platform linux/amd64`.

---

## Folder Structure

```
Begineer_Lab/
├── app.py              # Flask ML API application
├── Dockerfile          # Container configuration
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

| File | Description |
|------|-------------|
| `app.py` | Flask app with ML model training and prediction endpoints |
| `Dockerfile` | Instructions to build the Docker image |
| `requirements.txt` | Python dependencies with pinned versions |
| `README.md` | This file |

---

## Setup Instructions

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop) installed and running
- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) (`gcloud`) installed
- A Google Cloud project with **Cloud Run API** and **Container Registry API** enabled
- Billing enabled on your GCP project

---

### Step 1: Clone the Repository

```bash
git clone https://github.com/Nikhil20012/MLOps.git
cd MLOps/Labs/Begineer_Lab
```

---

### Step 2: Run Locally (Optional)

```bash
pip install -r requirements.txt
python app.py
```

Visit `http://localhost:8080` to test locally.

---

### Step 3: Set Up GCP Project

```bash
# Login
gcloud auth login

# Create project
gcloud projects create wine-classifier-lab --name="Wine Classifier Lab"

# Set active project
gcloud config set project wine-classifier-lab

# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

> Make sure billing is enabled on your project at https://console.cloud.google.com/billing

---

### Step 4: Build the Docker Image

> **Note for Mac Apple Silicon (M1/M2) users:** Use the `--platform linux/amd64` flag to ensure compatibility with Cloud Run.

```bash
docker build --platform linux/amd64 -t gcr.io/wine-classifier-lab/wine-classifier .
```

---

### Step 5: Authenticate & Push to Container Registry

```bash
gcloud auth configure-docker
docker push gcr.io/wine-classifier-lab/wine-classifier
```

---

### Step 6: Deploy to Cloud Run

```bash
gcloud run deploy wine-classifier \
  --image gcr.io/wine-classifier-lab/wine-classifier \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

Once deployed, Cloud Run will print a **Service URL**.

---

### Step 7: Test the Deployed App

```bash
# Check app info
curl https://YOUR-CLOUD-RUN-URL/

# Check health
curl https://YOUR-CLOUD-RUN-URL/health

# List input features
curl https://YOUR-CLOUD-RUN-URL/features

# Make a prediction
curl -X POST https://YOUR-CLOUD-RUN-URL/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [13.2, 2.77, 2.51, 18.5, 96.6, 1.04, 2.55, 0.57, 1.47, 6.2, 1.05, 3.33, 820]}'
```

**Example Prediction Response:**
```json
{
  "predicted_class": 0,
  "wine_type": "class_0",
  "confidence": "73.00%",
  "class_probabilities": {
    "class_0": "73.00%",
    "class_1": "15.00%",
    "class_2": "12.00%"
  }
}
```

---

## How the Pipeline Works

The container runs the following steps in order:

1. Load the Wine dataset from `scikit-learn`
2. Split data into training and test sets (80/20)
3. Apply `StandardScaler` preprocessing
4. Train a `RandomForestClassifier`
5. Start the Flask server and expose prediction endpoints
6. Accept POST requests at `/predict` and return wine class predictions

---

## Model Details

| Property | Value |
|----------|-------|
| Dataset | sklearn Wine Dataset |
| Algorithm | Random Forest Classifier |
| Train/Test Split | 80% / 20% |
| Features | 13 chemical properties |
| Classes | 3 wine types (class_0, class_1, class_2) |
| Preprocessing | StandardScaler |
| Model Accuracy | 100% on test set |

---

## Troubleshooting

**Docker image not building**
- Ensure Docker Desktop is running before running the build command.

**Cloud Run deployment fails with architecture error**
- Rebuild using `--platform linux/amd64` flag (required for Apple Silicon Macs).

**Container fails to start on Cloud Run**
- Make sure `requirements.txt` includes all dependencies (`flask`, `scikit-learn`, `numpy`). Installing only `flask` causes startup crashes since `sklearn` imports will fail.

**Billing error when enabling APIs**
- Enable billing at https://console.cloud.google.com/billing and link your project before running `gcloud services enable`.

---

## Files Modified

- `app.py` — Built from scratch with Wine dataset, Random Forest model, and 4 REST API endpoints
- `requirements.txt` — Added `scikit-learn` and `numpy` with pinned versions
- `Dockerfile` — Updated base image to `python:3.9-slim`, optimized layer caching
- `README.md` — Written from scratch with comprehensive documentation

---

## Resources

- [Google Cloud Run Docs](https://cloud.google.com/run/docs)
- [scikit-learn Wine Dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-recognition-dataset)
- [Flask Documentation](https://flask.palletsprojects.com/)
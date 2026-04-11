# ELK Stack Lab - Ridge Regression on California Housing Dataset

**Author:** Nikhil Yellapragada  
**Student ID:** 002567331  
**Course:** Data Analytics Engineering  
**University:** Northeastern University  
**Semester:** Spring 2026  
**GitHub:** https://github.com/Nikhil20012/MLOps

---

## Project Overview

This project sets up the ELK (Elasticsearch, Logstash, Kibana) stack to manage and visualize logs generated during machine learning model training. It trains a Ridge Regression model on the California Housing dataset, logs key metrics in structured JSON format, ships the logs through Logstash into Elasticsearch, and visualizes them in Kibana.

---

## What This Does

The system loads the California Housing dataset, scales the features, trains a Ridge Regression model with L2 regularization, logs training metrics in JSON format, pipes the logs through Logstash into Elasticsearch, and visualizes all log events in Kibana's Discover view.

---

## Technologies Used

- Elasticsearch 7.17.4
- Kibana 7.17.4
- Logstash 7.17.4
- Python 3.x
- scikit-learn (Ridge Regression)
- numpy
- Homebrew (macOS)

---

## My Modifications

### Dataset
Changed from the original Iris dataset to the **California Housing dataset** from sklearn. This is a real-world regression dataset with 20,640 samples and 8 features predicting median house values across California districts.

### Machine Learning Model
Changed from **Logistic Regression** (classification) to **Ridge Regression** (regression with L2 regularization). Ridge adds a regularization term controlled by alpha to prevent overfitting on continuous target prediction tasks.

### Logging Format
Replaced the plain-text `basicConfig` logger with a **custom JSON formatter**. Each log line is a structured JSON object with `timestamp`, `level`, and `message` fields. This eliminates the need for grok filters in Logstash entirely.

### Metrics Logged
Added regression-specific metrics beyond what the original lab tracked:
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R2 Score
- Model coefficients and intercept
- Alpha (regularization strength)
- Feature names

### Logstash Configuration
- Removed all grok filters (not needed with JSON logging)
- Changed input to `stdin` with `json` codec (file input plugin unsupported on Apple Silicon with x86 Logstash build)
- Changed Elasticsearch index from `logstash-training` to `ridge-training-logs`

---

## Results

| Metric | Value |
|--------|-------|
| Training Samples | 16,512 |
| Testing Samples | 4,128 |
| MSE | 0.5559 |
| RMSE | 0.7456 |
| R2 Score | 0.5758 |
| Alpha | 1.0 |

---

## Prerequisites

- Java 17 (via Homebrew `openjdk@17`)
- Python 3.x with `scikit-learn` and `numpy`
- Elasticsearch 7.17.4 (via Homebrew elastic tap)
- Kibana 7.17.4 (via Homebrew elastic tap)
- Logstash 7.17.4 (manual tar install)

---

## Setup Instructions

### Step 1: Install Elasticsearch and Kibana

```bash
brew tap elastic/tap
brew install elastic/tap/elasticsearch-full
brew install elastic/tap/kibana-full
```

### Step 2: Configure Elasticsearch

```bash
nano /opt/homebrew/etc/elasticsearch/elasticsearch.yml
```

Add at the bottom:

```yaml
xpack.ml.enabled: false
xpack.security.enabled: false
```

### Step 3: Install Logstash (manual install, Homebrew formula is broken on Apple Silicon)

```bash
curl -O https://artifacts.elastic.co/downloads/logstash/logstash-7.17.4-darwin-x86_64.tar.gz
tar -xzf logstash-7.17.4-darwin-x86_64.tar.gz
mv logstash-7.17.4 /opt/homebrew/opt/logstash-full
```

### Step 4: Start Elasticsearch (Tab 1)

```bash
/opt/homebrew/opt/elasticsearch-full/bin/elasticsearch
```

Verify at http://localhost:9200

### Step 5: Start Kibana (Tab 2)

```bash
/opt/homebrew/opt/kibana-full/bin/kibana
```

Verify at http://localhost:5601

### Step 6: Train the Model

```bash
python train_model.py
```

### Step 7: Send Logs to Elasticsearch via Logstash

```bash
cat training.log | LS_JAVA_HOME=/opt/homebrew/opt/openjdk@17 \
  /opt/homebrew/opt/logstash-full/bin/logstash -f logstash.conf
```

### Step 8: Visualize in Kibana

1. Go to http://localhost:5601
2. **Stack Management** -> **Index Patterns** -> **Create index pattern**
3. Enter `ridge-training-logs*` -> Next -> select `@timestamp` -> Create
4. Go to **Discover** -> select `ridge-training-logs*`
5. Set time range to **Last 1 year**

---

## Project Structure

```
Lab2_ELK_Setup_Mac/
├── train_model.py        # Ridge Regression training with JSON logging
├── logstash.conf         # Logstash pipeline config (stdin to ES)
├── training.log          # Generated JSON log output
└── README.md             # This file
```

---

## How the Pipeline Works

1. `train_model.py` trains Ridge Regression on California Housing data
2. Structured JSON logs are written to `training.log`
3. Logstash reads logs via stdin, renames fields, and ships to Elasticsearch
4. Kibana visualizes all log events under the `ridge-training-logs` index

---

## Troubleshooting

**Elasticsearch fails to start**  
Check `elasticsearch.yml` and remove `xpack.security.enrollment.enabled` if present. That setting only exists in v8.x and will cause a startup error on 7.17.4.

**Logstash file input crashes on Apple Silicon**  
The x86 Logstash build has JRuby native library issues on macOS ARM. Use stdin with a pipe instead: `cat training.log | logstash -f logstash.conf`

**No data in Kibana**  
Set the time filter to Last 1 year. Logs are ingested with the current timestamp, not the original log timestamp, so the default 15-minute window will show nothing.

---

## Files Modified from Original Lab

| File | Change |
|------|--------|
| `train_model.py` | Dataset (Iris to California Housing), model (Logistic to Ridge), logging (plaintext to JSON), metrics (accuracy to MSE/RMSE/R2) |
| `logstash.conf` | Removed grok filters, changed input to stdin, changed index name |
| `README.md` | Written from scratch with full documentation |
# Airflow Lab 2 - ML Pipeline with Automated Notifications

**Author:** Nikhil Yellapragada  
**Student ID:** 002567331  
**Course:** Data Analytics Engineering  
**University:** Northeastern University  
**Semester:** Spring 2026  
**GitHub:** https://github.com/Nikhil20012/MLOps

---

## Project Overview

This project uses Apache Airflow to automate a machine learning pipeline. It processes advertising data, trains a predictive model, sends email notifications when done, and provides a web API to check the status. Everything runs in Docker containers.

### What This Does

The system loads advertising data, cleans it, trains a machine learning model, saves the model, and then notifies you via email. You can also check the pipeline status through a web interface.

---

## Technologies Used

- Apache Airflow 2.9.2
- Docker and Docker Compose
- Python 3.12
- Flask
- scikit-learn
- pandas
- Gmail SMTP

---

## My Modifications

**Email Notifications**
- Changed the email subject to include my name
- Created custom HTML email content with my student information and course details
- Configured it to send to my Gmail account

**Success Page**
- Redesigned the success.html page with a purple gradient background
- Added a panel showing my name, student ID, course, university, and lab information
- Improved the styling with modern CSS including shadows and rounded corners

**Pipeline Configuration**
- Updated the owner field to "Nikhil Yellapragada - Northeastern University"
- Changed the retry setting from 0 to 2 retries with 5-minute delays between attempts
- Modified the start date to February 13, 2024
- Updated the description to include my name
- Added custom tags: mlops, machine-learning, nikhil-yellapragada
- Changed the owner link to point to my GitHub repository

**Flask API Improvements**
- Added a new /student-info endpoint that returns my project information in JSON format
- Updated the Flask DAG owner information
- Added better logging messages when the Flask server starts
- Included personalized tags and descriptions

**Infrastructure Changes**
- Modified docker-compose.yaml to expose port 5555 so the Flask API is accessible from my browser
- Set up SMTP connection in Airflow for Gmail
- Made both DAGs use consistent dates and settings

---

## Prerequisites

- Docker Desktop with at least 4GB of memory allocated
- Git
- Gmail account with 2-Step Verification enabled
- Basic knowledge of Apache Airflow

---

## Setup Instructions

**Step 1: Clone the Repository**
```bash
git clone https://github.com/Nikhil20012/MLOps.git
cd MLOps/Labs/Airflow_Labs/Lab_2
```

**Step 2: Get a Gmail App Password**

Go to https://myaccount.google.com/apppasswords and create a new app password. Name it "Airflow" and copy the 16-character password. Remove any spaces from it and save it for later.

**Step 3: Download Docker Configuration**
```bash
curl -LfO 'https://airflow.apache.org/docs/apache-airflow/2.9.2/docker-compose.yaml'
```

**Step 4: Create Folders**
```bash
mkdir -p logs plugins working_data
```

**Step 5: Set Up Environment File**

For Mac/Linux:
```bash
echo "AIRFLOW_UID=$(id -u)" > .env
```

For Windows, create a file called .env and put this in it:
```
AIRFLOW_UID=50000
```

**Step 6: Edit docker-compose.yaml**

Open the docker-compose.yaml file and make these changes:

Change LOAD_EXAMPLES to false:
```yaml
AIRFLOW__CORE__LOAD_EXAMPLES: 'false'
```

Add Python packages:
```yaml
_PIP_ADDITIONAL_REQUIREMENTS: ${_PIP_ADDITIONAL_REQUIREMENTS:- flask scikit-learn scipy pandas numpy}
```

In the volumes section, add:
```yaml
- ${AIRFLOW_PROJ_DIR:-.}/working_data:/opt/airflow/working_data
```

Update admin credentials:
```yaml
_AIRFLOW_WWW_USER_USERNAME: ${_AIRFLOW_WWW_USER_USERNAME:-airflow2}
_AIRFLOW_WWW_USER_PASSWORD: ${_AIRFLOW_WWW_USER_PASSWORD:-airflow2}
```

In the airflow-worker section, add:
```yaml
airflow-worker:
  command: celery worker
  ports:
    - "5555:5555"
```

**Step 7: Initialize Airflow**
```bash
docker compose up airflow-init
```

Wait until you see "exited with code 0". This takes 2-3 minutes.

**Step 8: Start Airflow**
```bash
docker compose up
```

Wait until you see a message about "GET /health HTTP/1.1" 200. Keep this terminal window open.

**Step 9: Open Airflow in Your Browser**

Go to http://localhost:8080 and log in with:
- Username: airflow2
- Password: airflow2

**Step 10: Set Up Email Connection**

In Airflow, click Admin, then Connections. Click the plus button and fill in:

- Connection Id: smtp_default
- Connection Type: SMTP
- Host: smtp.gmail.com
- Login: your email address
- Password: your 16-character app password without spaces
- Port: 587
- From email: your email address
- Disable TLS: Leave unchecked
- Disable SSL: Check this box

Click Save.

**Step 11: Run the Pipeline**

Find the Airflow_Lab2 DAG in the list. Toggle the switch to turn it on. Click the play button and select "Trigger DAG". Watch the tasks turn green in the Graph view. This takes 2-3 minutes. Check your email for the completion notification.

**Step 12: Check the Flask API**

Once the pipeline finishes, wait about 30 seconds, then open:

- http://localhost:5555/success
- http://localhost:5555/student-info
- http://localhost:5555/health

---

## Project Structure
```
Lab_2/
├── dags/
│   ├── main.py
│   ├── Flask_API.py
│   ├── data/
│   │   └── advertising.csv
│   ├── src/
│   │   └── model_development.py
│   └── templates/
│       ├── success.html
│       └── failure.html
├── docker-compose.yaml
├── .env
└── README.md
```

---

## How the Pipeline Works

The main pipeline runs these tasks in order:

1. task_using_linked_owner
2. load_data_task
3. data_preprocessing_task
4. separate_data_outputs_task
5. build_save_model_task
6. load_model_task
7. my_trigger_task
8. send_email

The pipeline retries failed tasks twice with 5-minute waits between attempts.

---

## Flask API Endpoints

**/success** - Custom page with purple gradient background and student information

**/failure** - Failure page if something went wrong

**/student-info** - JSON data with project information

**/health** - Health check endpoint

---

## Troubleshooting

**DAGs not showing up**

Wait 30-60 seconds for Airflow to scan the folder.

**Email not sending**

Make sure your app password has no spaces. Verify 2-Step Verification is enabled. Check SMTP settings use port 587.

**Cannot access Flask API**

Check that port 5555 is in docker-compose.yaml under airflow-worker. Make sure Airflow_Lab2_Flask DAG is running.

**Docker running out of memory**

Increase Docker Desktop memory to at least 4GB in settings.

---

## Stopping the Lab

Press Ctrl+C in the terminal, then run:
```bash
docker compose down
```

---

## Files Modified

- main.py - Owner, retries, email content, tags, description
- Flask_API.py - Custom endpoint, owner, tags
- success.html - Complete redesign with styling and information
- docker-compose.yaml - Port 5555, credentials
- README.md - Written from scratch

---

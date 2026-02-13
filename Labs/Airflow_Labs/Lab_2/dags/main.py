# File: main.py
from __future__ import annotations

import pendulum
from datetime import timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.providers.smtp.operators.smtp import EmailOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.trigger_rule import TriggerRule

from src.model_development import (
    load_data,
    data_preprocessing,
    separate_data_outputs,
    build_model,
    load_model,
)

# ---------- Default args ----------
default_args = {
    "owner": "Nikhil Yellapragada - Northeastern University",
    "start_date": pendulum.datetime(2024, 2, 13, tz="UTC"),
    "retries": 2,  # Changed from 0 to 2 for better reliability
    "retry_delay": timedelta(minutes=5),
}

# ---------- DAG ----------
dag = DAG(
    dag_id="Airflow_Lab2",
    default_args=default_args,
    description="ML Pipeline for Advertising Data - Created by Nikhil Yellapragada, Northeastern University Data Analytics Engineering",
    schedule="@daily",
    catchup=False,
    tags=["mlops", "machine-learning", "nikhil-yellapragada"],
    owner_links={"Nikhil Yellapragada": "https://github.com/Nikhil20012/MLOps"},
    max_active_runs=1,
)

# ---------- Tasks ----------
owner_task = BashOperator(
    task_id="task_using_linked_owner",
    bash_command="echo 1",
    owner="Ramin Mohammadi",
    dag=dag,
)

send_email = EmailOperator(
    task_id='send_email',
    to='nikhil.yellapragada@gmail.com',
    subject='Airflow Lab 2 - Pipeline Completed by Nikhil Yellapragada',
    html_content=""" 
    <h3>Machine Learning Pipeline Completed Successfully!</h3>
    <p>Student: Nikhil Yellapragada</p>
    <p>Course: Data Analytics Engineering - Northeastern University</p>
    <p>The advertising data model has been trained and saved.</p>
    """,
    dag=dag
)

load_data_task = PythonOperator(
    task_id="load_data_task",
    python_callable=load_data,
    dag=dag,
)

data_preprocessing_task = PythonOperator(
    task_id="data_preprocessing_task",
    python_callable=data_preprocessing,
    op_args=[load_data_task.output],
    dag=dag,
)

separate_data_outputs_task = PythonOperator(
    task_id="separate_data_outputs_task",
    python_callable=separate_data_outputs,
    op_args=[data_preprocessing_task.output],
    dag=dag,
)

build_save_model_task = PythonOperator(
    task_id="build_save_model_task",
    python_callable=build_model,
    op_args=[separate_data_outputs_task.output, "model.sav"],
    dag=dag,
)

load_model_task = PythonOperator(
    task_id="load_model_task",
    python_callable=load_model,
    op_args=[separate_data_outputs_task.output, "model.sav"],
    dag=dag,
)

# Fire-and-forget trigger so this DAG can finish cleanly.
trigger_dag_task = TriggerDagRunOperator(
    task_id="my_trigger_task",
    trigger_dag_id="Airflow_Lab2_Flask",
    conf={"message": "Data from upstream DAG"},
    reset_dag_run=False,
    wait_for_completion=False,          # don't block
    trigger_rule=TriggerRule.ALL_DONE,  # still run even if something upstream fails
    dag=dag,
)

# ---------- Dependencies ----------
owner_task >> load_data_task >> data_preprocessing_task >> \
    separate_data_outputs_task >> build_save_model_task >> \
    load_model_task >> trigger_dag_task

# # Optional: email after model loads (independent branch)
load_model_task >> send_email

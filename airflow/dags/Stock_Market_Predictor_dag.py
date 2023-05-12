import os
import sys
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from pyprojroot import here as get_project_root

# Docker container check
# container DAG files are stored in opt/airflow/dags
if get_project_root() == '/':
    os.chdir(get_project_root() / 'src')
    sys.path.append(str(get_project_root() / 'src'))
else:
    os.chdir(get_project_root())
    sys.path.append(str(get_project_root()))

# function imports
from data_acquisition.main import main as data_aq_main
from data_processing.main import main as data_pr_main
from model_training.main import main as model_tr_main

DAG_args = {'owner': 'admin',
            'retries': 5,
            'retry delay': timedelta(minutes=5)}

with DAG(
    default_args=DAG_args,
    dag_id='data_pipeline_01',
    description='Runs full pipeline for the project',
    start_date=datetime(2023, 5, 11),
    schedule_interval='@daily'
)as dag:
    task1 = PythonOperator(
        task_id='data_acquisition',
        python_callable=data_aq_main
    )

    task2 = PythonOperator(
        task_id='feature_engineering',
        python_callable=data_pr_main
    )

    task3 = PythonOperator(
        task_id='model_training',
        python_callable=model_tr_main,
        op_kwargs={'name': 'DAG_SGDRegressor.joblib'}
    )

    task1 >> task2 >> task3
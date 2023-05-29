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
from model_training.main import model_train
from model_training.model import SGDregressor_train, RFregressor_train
import model_training.config as cfg


def select_best_model(ti):
    model_RMSEs = ti.xcom_pull(key='return_value', task_ids=['SGDmodel_training', 'RFmodel_training'])
    best_RMSE = ('model_name', float('inf'))
    for model in model_RMSEs:
        if model[1] < best_RMSE[1]:
            best_RMSE = model
    print(f'Model with lowest RMSE: {best_RMSE}')


DAG_args = {'owner': 'admin',
            'retries': 5,
            'retry delay': timedelta(minutes=5)}

with DAG(
    default_args=DAG_args,
    dag_id='data_pipeline_01',
    description='Runs full pipeline for the project',
    schedule_interval='@daily'
)as dag:
    task1 = PythonOperator(
        task_id='data_acquisition',
        python_callable=data_aq_main,
        do_xcom_push=False
    )

    task2 = PythonOperator(
        task_id='feature_engineering',
        python_callable=data_pr_main,
        do_xcom_push=False
    )

    task3 = [PythonOperator(
        task_id='SGDmodel_training',
        python_callable=model_train,
        op_kwargs={'estimator_func': SGDregressor_train,
                   'estimator_func_params': cfg.SGD_reg_params_grid,
                   'model_name': 'DAG_SGDRegressor'}
    ),
        PythonOperator(
            task_id='RFmodel_training',
            python_callable=model_train,
            op_kwargs={'estimator_func': RFregressor_train,
                       'estimator_func_params': cfg.RF_reg_params_grid,
                       'model_name': 'DAG_RFRegressor'}
    )]

    task4 = PythonOperator(
        task_id='select_model',
        python_callable=select_best_model
    )

    task1 >> task2 >> task3 >> task4
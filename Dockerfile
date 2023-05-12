# use apache airflow container instead
# syntax=docker/dockerfile:1
FROM apache/airflow:2.6.0-python3.8
# USER root
# RUN mkdir src
USER airflow

COPY requirements.txt /
RUN pip install --no-cache-dir -r /requirements.txt

#Copy all files in the current dir to main dir of container
# need to add some files in .dockerignore like data
COPY --chown=airflow:root . /src
COPY --chown=airflow:root dags/Stock_Market_Predictor_dag.py opt/airflow/dags
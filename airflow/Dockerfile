FROM apache/airflow:2.10.5-python3.10

USER root
RUN apt-get update && apt-get install -y postgis gdal-bin && apt-get clean

USER airflow
RUN pip install statsmodels pydeck geopy matplotlib pandas

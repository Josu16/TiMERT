# FROM pytorch/pytorch:1.10.2-cuda10.2-cudnn7-runtime
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

# Instalar las dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    pkg-config \
    libfreetype6-dev \
    libpng-dev \
    wget \
    unzip \
    graphviz \
    alpine-pico \
    tmux \
    && apt-get clean

    RUN mkdir /opt/code
    WORKDIR /opt/code
    
    COPY requirements.txt /opt/code/requirements.txt
    RUN pip install -r /opt/code/requirements.txt
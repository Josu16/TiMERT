# Usar la imagen base de PyTorch
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

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
    git \
    && apt-get clean

    
# Establecer (crear si no existe) el directorio de trabajo
WORKDIR /opt/code

# Configurar Git para que considere /opt/code como un directorio seguro
# Se necesita GIT para que MLflow registre los commits con los experimentos
RUN git config --global --add safe.directory /opt/code

# Copiar e instalar las dependencias de Python
COPY requirements.txt /opt/code/requirements.txt
RUN pip install -r /opt/code/requirements.txt

# # Crear un usuario llamado timert_dev en el contenedor
# TODO: Mejorar manejo de usuario en el contenedor
# RUN useradd -m timert_dev

# # Cambiar al nuevo usuario
# USER timert_dev

# Exponer el puerto en el que se ejecutará MLflow
EXPOSE 5000

EXPOSE 8888

# Establecer el punto de entrada y el comando por defecto
ENTRYPOINT ["mlflow", "ui"]
CMD ["--host", "0.0.0.0", "--port", "5000"]

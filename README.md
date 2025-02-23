# TiMERT: Time-series Modeling with Enhanced Representation Transformer for Classification
TiMERT is a foundational model designed for robust time-series analysis, combining the power of transformers with advanced pre-training techniques. This model excels in capturing complex temporal dependencies and patterns, making it ideal for Time Series Classification.

**Key Features:**

- Pre-training: TiMERT leverages a pre-training phase that enhances its ability to model time-series data, learning rich temporal representations that can generalize across various domains.
- Fine-tuning for Classification: The model includes a fine-tuning phase specifically tailored for time-series classification tasks, ensuring high accuracy and adaptability to different datasets.
- Versatility: TiMERT is suitable for applications across finance, healthcare, IoT, and more, where precise time-series classification is crucial.

**How It Works:**

1. Pre-training Phase: TiMERT is extensively trained on the UCR time series data set to learn generic temporal features, using a pretext-task called MAE (Masked AutoEncoder). Similar to the one used in RoBERTa, but prepared for temporal data.
2. Fine-tuning Phase: The model is then fine-tuned on specific classification tasks, allowing it to specialize in identifying and categorizing time-series patterns.

**Credits and References:**

Code Base: This project was built upon the foundation of a important research: https://arxiv.org/pdf/2310.03916

**Inspiration from Papers:**

To use TiMERT in your project, simply clone the repository and follow the instructions provided in the following documentation:

## Environment
This project uses Docker so it is not necessary to install anything directly on the host, you only need to have Docker (version 24.0.5 used). It is also advisable to have CUDA-compatible GPUs to make model wait times viable.

### Build the image

The source code includes a Dockerfile to build the container, the suggested way to do this is to run the command in the root directory of the project:

```
docker build -t timert-image -f Dockerfile .
```

Once the image is built, the container can be run with the respective parameters:

- -it: to prevent the execution from being left in the background.
- -- gpus all: To take all the GPUs available in the system, this code allows you to select different GPUs to do various experiments. (remove if you do not have GPUS)
- -v: to map the directories and make it possible to edit code from the host and have it immediately reflected in the container.

``` bash
// For GPUs Infraestructure
docker run -d -p 5000:5000 --gpus all --name timert -v ${PWD}:/opt/code timert-image

// For CPU only Infraestructure (not recomended)
docker run -d -p 5000:5000 --name timert -v ${PWD}:/opt/code timert-image

```

## Enter to container bash

``` bash
docker exec -it timert bash
```

## Data repository
TiMERT was trained for the UCR dataset, to get started you need to download the dataset and place it in your project directory.

https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCRArchive_2018.zip

The parameter files include the dataset directory parameter, put the path you have chosen.

## MLFlow for experiment tracking

The container includes an **instance of MLFLow UI** running in localhost and ready on the **port 5000**. All the experiments an models are located in the /mlruns directory.

http://localhost:5000

Using MLflow enriches the TiMERT experimentation environment by managing:
- Experiments.
- Model executions.
- Artifacts.
- Models.
- Traceability of experiments with their parameters and code version control.

## Reference model
For comparison purposes, classification the ```KNeighborsTimeSeriesClassifier``` classifier with DTW distance is available. This script, like the fine-tuning, will take all the datasets for classification, the obtained metrics serve as a reference for the difficulty of the datasets.

```bash
python timert_cli.py dtw --conf-file dtw_0000
```
Where:
- ```--conf-file```: It is the file where all the model parameters are.

## Non-pre-train model
For comparative purposes, the transformer-based time series classifier can also be run using the encoder without prior training, this to determine if pre-training really improves the capacity of the classifiers.

```
python timert_cli.py non-pretrain-classifier --conf-file xfmr_class_0000 --gpu-id 3
```


## Pre-Train

To start pre-training you must properly configure the parameter file for pre-training located on /parameters or create your own and indicate it in the following command:

```bash
python timert_cli.py pretrain --conf-file pre_mae_0000 --gpu-id 0 --register
```

Where:
- ```--gpu-id```: It is the identifier of the gpu to use (default is zero)
- ```--conf-file```: It is the file where all the model parameters are. Review the existing files or create your own.
- ```--register```: if the parameter appears, MLflow will register and version the output model, otherwise it will just save. Avoid this parameter is useful to execute "testing" version models for check the environment or try other configurations. **Use this parameter if you will fine tune the model**

## Fine-Tuning

To fine-tune a model it is necessary to have run the pre-training process with the ```--register``` flag. This script takes the model and the specified version from the mlflow file system.

```bash
python timert_cli.py finetuning --conf-file finet_class_0000 --gpu-id 3 --enc-name mae_first_approach --enc-ver 1
```

Where:
- ```--gpu-id```: It is the identifier of the gpu to use (default is zero)
- ```--conf-file```: It is the file where all the model parameters are.
- ```--enc-name```: The name of base model
- ```--enc-version```: The version of the base model

## Review of results.

All executions of both scripts are logged in MLflow, separated into two experiments: pre-train and fine-tuning. The logged artifacts and models also have their place in mlflow. http://localhost:5000

### Jupyter notebook for advanced results analytics
With the MLflow api and MLflow client you can get all the metrics of the executions and experiments, by default TiMERT includes an instance of Jupyter Lab running in the 8888 port. So You can check the localhost:8888 url and set the pasword for jupyter lab.

The ```Results analysis.ipynb``` file includes the analysis of original results.

## Troubleshooting

### MLflow

#### Multiple tunel ssh

Getting MLflow UI up and running is easy if you are working directly on the container from the local machine because the container run command maps the corresponding ports, but, if there are more ssh connections in between. For example: Host -> Server 1 -> Server 2 -> TiMERT container, it will be necessary to set the number of ssh tunnels needed to reach the container, for this case it would have to be done like this:

```
ssh -L 5000:localhost:5000 user@server1
```

inside server 1 do:

```
ssh -L 5000:localhost:5000 user@server2
```

and this as many times as there are servers to connect to in order to reach the container, this way you can open the mlflow manager on the local host like this:

http://localhost:5000

**NOTE**: you must ensure that on none of the servers or the host, port 5000 is used (configured by default for mlflow with TiMERT), this is done with the command:

```
netstat -tuln | grep 5000
```

If the above command gives an output before starting the container, then it is busy and you need to modify the dockerfile and port mapping to a free one before running the container.

### Jupyter

#### Stability connection
Sometimes if the container is restarted the jupyter server is down. To fix that it is necessary check the status of the process:
```
ps aux | grep jupyter
```

If the command responds with the jupyter in execution and the servers does not resopond, yo must to kill the process, check the PID of jupyter and execute:

```
kill -9 <PID>
```

After start the jupyter server:

´´´
jupyter notebook --no-browser --port=8888 --ip=0.0.0.0 --allow-root
´´´
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

```
docker run -it --gpus all --name TiMERT-container -v ${PWD}:/opt/code timert-image /bin/bash
```

## Pre-Train

To start pre-training you must properly configure the configuration file for pre-training or create your own and indicate it in the following command:

````
python timert_cli.py pretrain --gpu_id 0 --conf_file mae_001
````

Where:
- gpu_id: It is the identifier of the gpu to use (default is zero)
- conf_file: It is the file where all the model parameters are.


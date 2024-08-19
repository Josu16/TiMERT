import os
import torch
import time
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.optim as optim
from scipy import signal
from torch.utils.data import DataLoader, TensorDataset
from transformers import RobertaConfig, RobertaModel, Adafactor

from engine.core.ts_transformer import Transformer
from engine.core.timert_utils import _normalize_dataset, format_time, get_dataset, get_ucr_dataset_names, mask_data, timert_split_data


class TimertPreTrain:
    
    def __init__(self, params, mlflow, gpu_id: str):
        self.mlflow = mlflow

        self.gpu_id = gpu_id

        self.global_params = params["global"]
        self.prep_params = params["preprocess"]
        self.model_params = params["model"]
        self.pretext_params = params["pretext_task"]
        self.train_params = params["train"]

        ## Initial settings
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        np.random.seed(self.global_params["seed"])

        self.mlflow.log_param("gpu", gpu_id)  # MLflow

        # MLflow: Registrar hiperparámetros globales
        for key, value in self.global_params.items():  # MLflow
            self.mlflow.log_param(key, value)  # MLflow

        # MLflow: Registrar hiperparámetros de preprocesamiento
        for key, value in self.prep_params.items():  # MLflow
            self.mlflow.log_param(key, value)  # MLflow

        # MLflow: Registrar hiperparámetros de modelo
        for key, value in self.model_params.items():  # MLflow
            self.mlflow.log_param(key, value)  # MLflow

        # MLflow: Registrar hiperparámetros de la tarea de pretexto
        for key, value in self.pretext_params.items():  # MLflow
            self.mlflow.log_param(key, value)  # MLflow

        # Split dataset
        self.pretrain_names, _ = timert_split_data(self.prep_params["pretrain_fracc"], self.mlflow)

    def preprocessing_time_series(self):
        # preparación de las series temporales
        pretrain_data = []
        for data_name in self.pretrain_names:
            print("Processing dataset: ", data_name)
            dataset, _ = get_dataset("UCRArchive_2018", data_name, max_len=self.model_params["out_dim"])
            data_pretrain = _normalize_dataset(dataset)

            pretrain_data.append(data_pretrain)

        pretrain_data = np.concatenate(pretrain_data, axis=0)

        print("Pre-train dataset final shape: ", pretrain_data.shape)
        self.pretrain_data = torch.tensor(pretrain_data, dtype=torch.float32).to(self.device)  # Convertir los datos a un tensor de PyTorch

    def start_pretrain(self, register):
        batch_size = self.train_params["batch_size"]  # Definir el tamaño del batch
        dataset = TensorDataset(self.pretrain_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Definir el modelo
        model = Transformer(
            in_dim = self.model_params["in_dim"],
            out_dim = self.model_params["out_dim"],
            n_layer = self.model_params["n_layer"],
            n_dim = self.model_params["n_dim"],
            n_head = self.model_params["n_head"],
            norm_first = self.model_params["norm_first"],
            is_pos = self.model_params["is_pos"],
            is_projector = self.model_params["is_projector"],
            project_norm = self.model_params["project_norm"],
            dropout = self.model_params["dropout"]
        ).to(self.device)

        print("Modelo inicializado:")
        print(model)

        # MLflow: Registrar la arquitectura del modelo
        self.mlflow.log_param("model_architecture", model.__str__())  # MLflow

        # Definir la función de pérdida y el optimizador
        criterion = nn.MSELoss()  # Por ejemplo, MSE para series temporales
        optimizer = optim.Adam(model.parameters(), lr=self.train_params["lr"])
        aux_criterion = nn.L1Loss()  # evaluación adicional

        # MLflow: Registrar los parámetros de entrenamiento
        self.mlflow.log_param("optimizer_name", optimizer.__str__())  # MLflow
        self.mlflow.log_param("learning_rate", self.train_params["lr"])  # MLflow
        self.mlflow.log_param("loss_function", criterion.__str__())  # MLflow
        self.mlflow.log_param("n_epoch", self.train_params["n_epoch"])  # MLflow

        # Iniciar el contador total de tiempo
        total_start_time = time.time()

        best_loss = float("inf")
        best_epoch = None

        # Entrenamiento
        n_epoch = self.train_params["n_epoch"]

        for epoch in range(n_epoch):  # Número de épocas
            start_time = time.time()
            total_loss = 0.0
            total_mae = 0.0

            # Usamos tqdm para mostrar el progreso
            with tqdm(dataloader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {epoch+1}/{n_epoch}")
                for batch in tepoch:
                    masked_data, mask = mask_data(batch[0])  # Enmascarar datos
                    optimizer.zero_grad()

                    outputs = model.forward(
                        masked_data,
                        normalize=False,
                        to_numpy=False
                    )

                    outputs = outputs.unsqueeze(1)

                    loss = criterion(outputs[mask], batch[0][mask])  # Comparar solo los valores enmascarados
                    mae_loss = aux_criterion(outputs[mask], batch[0][mask])  # Calcular MAE (auxiliar)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    total_mae += mae_loss.item()
                    tepoch.set_postfix(batch_loss=loss.item())  # Mostrar la pérdida del batch

            # Calcular el tiempo de la época y la pérdida media
            epoch_time = time.time() - start_time
            avg_loss = total_loss / len(dataloader)
            avg_mae = total_mae / len(dataloader)

            # MLflow: Registrar la pérdida y el tiempo de la época
            self.mlflow.log_metric("epoch_time", epoch_time, step=epoch)  # MLflow
            self.mlflow.log_metric("average_loss", avg_loss, step=epoch)  # MLflow
            self.mlflow.log_metric("average_mae", avg_mae, step=epoch)  # MLflow

            # Convertir a minutos y segundos
            minutes_epoch = int(epoch_time // 60)
            seconds_epoch = int(epoch_time % 60)

            print(f"Epoch {epoch+1} completed in {minutes_epoch} minutes and {seconds_epoch} seconds ({epoch_time:.2f} seconds).")
            print(f" - (last) Batch Loss: {loss.item():.6f}, Average Loss: {avg_loss:.6f}")

            # Buscar la mejor pérdida
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_epoch = epoch

        total_end_time = time.time()

        formated_time = format_time(total_end_time, total_start_time)

        print(f"Pre-train completed in: {formated_time}")

        # MLflow: Registrar el tiempo total de entrenamiento
        self.mlflow.log_metric("total_training_time", total_end_time)  # MLflow
        self.mlflow.log_param("total_training_time_formated", formated_time)  # MLflow

        self.mlflow.log_metric("best_loss", best_loss)
        self.mlflow.log_metric("best_epoch", best_epoch)
        
        # MLflow: Guardar el modelo en MLflow
        self.mlflow.pytorch.log_model(
            model,
            "timert-xfmr-2024-ir0-beta",
            signature=False
        )  # MLflow 

        if register:
            # MLflow: Registrar formalmente el modelo en el Model Registry
            model_uri = f"runs:/{self.mlflow.active_run().info.run_id}/timert-xfmr-2024-ir0-beta"
            self.mlflow.register_model(model_uri, "mae_first_approach")

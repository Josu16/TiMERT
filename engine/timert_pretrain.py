## RoBERTa approach
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
from engine.core.timert_utils import _normalize_dataset, get_dataset, get_ucr_dataset_names, mask_data
from engine.data_models.global_params_model import TimertGlobalParams
from engine.data_models.pretrain_params_model import TimertPreTrainParams


class TimertPreTrain:
    
    def __init__(
            self, 
            model_params: TimertPreTrainParams, 
            global_params: TimertGlobalParams, 
            gpu_id: str
        ):

        self.model_params = model_params

        ## Initial settings
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        self.device = "cpu" #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        np.random.seed(global_params.seed)


        # Split dataset
        data_names = get_ucr_dataset_names()

        total_datasets = len(data_names)
        pretrain_size = int(total_datasets * global_params.pretrain_frac)

        index = np.random.permutation(total_datasets)

        pretrain_index = index[:pretrain_size]
        remaining_index = index[pretrain_size:]

        print("Pretrain size dataset (joined): ", pretrain_index.shape)
        print("Remaining size dataset (joined):", remaining_index.shape)

        self.pretrain_names = data_names[pretrain_index]
        downstream_names = data_names[remaining_index]

        # checking the datasets
        print("Datasets for pre-training: ", self.pretrain_names)
        print("Datasets for downstream tasks: ", downstream_names)



    def preprocessing_time_series(self):
        # preparación de las series temporales
        pretrain_data = []
        for data_name in self.pretrain_names:
            print("Processing dataset: ", data_name)
            dataset, _ = get_dataset("UCRArchive_2018", data_name, max_len=self.model_params.out_dim)
            data_pretrain = _normalize_dataset(dataset)

            # fasjfsfjsdfkasdj
            pretrain_data.append(data_pretrain)

        pretrain_data = np.concatenate(pretrain_data, axis=0)

        print("Pre-train dataset final shape: ", pretrain_data.shape)
        self.pretrain_data = torch.tensor(pretrain_data, dtype=torch.float32).to(self.device)  # Convertir los datos a un tensor de PyTorch


    def start_pretrain(self):
        batch_size = self.model_params.batch_size  # Definir el tamaño del batch
        dataset = TensorDataset(self.pretrain_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Definir el modelo
        model = Transformer(
            in_dim = self.model_params.in_dim,
            out_dim = self.model_params.out_dim,
            n_layer = self.model_params.n_layer,
            n_dim = self.model_params.n_dim,
            n_head = self.model_params.n_head,
            norm_first = self.model_params.norm_first,
            is_pos = self.model_params.is_pos,
            is_projector = self.model_params.is_projector,
            project_norm = self.model_params.project_norm,
            dropout = self.model_params.dropout
        ).to(self.device)

        print("Modelo inicializado:")
        print(model)


        # Definir la función de pérdida y el optimizador
        criterion = nn.MSELoss()  # Por ejemplo, MSE para series temporales
        optimizer = optim.Adam(model.parameters(), lr=self.model_params.lr)

        # Iniciar el contador total de tiempo
        total_start_time = time.time()

        # Entrenamiento
        for epoch in range(self.model_params.n_epoch):  # Número de épocas
            start_time = time.time()
            total_loss = 0.0

            # Usamos tqdm para mostrar el progreso
            with tqdm(dataloader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {epoch+1}/{self.model_params.n_epoch}")
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
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    tepoch.set_postfix(batch_loss=loss.item())  # Mostrar la pérdida del batch

            # Calcular el tiempo de la época y la pérdida media
            epoch_time = time.time() - start_time
            # Convertir a minutos y segundos
            minutes = int(epoch_time // 60)
            seconds = int(epoch_time % 60)

            avg_loss = total_loss / len(dataloader)

            print(f"Epoch {epoch+1} completed in {minutes} minutes and {seconds} seconds ({epoch_time:.2f} seconds).")
            print(f" - (last) Batch Loss: {loss.item():.6f}, Average Loss: {avg_loss:.6f}")

        # Calcular el tiempo total del entrenamiento
        total_time = time.time() - total_start_time
        total_days = int(total_time // (24 * 3600))
        total_time = total_time % (24 * 3600)
        total_hours = int(total_time // 3600)
        total_time %= 3600
        total_minutes = int(total_time // 60)
        total_seconds = int(total_time % 60)

        print(f"Pre-train completed in: {total_days} days, {total_hours} hours, {total_minutes} minutes, and {total_seconds} seconds.")
        torch.save(model, 'TiMER-768-exp-7.pth')
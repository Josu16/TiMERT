## RoBERTa approach
import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from scipy import signal
from torch.utils.data import DataLoader, TensorDataset
from transformers import RobertaConfig, RobertaModel

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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
            dataset, _ = get_dataset("UCRArchive_2018", data_name)  # TODO: regresar a ../UCRArchive cuando se refactorice.
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


        # config = RobertaConfig(
        #     hidden_size=512,  # Ajustar el tamaño de embeddings a 512
        #     num_hidden_layers=12,  # Mantener el mismo número de capas
        #     num_attention_heads=8,  # Ajustar el número de cabezas de atención proporcionalmente
        #     intermediate_size=2048  # Mantener el tamaño de la capa intermedia
        # )
        # model = RobertaModel(config).to(device)

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


        print(model)

        # print(model.num_parameters())


        # Definir la función de pérdida y el optimizador
        criterion = nn.MSELoss()  # Por ejemplo, MSE para series temporales
        optimizer = optim.Adam(model.parameters(), lr=self.model_params.lr)

        # Paso 4: Entrenamiento (esquemático)
        for epoch in range(self.model_params.n_epoch):  # Número de épocas
            for batch in dataloader:
                masked_data, mask = mask_data(batch[0])  # Enmascarar datos
                optimizer.zero_grad()

                # outputs = model(inputs_embeds=masked_data).last_hidden_state # se utiliza input_embeds para pasar directametne los embeddings

                
                outputs = model.forward(
                    masked_data,
                    normalize = False,
                    to_numpy = False
                )

                outputs = outputs.unsqueeze(1)

                # print(outputs.shape)
                # print(batch[0].shape)


                loss = criterion(outputs[mask], batch[0][mask])  # Comparar solo los valores enmascarados
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

        print("Entrenamiento completado")
        torch.save(model, 'TiMER-test-env.pth')
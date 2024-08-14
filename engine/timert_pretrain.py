## RoBERTa approach
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import RobertaConfig, RobertaModel
from engine.core.ts_transformer import Transformer
from scipy import signal

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Función de enmascarado para series temporales
def mask_data(data, mask_prob=0.15):
    mask = torch.full(data.shape, mask_prob, device=data.device)
    mask = torch.bernoulli(mask).bool()
    masked_data = data.clone()
    masked_data[mask] = 0  # Asignar cero a los valores enmascarados
    return masked_data, mask

def get_dataset(route, name, norm = True,  max_len = 512):
    # train_path = route + '/' + name + '/' + name + "_TRAIN.tsv"
    # test_path = route + '/' + name + '/' + name + "_TEST.tsv"


    train_path = os.path.join(route, 'Missing_value_and_variable_length_datasets_adjusted', name, name + "_TRAIN.tsv")
    test_path = os.path.join(route, 'Missing_value_and_variable_length_datasets_adjusted', name, name + "_TEST.tsv")

    # Verificar si los archivos existen en el subdirectorio
    if not os.path.isfile(train_path) or not os.path.isfile(test_path):
        # Si no existen, utilizar la ruta por defecto
        train_path = os.path.join(route, name, name + "_TRAIN.tsv")
        test_path = os.path.join(route, name, name + "_TEST.tsv")

    # print("ruta actual: ", os.listdir(os.getcwd()))

    print(train_path)
    dataset = np.concatenate(
            (np.loadtxt(train_path), np.loadtxt(test_path)),
            axis=0
        )

    # spec del dataset
    rows = dataset.shape[0]
    ts_len = dataset.shape[1] - 1

    # aislar la columna de etiquetas
    labels = dataset[:, 0].astype(int)
    # aislar el resto del dataset
    data = dataset[:, 1:]
    # expandir el dataset a las dimensiones que espera el codificador
    data = np.expand_dims(data, 1)

    if ts_len != max_len:
        # resampleo con transformada de Fourier
        data = signal.resample(data, max_len, axis = 2)  
    
    # print(data.shape)
    # print(labels.shape)
    
    return data, ts_len

def _normalize_dataset(data):
    data_mu = np.mean(data, axis=2, keepdims=True)
    data_sigma = np.std(data, axis=2, keepdims=True)
    data_sigma[data_sigma <= 0] = 1
    data = (data - data_mu) / data_sigma
    return data
## --------------------------- Configuración del dataset --------------------------------
# data = np.load('../pretrain.npy')  # Cargar los datos desde el archivo .npy

# print("primera fila del dataset", data[0, :, :])
# pretrain_data = torch.tensor(data, dtype=torch.float32).to(device)  # Convertir los datos a un tensor de PyTorch

np.random.seed(666)

data_names = np.array([
        'Adiac',
        'ArrowHead',
        'Beef',
        'BeetleFly',
        'BirdChicken',
        'Car',
        'CBF',
        'ChlorineConcentration',
        'CinCECGTorso',
        'Coffee',
        'Computers',
        'CricketX',
        'CricketY',
        'CricketZ',
        'DiatomSizeReduction',
        'DistalPhalanxOutlineAgeGroup',
        'DistalPhalanxOutlineCorrect',
        'DistalPhalanxTW',
        'Earthquakes',
        'ECG200',
        'ECG5000',
        'ECGFiveDays',
        'ElectricDevices',
        'FaceAll',
        'FaceFour',
        'FacesUCR',
        'FiftyWords',
        'Fish',
        'FordA',
        'FordB',
        'GunPoint',
        'Ham',
        'HandOutlines',
        'Haptics',
        'Herring',
        'InlineSkate',
        'InsectWingbeatSound',
        'ItalyPowerDemand',
        'LargeKitchenAppliances',
        'Lightning2',
        'Lightning7',
        'Mallat',
        'Meat',
        'MedicalImages',
        'MiddlePhalanxOutlineAgeGroup',
        'MiddlePhalanxOutlineCorrect',
        'MiddlePhalanxTW',
        'MoteStrain',
        'NonInvasiveFetalECGThorax1',
        'NonInvasiveFetalECGThorax2',
        'OliveOil',
        'OSULeaf',
        'PhalangesOutlinesCorrect',
        'Phoneme',
        'Plane',
        'ProximalPhalanxOutlineAgeGroup',
        'ProximalPhalanxOutlineCorrect',
        'ProximalPhalanxTW',
        'RefrigerationDevices',
        'ScreenType',
        'ShapeletSim',
        'ShapesAll',
        'SmallKitchenAppliances',
        'SonyAIBORobotSurface1',
        'SonyAIBORobotSurface2',
        'StarLightCurves',
        'Strawberry',
        'SwedishLeaf',
        'Symbols',
        'SyntheticControl',
        'ToeSegmentation1',
        'ToeSegmentation2',
        'Trace',
        'TwoLeadECG',
        'TwoPatterns',
        'UWaveGestureLibraryAll',
        'UWaveGestureLibraryX',
        'UWaveGestureLibraryY',
        'UWaveGestureLibraryZ',
        'Wafer',
        'Wine',
        'WordSynonyms',
        'Worms',
        'WormsTwoClass',
        'Yoga',
        'ACSF1',
        'AllGestureWiimoteX',
        'AllGestureWiimoteY',
        'AllGestureWiimoteZ',
        'BME',
        'Chinatown',
        'Crop',
        'DodgerLoopDay',
        'DodgerLoopGame',
        'DodgerLoopWeekend',
        'EOGHorizontalSignal',
        'EOGVerticalSignal',
        'EthanolLevel',
        'FreezerRegularTrain',
        'FreezerSmallTrain',
        'Fungi',
        'GestureMidAirD1',
        'GestureMidAirD2',
        'GestureMidAirD3',
        'GesturePebbleZ1',
        'GesturePebbleZ2',
        'GunPointAgeSpan',
        'GunPointMaleVersusFemale',
        'GunPointOldVersusYoung',
        'HouseTwenty',
        'InsectEPGRegularTrain',
        'InsectEPGSmallTrain',
        'MelbournePedestrian',
        'MixedShapesRegularTrain',
        'MixedShapesSmallTrain',
        'PickupGestureWiimoteZ',
        'PigAirwayPressure',
        'PigArtPressure',
        'PigCVP',
        'PLAID',
        'PowerCons',
        'Rock',
        'SemgHandGenderCh2',
        'SemgHandMovementCh2',
        'SemgHandSubjectCh2',
        'ShakeGestureWiimoteZ',
        'SmoothSubspace',
        'UMD'
    ])

total_datasets = len(data_names)
pretrain_size = int(total_datasets * 0.65)

index = np.random.permutation(total_datasets)

pretrain_index = index[:pretrain_size]
remaining_index = index[pretrain_size:]

print("Tamaño del pretrain: ", pretrain_index.shape)
print("Tamaño del restante: ", remaining_index.shape)

pretrain_names = data_names[pretrain_index]
downstream_names = data_names[remaining_index]

### comprobación de datasets
print("datasets de pre entrenamiento: ", pretrain_names)
print("datasets para tareas posteriores: ", downstream_names)

# preparación de las series temporales

pretrain_data = []
for data_name in pretrain_names:
    dataset, ts_len = get_dataset("UCRArchive_2018", data_name)  # TODO: regresar a ../UCRArchive cuando se refactorice.
    data_pretrain = _normalize_dataset(dataset)

    # fasjfsfjsdfkasdj
    pretrain_data.append(data_pretrain)

pretrain_data = np.concatenate(pretrain_data, axis=0)

print("tamaño del pretrain (ahora está completo) ", pretrain_data.shape)
pretrain_data = torch.tensor(pretrain_data, dtype=torch.float32).to(device)  # Convertir los datos a un tensor de PyTorch


# print("primera fila del dataset", pretrain_data[0, :, :])
## ----------------------- FIN Configuración del dataset --------------------------------


batch_size = 128  # Definir el tamaño del batch
dataset = TensorDataset(pretrain_data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# config = RobertaConfig(
#     hidden_size=512,  # Ajustar el tamaño de embeddings a 512
#     num_hidden_layers=12,  # Mantener el mismo número de capas
#     num_attention_heads=8,  # Ajustar el número de cabezas de atención proporcionalmente
#     intermediate_size=2048  # Mantener el tamaño de la capa intermedia
# )
# model = RobertaModel(config).to(device)

model = Transformer(
    in_dim = 1,
    out_dim = 512,
    n_layer = 4,
    n_dim = 64,
    n_head = 8,
    norm_first = True,
    is_pos = True,
    is_projector = True,
    project_norm = 'LN',
    dropout = 0.0
).to(device)


print(model)

# print(model.num_parameters())


# Definir la función de pérdida y el optimizador
criterion = nn.MSELoss()  # Por ejemplo, MSE para series temporales
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Paso 4: Entrenamiento (esquemático)
for epoch in range(400):  # Número de épocas
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
torch.save(model, 'roberta_pretrained_model_exp1.pth')
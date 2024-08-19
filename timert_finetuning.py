from sklearn.model_selection import train_test_split
from collections import OrderedDict
import numpy as np
from scipy import signal
from engine.core.timert_utils import format_time, get_dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from TSTransformer import Transformer
from tqdm import tqdm
# from torch.optim import Adam
import time
import pandas as pd
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

class CustomTSClassifier(torch.nn.Module):
        def __init__(self, encoder, n_class, n_dim = 64, n_layer = 2):
            super(CustomTSClassifier, self).__init__()
            self.encoder = encoder
            self.add_module('encoder', encoder)

            in_dim_ = 512  ## TODO: Volver variable
            out_dim_ = n_dim
            layers = OrderedDict()

            for i in range(n_layer - 1):
                layers[f'linear_{i:02d}'] = nn.Linear(in_dim_, out_dim_)
                layers[f'relu_{i:02d}'] = nn.ReLU()
                in_dim_ = out_dim_
                out_dim_ = n_dim

            layers[f'linear_{n_layer - 1:02d}'] = nn.Linear(in_dim_, n_class)
            self.classifier = nn.Sequential(layers)

        def forward(self, ts):
            transformer_rep = self.encoder(
                ts = ts,
                normalize = True,
                to_numpy = False
            )
            
            logits = self.classifier(transformer_rep)
            logits = logits.squeeze(1) # TODO: revisar por qué se mantiene esa dimensión.

            return logits

def _normalize_dataset(data):
    data_mu = np.mean(data, axis=2, keepdims=True)
    data_sigma = np.std(data, axis=2, keepdims=True)
    data_sigma[data_sigma <= 0] = 1
    data = (data - data_mu) / data_sigma
    return data

def _relabel(label):
    label_set = np.unique(label)
    n_class = len(label_set)

    label_re = np.zeros(label.shape[0], dtype=int)
    for i, label_i in enumerate(label_set):
        label_re[label == label_i] = i
    return label_re, n_class


# Cargar el modelo preentrenado
model = torch.load('best_model.pth')  # <- actualmente v1 custom es el que se entreno con TODOS Los datos. el modelo tramposo
model.to(device)
print("tipo de dato: ", type(model))

    ## -------------------- PREPARACIÓN DE LOS CONJUNTOS ------------------------------
results_df = pd.DataFrame(columns=['Dataset', 'Validation Loss', 'Validation Accuracy', 'Test Loss', 'Test Accuracy'])

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
        'UMD',
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

all_metrics = []

for data_name in downstream_names:
    print("Dataset: ", data_name)

    data, labels = get_dataset("UCRArchive_2018",data_name)

    labels, n_class = _relabel(labels)

    # separación de conjuntos de entrenamiento, validación y prueba

    train_frac = 0.6
    valid_frac = 0.2
    test_frac = 0.2

    assert np.isclose(train_frac + valid_frac + test_frac, 1.0)

    # print("data: ", data)

    train_data, tmp_data, train_labels, tmp_labels = train_test_split(
        data, labels, train_size = train_frac , stratify = labels, random_state = 666
    )

    # print("tentrada x", train_data)
    # separar el resto en validación y prueba, ajustando las fracciones
    remaining_frac = valid_frac + test_frac
    valid_size = valid_frac / remaining_frac
    test_size = test_frac / remaining_frac

    try:
        valid_data, test_data, valid_labels, test_labels = train_test_split(
            tmp_data, tmp_labels, train_size = valid_size, stratify = tmp_labels, random_state = 666
        )
    except Exception as e:
        # Podría ocurrir que el número de clases no alcancen para estratificar
        print(e)
        valid_data, test_data, valid_labels, test_labels = train_test_split(
            tmp_data, tmp_labels, train_size = valid_size, random_state = 666
        )
        # agregar más detalles de la distribución de clases.

    print("Size of train: ", train_data.shape)
    print("Size of validation: ", valid_data.shape)
    print("Size of test: ", test_data.shape)


    # data = np.load('../pretrain.npy')
    # data = torch.tensor(data, dtype=torch.float32).to(device)


    ## -------------------- PREPROCESAMIENTO DE LAS SEREIS ------------------------------

    train_data = _normalize_dataset(train_data)
    valid_data = _normalize_dataset(valid_data)
    test_data = _normalize_dataset(test_data)

    ## -------------------- PREPARACIÓN DEL MODELO ------------------------------

    # parámetros del clasificador

    classifier = CustomTSClassifier(model, n_class, n_dim = 64, n_layer = 2).to(device)

    ## -------------------- PREPARACIÓN y ENTRENAMIENTO DEL MODELO ------------------------------

    # print("Número de dimensiones de salida: ", model.out_dim)

    ## Parámetros:
    lr = 0.0001
    optimizer = torch.optim.AdamW(
        model.parameters(), lr = lr
    )
    criterion = nn.CrossEntropyLoss()
    n_data = train_data.shape[0]
    batch_size = 64
    n_iter = np.ceil((n_data / batch_size))
    n_iter = int(n_iter)
    n_epoch = 10

    train_data = torch.tensor(train_data, dtype=torch.float32).to(device)
    train_labels = torch.tensor(train_labels, dtype=torch.long).to(device)

    valid_data = torch.tensor(valid_data, dtype=torch.float32).to(device)
    valid_labels = torch.tensor(valid_labels, dtype=torch.long).to(device)

    test_data = torch.tensor(test_data, dtype=torch.float32).to(device)
    test_labels = torch.tensor(test_labels, dtype=torch.long).to(device)


    dataset = TensorDataset(train_data, train_labels)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

    dataset_valid = TensorDataset(valid_data, valid_labels)
    dataloader_valid = DataLoader(dataset_valid, batch_size = batch_size, shuffle = True)

    dataset_test = TensorDataset(test_data, test_labels)
    dataloader_test = DataLoader(dataset_test, batch_size = batch_size, shuffle = True)

    total_start_time = time.time()

    best_eval_loss = float("inf")
    best_epoch = None

    for epoch in range(n_epoch):
        classifier.train()
        start_time = time.time()
        train_loss = 0

        correct_pred = 0
        total_examples = 0

        with tqdm(dataloader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}/{n_epoch}")
            for batch in tepoch:
                optimizer.zero_grad()

                ## Forward + backward + optimizar
                logits = classifier.forward(batch[0])
                loss = criterion(logits, batch[1])
                loss.backward()
                optimizer.step()

                # Estadísticas
                train_loss += loss.item()
                _, pred = torch.max(logits, dim=1)
                correct_pred += torch.sum(pred == batch[1]).item()
                total_examples += batch[1].size(0)

                # Impresión consola
                tepoch.set_postfix(batch_loss=loss.item())

        # Calcular la pérdida promedio
        avg_train_loss = train_loss / len(dataloader)
        # Calcular la exacitud.
        train_accuracy = (100 * correct_pred / total_examples)

        # Estadísticas de entrenamiento
        print(f" - TRAIN: (last) Batch Loss: {loss.item():.6f}, Average Loss: {avg_train_loss:.6f}, --->[Accuracy: {train_accuracy:0.6f}%]")

        # ----------------------- COMIENZA LA EVALUACIÓN
        val_loss = 0
        total_examples = 0
        correct_pred = 0
        # Validación del modelo
        classifier.eval()
        with torch.no_grad():
            with tqdm(dataloader_valid, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch validation")
                for batch in tepoch:
                    outputs_valid = classifier.forward(batch[0])
                    loss = criterion(outputs_valid, batch[1])

                    # estadísticas
                    val_loss += loss.item()
                    _, pred = torch.max(outputs_valid, dim=1)
                    correct_pred += torch.sum(pred == batch[1]).item()
                    total_examples += batch[1].size(0)  # TODO: este puede ya no ser necesario, está calculado en el train
                
                # Calcular la pérdida promedio de la validación
                avg_val_loss = val_loss / len(dataloader_valid)
                # Calcular la exactitud de validación
                eval_accuracy = (100 * correct_pred / total_examples)
                # Estadísticas de entrenamiento
                print(f" - VALIDATION: (last) Batch Loss: {loss.item():.6f}, Average Loss: {avg_val_loss:.6f}, Accuracy: {eval_accuracy:0.6f}%")

        # calcular el tiempo de la época
        epoch_time = time.time() - start_time
        # Convertir a minutos y segundos
        minutes_epoch = int(epoch_time // 60)
        seconds_epoch = int(epoch_time % 60)
        print(f"Epoch {epoch+1} completed in {minutes_epoch} minutes and {seconds_epoch} seconds ({epoch_time:.2f} seconds).")

        # # Buscar el mejor modelo
        if avg_val_loss < best_eval_loss:
            best_epoch = epoch
            best_model_params = classifier.state_dict()

            best_train_loss = avg_train_loss
            best_train_accuracy = train_accuracy
            best_eval_loss = avg_val_loss
            best_eval_accuracy = eval_accuracy

    total_end_time = time.time()
    formated_time = format_time(total_end_time, total_start_time)
    print(f"train and validate completed in: {formated_time}")


    ## -------------------- PRUEBA DEL MODELO. ------------------------------
    classifier.load_state_dict(best_model_params)
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader_test:
            outputs = classifier.forward(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_accuracy = 100 * correct / total
    print(f"\n\t --- TEST --- [{data_name}] Accuracy: {test_accuracy:0.6f}% \n")

    # # Agregar los resultados a la lista
    all_metrics.append([data_name, best_epoch, best_train_loss, best_train_accuracy, best_eval_loss, best_eval_accuracy, test_accuracy])

    hshfdhsakdfhahkk
all_metrics_array = np.array(all_metrics)
header = "dataset_name, best_epoch, best_train_loss, best_train_accuracy, best_eval_loss, best_eval_accuracy, test_accuracy"
np.savetxt('all_datasets_metrics.csv', all_metrics_array, delimiter=',', fmt='%s', header=header, comments='')
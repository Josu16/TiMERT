from sklearn.model_selection import train_test_split
from collections import OrderedDict
import numpy as np
from scipy import signal
import torch
import torch.nn as nn
import time
import pandas as pd
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

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

def get_dataset(route, name, norm = True,  max_len = 512):
    train_path = route + '/' + name + '/' + name + "_TRAIN.tsv"
    test_path = route + '/' + name + '/' + name + "_TEST.tsv"

    print(os.getcwd())

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
    
    print(data.shape)
    print(labels.shape)
    
    return data, labels

# Cargar el modelo preentrenado
model = torch.load('roberta_pretrained_model_exp1.pth') # <- actualmente v1 custom es el que se entreno con TODOS Los datos. el modelo tramposo
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



for data_name in downstream_names:
    print("Dataset: ", data_name)

    data, labels = get_dataset("../UCRArchive_2018",data_name)

    labels, n_class = _relabel(labels)

    # separación de conjuntos de entrenamiento, validación y prueba

    train_frac = 0.6
    valid_frac = 0.2
    test_frac = 0.2

    assert np.isclose(train_frac + valid_frac + test_frac, 1.0)

    train_data, tmp_data, train_labels, tmp_labels = train_test_split(
        data, labels, train_size = train_frac , stratify = labels, random_state = 666
    )

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

    ## -------------------- PREPARACIÓN DEL MODELO ------------------------------

    # parámetros del clasificador

    classifier = CustomTSClassifier(model, n_class, n_dim = 64, n_layer = 2).to(device)

    ## -------------------- PREPARACIÓN y ENTRENAMIENTO DEL MODELO ------------------------------


    # print("Número de dimensiones de salida: ", model.out_dim)
    model.train() 

    ## Parámetros:
    lr = 0.0001
    optimizer = torch.optim.AdamW(
        model.parameters(), lr = lr
    )
    n_data = train_data.shape[0]
    batch_size = 64
    n_iter = np.ceil((n_data / batch_size))
    n_iter = int(n_iter)
    n_epoch = 400

    loss_train = np.zeros(n_epoch)
    toc_train = np.zeros(n_epoch)

    for i in range(0, n_epoch):
        tic = time.time()
        loss_epoch = 0
        idx_order = np.random.permutation(n_data)
        # print("tamaño de idx order ", idx_order.shape)
        # print("idx_order ", idx_order)

        for j in range(n_iter):
            optimizer.zero_grad()

            idx_start = j * batch_size
            idx_end = (j + 1) * batch_size
            if idx_end > n_data:
                idx_end = n_data
            idx_batch = idx_order[idx_start:idx_end]

            batch_size_ = idx_end - idx_start
            if batch_size_ < batch_size:
                n_fill = batch_size - batch_size_
                idx_fill = idx_order[:n_fill]
                idx_batch = np.concatenate((idx_batch, idx_fill), axis=0)

            # print("idx start ", idx_start)
            # print("idx end ", idx_end)
            # print("Batch size ", batch_size_)
            # print("idx_batch ", idx_batch)


            data_batch = train_data[idx_batch, :, :]
            label_batch = train_labels[idx_batch]

            data_batch = torch.from_numpy(data_batch).to(device, dtype=torch.long)

            label_batch = torch.from_numpy(label_batch)
            label_batch = label_batch.to(device, dtype=torch.long)

            logits = classifier.forward(data_batch)

            # print("forma de logits: ", logits.shape)
            # print("forma de label_batch: ", label_batch.shape)

            loss = nn.CrossEntropyLoss()(logits, label_batch)
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()

        loss_epoch /= n_iter
        toc_epoch = time.time() - tic

        loss_train[i] = loss_epoch
        toc_train[i] = toc_epoch

        print((f'epoch {i + 1}/{n_epoch}, '
                f'loss={loss_epoch:0.4f}, '
                f'time={toc_epoch:0.2f}.'))

    ## -------------------- VALIDACIÓN y PRUEBA DEL MODELO. ------------------------------

    def evaluate(model, data, labels, batch_size):
        model.eval()  # Poner el modelo en modo de evaluación
        n_data = data.shape[0]
        n_iter = np.ceil(n_data / batch_size)
        n_iter = int(n_iter)
        
        correct_predictions = 0
        total_predictions = 0
        running_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():  # Desactivar el cálculo de gradientes
            for i in range(n_iter):
                idx_start = i * batch_size
                idx_end = (i + 1) * batch_size
                if idx_end > n_data:
                    idx_end = n_data
                
                data_batch = data[idx_start:idx_end, :, :]
                label_batch = labels[idx_start:idx_end]

                data_batch = torch.from_numpy(data_batch).to(device, dtype=torch.float32)
                label_batch = torch.from_numpy(label_batch).to(device, dtype=torch.long)
                
                logits = model.forward(data_batch)
                loss = criterion(logits, label_batch)
                running_loss += loss.item()
                
                _, predicted = torch.max(logits, 1)
                correct_predictions += (predicted == label_batch).sum().item()
                total_predictions += label_batch.size(0)
        
        accuracy = correct_predictions / total_predictions
        avg_loss = running_loss / n_iter
        return avg_loss, accuracy


    valid_data = _normalize_dataset(valid_data)
    test_data = _normalize_dataset(test_data)

    batch_size = 64  # Define el tamaño del batch para la validación y prueba

    # Validación del modelo
    val_loss, val_accuracy = evaluate(classifier, valid_data, valid_labels, batch_size)
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

    # Prueba del modelo
    test_loss, test_accuracy = evaluate(classifier, test_data, test_labels, batch_size)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    # Agregar los resultados al DataFrame
    results_df = results_df.append({
        'Dataset': data_name,
        'Validation Loss': val_loss,
        'Validation Accuracy': val_accuracy,
        'Test Loss': test_loss,
        'Test Accuracy': test_accuracy
    }, ignore_index=True)

results_df.to_csv("Resultados_segundo_enfoque_exp1.csv")
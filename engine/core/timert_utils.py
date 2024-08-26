import torch
import os
import numpy as np
from scipy import signal


def _normalize_dataset(data):
    data_mu = np.mean(data, axis=2, keepdims=True)
    data_sigma = np.std(data, axis=2, keepdims=True)
    data_sigma[data_sigma <= 0] = 1
    data = (data - data_mu) / data_sigma
    return data

# TODO: Este no debería ir acá
def get_dataset(route, name, max_len = 512):
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

    # print(train_path)
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

    if max_len != None and ts_len != max_len:
        # resampleo con transformada de Fourier
        data = signal.resample(data, max_len, axis = 2)  
    
    # print(data.shape)
    # print(labels.shape)
    
    return data, labels

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def format_time(end_time, start_time):
    # Calcular el tiempo total del entrenamiento
    total_time = end_time - start_time
    total_days = int(total_time // (24 * 3600))
    total_time = total_time % (24 * 3600)
    total_hours = int(total_time // 3600)
    total_time %= 3600
    total_minutes = int(total_time // 60)
    total_seconds = int(total_time % 60)

    return f"{total_days} days, {total_hours} hours, {total_minutes} minutes, and {total_seconds} seconds"

def timert_split_data(pretrain_frac, mlflow):
    data_names = get_ucr_dataset_names()

    total_datasets = len(data_names)
    pretrain_size = int(total_datasets * pretrain_frac)

    index = np.random.permutation(total_datasets)

    pretrain_index = index[:pretrain_size]
    remaining_index = index[pretrain_size:]

    print("Pretrain size dataset (joined): ", pretrain_index.shape)
    print("Remaining size dataset (joined):", remaining_index.shape)

    pretrain_names = data_names[pretrain_index]
    downstream_names = data_names[remaining_index]

    # MLflow: Registrar los nombres de los datasets usados
    mlflow.log_param("pretrain_datasets", str(pretrain_names))  # MLflow
    mlflow.log_param("downstream_datasets", str(downstream_names))  # MLflow

    # checking the datasets
    print("Datasets for pre-training: ", pretrain_names)
    print("Datasets for downstream tasks: ", downstream_names)

    return pretrain_names, downstream_names

def _relabel(label):
    label_set = np.unique(label)
    n_class = len(label_set)

    label_re = np.zeros(label.shape[0], dtype=int)
    for i, label_i in enumerate(label_set):
        label_re[label == label_i] = i
    return label_re, n_class, label_set

def get_ucr_dataset_names() -> np.array:
    return np.array([
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

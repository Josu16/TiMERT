from sklearn.model_selection import train_test_split
import numpy as np
from scipy import signal
from engine.core.classifier import CustomTSClassifier
from engine.core.timert_utils import _normalize_dataset, _relabel, format_time, get_dataset, get_ucr_dataset_names, timert_split_data
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from TSTransformer import Transformer
from tqdm import tqdm
# from torch.optim import Adam
import time
import pandas as pd
import os


class TimertFineTuning:
    def __init__(self, params, encoder, mlflow, gpu_id):
        self.mlflow = mlflow
        self.gpu_id = gpu_id

        self.global_params = params["global"]
        self.prep_params = params["preprocess"]
        self.encoder_params = params["encoder"]
        self.class_params = params["classifier"]
        self.train_params = params["train"]

        ## initial settings
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        np.random.seed(self.global_params["seed"])

        self.mlflow.log_param("gpu", gpu_id)  # MLflow
        self.mlflow.log_param("encoder_name", encoder["name"])
        self.mlflow.log_param("encoder_version", encoder["version"])

        # MLflow: Registrar hiperparámetros globales
        for key, value in self.global_params.items():  # MLflow
            self.mlflow.log_param(key, value)  # MLflow

        # MLflow: Registrar hiperparámetros de preprocesamiento
        for key, value in self.prep_params.items():  # MLflow
            self.mlflow.log_param(key, value)  # MLflow

        # MLflow: Registrar hiperparámetros de modelo
        for key, value in self.encoder_params.items():  # MLflow
            self.mlflow.log_param(key, value)  # MLflow

        # MLflow: Registrar hiperparámetros de la tarea de pretexto
        for key, value in self.class_params.items():  # MLflow
            self.mlflow.log_param(key, value)  # MLflow

        # Cargar el modelo preentrenado
        model_uri = f"models:/{encoder['name']}/{encoder['version']}"
        self.model = self.mlflow.pytorch.load_model(model_uri)
        self.model.to(self.device)
        print("tipo de dato: ", type(self.model))

        # split dataset (names)
        _, self.downstream_names = timert_split_data(self.prep_params["pretrain_frac"], mlflow)

    def runn_all_models(self):
        all_metrics = []
        total_start_time = time.time()

        for data_name in self.downstream_names:
            with self.mlflow.start_run(run_name=str(data_name), nested=True):
                print("Dataset: ", data_name)

                data, labels = get_dataset("UCRArchive_2018",data_name)

                labels, n_class = _relabel(labels)

                # separación de conjuntos de entrenamiento, validación y prueba

                train_frac = self.prep_params["train_frac"]
                valid_frac = self.prep_params["valid_frac"]
                test_frac = self.prep_params["test_frac"]

                self.mlflow.log_param("train_fraction", train_frac)
                self.mlflow.log_param("valid_fraction", valid_frac)
                self.mlflow.log_param("test_fraction", test_frac)

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

                ## -------------------- PREPROCESAMIENTO DE LAS SEREIS ------------------------------

                train_data = _normalize_dataset(train_data)
                valid_data = _normalize_dataset(valid_data)
                test_data = _normalize_dataset(test_data)

                ## -------------------- PREPARACIÓN DEL MODELO ------------------------------

                # parámetros del clasificador

                classifier = CustomTSClassifier(self.model, self.encoder_params["out_dim"], n_dim = 64, n_layer = 2).to(self.device)
                print("Classifier created")

                self.mlflow.log_param("classifier_architecture", classifier.__str__())

                ## -------------------- PREPARACIÓN y ENTRENAMIENTO DEL MODELO ------------------------------

                ## Parámetros:
                lr = 0.0001
                optimizer = torch.optim.AdamW(
                    self.model.parameters(), lr = self.train_params["lr"]
                )
                criterion = nn.CrossEntropyLoss()
                batch_size = self.train_params["batch_size"]
                n_epoch = self.train_params["n_epoch"]

                self.mlflow.log_param("optimizer_name", optimizer.__str__())  # MLflow
                self.mlflow.log_param("learning_rate", self.train_params["lr"])  # MLflow
                self.mlflow.log_param("loss_function", criterion.__str__())  # MLflow
                self.mlflow.log_param("n_epoch", n_epoch)  # MLflow

                train_data = torch.tensor(train_data, dtype=torch.float32).to(self.device)
                train_labels = torch.tensor(train_labels, dtype=torch.long).to(self.device)

                valid_data = torch.tensor(valid_data, dtype=torch.float32).to(self.device)
                valid_labels = torch.tensor(valid_labels, dtype=torch.long).to(self.device)

                test_data = torch.tensor(test_data, dtype=torch.float32).to(self.device)
                test_labels = torch.tensor(test_labels, dtype=torch.long).to(self.device)


                dataset = TensorDataset(train_data, train_labels)
                dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

                dataset_valid = TensorDataset(valid_data, valid_labels)
                dataloader_valid = DataLoader(dataset_valid, batch_size = batch_size, shuffle = True)

                dataset_test = TensorDataset(test_data, test_labels)
                dataloader_test = DataLoader(dataset_test, batch_size = batch_size, shuffle = True)

                current_dataset_start_time = time.time()

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

                    self.mlflow.log_metric("average_loss_train", avg_train_loss, step=epoch)  # MLflow
                    self.mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)  # MLflow

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

                            self.mlflow.log_metric("average_loss_validation", avg_val_loss, step=epoch)  # MLflow
                            self.mlflow.log_metric("validation_accuracy", eval_accuracy, step=epoch)  # MLflow
                            
                            # Estadísticas de entrenamiento
                            print(f" - VALIDATION: (last) Batch Loss: {loss.item():.6f}, Average Loss: {avg_val_loss:.6f}, Accuracy: {eval_accuracy:0.6f}%")

                    # calcular el tiempo de la época
                    epoch_time = time.time() - start_time
                    self.mlflow.log_metric("epoch_time", epoch_time, step=epoch)  # MLflow
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

                current_dataset_end_time = time.time()
                formated_time = format_time(current_dataset_end_time, current_dataset_start_time)
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

                self.mlflow.log_metric("best_epoch", best_epoch)
                self.mlflow.log_metric("best_train_loss", best_train_loss)
                self.mlflow.log_metric("best_train_accuracy", best_train_accuracy)
                self.mlflow.log_metric("best_eval_loss", best_eval_loss)
                self.mlflow.log_metric("best_eval_accuracy", best_eval_accuracy)
                self.mlflow.log_metric("test_accuracy", test_accuracy)
                # # Agregar los resultados a la lista
                all_metrics.append([data_name, best_epoch, best_train_loss, best_train_accuracy, best_eval_loss, best_eval_accuracy, test_accuracy])

                # hshfdhsakdfhahkk

        total_end_time = time.time()
        formated_time = format_time(total_end_time, total_start_time)

        print(f"Fine Tuning completed in: {formated_time}")

        self.mlflow.log_metric("total_training_time", total_end_time)
        self.mlflow.log_param("total_training_time_formated", formated_time)


        all_datasets_acc = [row[-1] for row in all_metrics]
        avg_test_accuracy = sum(all_datasets_acc) / len(all_datasets_acc)
        self.mlflow.log_metric("avg_test_accuracy", avg_test_accuracy)
        
        all_metrics_array = np.array(all_metrics)
        header = "dataset_name, best_epoch, best_train_loss, best_train_accuracy, best_eval_loss, best_eval_accuracy, test_accuracy"
        np.savetxt('all_datasets_metrics.csv', all_metrics_array, delimiter=',', fmt='%s', header=header, comments='')
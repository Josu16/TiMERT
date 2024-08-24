
import time
import numpy as np
import tslearn.neighbors
from sklearn.model_selection import train_test_split
from engine.core.timert_utils import _normalize_dataset, _relabel, format_time, get_dataset, timert_split_data


class TimertDTW():
    def __init__(self, params, mlflow, neighborg_conf = [1, 3, 5]):
        self.prep_params = params["preprocess"]
        self.global_params = params["global"]

        self.neighborg_conf = neighborg_conf
        self.mlflow = mlflow
        np.random.seed(self.global_params["seed"])
        
        for key, value in self.prep_params.items():  # MLflow
            self.mlflow.log_param(key, value)  # MLflow
        _, self.downstream_names = timert_split_data(self.prep_params["pretrain_frac"], self.mlflow)
        
    def run_all_classifiers(self):
        all_metrics = []
        total_start_time = time.time()

        for data_name in self.downstream_names:
            with self.mlflow.start_run(run_name=str(data_name), nested=True):
                print("Dataset: ", data_name)

                data, labels = get_dataset(self.global_params["data_dir"], data_name, max_len=None)

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

                self.mlflow.log_param("train_size", train_data.shape[0])
                self.mlflow.log_param("valid_size", valid_data.shape[0])
                self.mlflow.log_param("test_size", test_data.shape[0])

                ## -------------------- PREPROCESAMIENTO DE LAS SEREIS ------------------------------

                train_data = _normalize_dataset(train_data)
                valid_data = _normalize_dataset(valid_data)
                test_data = _normalize_dataset(test_data)

                ## -------------------- PREPARACIÓN DEL MODELO ------------------------------

                data_train = np.swapaxes(train_data, 1, 2)
                valid_data = np.swapaxes(valid_data, 1, 2)
                test_data = np.swapaxes(test_data, 1, 2)
                print("starting train...")
                model = tslearn.neighbors.KNeighborsTimeSeriesClassifier(
                    n_neighbors=1, n_jobs=-1, verbose=1)
                train_start_time = time.time()
                model = model.fit(data_train, train_labels)
                train_end_time = time.time()
                total_train_time = train_end_time - train_start_time

                predict_valid, acc_valid, time_valid = self._get_predict(
                    valid_data, valid_labels, model)
                predict_test, acc_test, time_test = self._get_predict(
                    test_data, test_labels, model)
                
                
                self.mlflow.log_metric("validation_accuracy", acc_valid)
                self.mlflow.log_metric("total_train_eval_time", time_valid)

                self.mlflow.log_metric("test_accuracy", acc_test)
                self.mlflow.log_metric("total_test_time", time_test)
            all_metrics.append([data_name, acc_valid, acc_test])

        total_end_time = time.time()
        formated_time = format_time(total_end_time, total_start_time)

        print(f"DTW classification completed in: {formated_time}")
        self.mlflow.log_metric("total_training_time", total_end_time)

        all_datasets_acc = [row[-1] for row in all_metrics]
        avg_test_accuracy = sum(all_datasets_acc) / len(all_datasets_acc)
        self.mlflow.log_metric("avg_test_accuracy", avg_test_accuracy)
                

    def _get_predict(self, data, label, model):
        tic = time.time()
        predict = model.predict(data)
        predict_time = time.time() - tic
        acc = np.sum(predict == label) / label.shape[0]
        return predict, acc, predict_time
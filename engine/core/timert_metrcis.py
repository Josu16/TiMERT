import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score, balanced_accuracy_score

class TimertClassifierMetrics:
    def __init__(self, all_labels, all_preds, mlflow, class_names=None):
        """
        Inicializa la clase con las etiquetas verdaderas y las predicciones.
        
        :param all_labels: Array de etiquetas verdaderas.
        :param all_preds: Array de predicciones realizadas por el modelo.
        :param class_names: Lista opcional de nombres de clases.
        """
        self.all_labels = all_labels
        self.all_preds = all_preds
        self.mlflow = mlflow
        if class_names is None:
            self.class_names = np.unique(all_labels)
        else:
            self.class_names = class_names
        
        # Calcula todas las métricas al inicializar la clase
        self._calculate_metrics()
        
    def _calculate_metrics(self):
        """
        Calcula todas las métricas de clasificación y las almacena como atributos de la clase.
        """
        # Calcular métricas básicas
        self.accuracy = accuracy_score(self.all_labels, self.all_preds)
        self.balanced_accuracy = balanced_accuracy_score(self.all_labels, self.all_preds)
        
        # Precision, Recall, F1 Score por clase con manejo de divisiones por cero
        self.precision_per_class = precision_score(
            self.all_labels, self.all_preds, average=None, labels=self.class_names, zero_division=0)
        self.recall_per_class = recall_score(
            self.all_labels, self.all_preds, average=None, labels=self.class_names, zero_division=0)
        self.f1_per_class = f1_score(
            self.all_labels, self.all_preds, average=None, labels=self.class_names, zero_division=0)
        
        # Precision, Recall, F1 Score promediados (macro y micro) con manejo de divisiones por cero
        self.precision_macro = precision_score(
            self.all_labels, self.all_preds, average='macro', labels=self.class_names, zero_division=0)
        self.recall_macro = recall_score(
            self.all_labels, self.all_preds, average='macro', labels=self.class_names, zero_division=0)
        self.f1_macro = f1_score(
            self.all_labels, self.all_preds, average='macro', labels=self.class_names, zero_division=0)
        
        self.precision_micro = precision_score(
            self.all_labels, self.all_preds, average='micro', labels=self.class_names, zero_division=0)
        self.recall_micro = recall_score(
            self.all_labels, self.all_preds, average='micro', labels=self.class_names, zero_division=0)
        self.f1_micro = f1_score(
            self.all_labels, self.all_preds, average='micro', labels=self.class_names, zero_division=0)
        
        # Matriz de confusión con el orden de etiquetas especificado
        conf_matrix = confusion_matrix(self.all_labels, self.all_preds, labels=self.class_names)
        self.confusion_matrix = pd.DataFrame(conf_matrix, index=self.class_names, columns=self.class_names)
        
    def save_confusion_matrix(self, artifact_path='confusion_matrix.csv'): ## TODO: corregir.
        """
        Guarda la matriz de confusión en un archivo CSV.
        
        :param filepath: Ruta del archivo CSV donde se guardará la matriz de confusión.
        """
        # Guardar la matriz de confusión en el path especificado
        # self.confusion_matrix.to_csv(artifact_path)
        self.mlflow.log_artifact(artifact_path)
        
    def log_metrics_to_mlflow(self, prefix_param = "test_"):
        """
        Registra las métricas calculadas en MLflow.
        """
        
        # Registra métricas simples
        self.mlflow.log_metric(f'{prefix_param}accuracy', self.accuracy)
        self.mlflow.log_metric(f'{prefix_param}balanced_accuracy', self.balanced_accuracy)
        self.mlflow.log_metric(f'{prefix_param}precision_macro', self.precision_macro)
        self.mlflow.log_metric(f'{prefix_param}recall_macro', self.recall_macro)
        self.mlflow.log_metric(f'{prefix_param}f1_macro', self.f1_macro)
        self.mlflow.log_metric(f'{prefix_param}precision_micro', self.precision_micro)
        self.mlflow.log_metric(f'{prefix_param}recall_micro', self.recall_micro)
        self.mlflow.log_metric(f'{prefix_param}f1_micro', self.f1_micro)
        
        # Registra Precision, Recall y F1 Score por clase
        for i, class_name in enumerate(self.class_names):
            self.mlflow.log_metric(f'precision_class_{class_name}', self.precision_per_class[i])
            self.mlflow.log_metric(f'recall_class_{class_name}', self.recall_per_class[i])
            self.mlflow.log_metric(f'f1_class_{class_name}', self.f1_per_class[i])
        
        # # Guarda la matriz de confusión como artefacto
        # self.save_confusion_matrix('confusion_matrix.txt')
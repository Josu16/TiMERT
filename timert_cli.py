from engine.core.reference_models.timert_dtw import TimertDTW
import typer
import yaml
import mlflow
from mlflow.tracking import MlflowClient

from engine.timert_pretrain import TimertPreTrain
from engine.timert_finetuning import TimertFineTuning

app = typer.Typer()
app = typer.Typer(pretty_exceptions_short=False, pretty_exceptions_enable=False)


@app.command()
def dtw(
    conf_file: str = typer.Option(..., help="The configuration file name"),
):
    print(f"Selected File: {conf_file}")
    with open(f'parameters/{conf_file}.yml', 'r') as file:
        all_params = yaml.safe_load(file)
    
    mlflow.set_experiment("DTW")
    mlflow.enable_system_metrics_logging() # enable if need logs
    mlflow.set_system_metrics_sampling_interval(500)
    mlflow.start_run()
    print("MLFlow Started")
    try:
        timert_model = TimertDTW(params = all_params, mlflow=mlflow)
        timert_model.run_all_classifiers()
    finally:
        print("MLFlow Finished")
        mlflow.end_run()

@app.command()
def non_pretrain_classifier(
    conf_file: str = typer.Option(..., help="The configuration file name"),
    gpu_id: str = typer.Option("0", help="The GPU ID to use")
):
    print(f"Selected File: {conf_file}")
    print(f"Selected GPU: {gpu_id}")

    with open(f'parameters/{conf_file}.yml', 'r') as file:
        all_params = yaml.safe_load(file)
    
    mlflow.set_experiment("No-pretrain-classifier")
    mlflow.enable_system_metrics_logging() # enable if need logs
    mlflow.set_system_metrics_sampling_interval(600)
    mlflow.start_run()
    print("MLFlow Started")
    try:
        timert_model = TimertFineTuning(params = all_params, encoder = None, mlflow=mlflow, client_mlflow=MlflowClient(), gpu_id=gpu_id)
        timert_model.runn_all_models()
    finally:
        print("MLFlow Finished")
        mlflow.end_run()

@app.command()
def pretrain(
    conf_file: str = typer.Option(..., help="The configuration file name"),
    gpu_id: str = typer.Option("0", help="The GPU ID to use"),
    register: bool = typer.Option(False, help="Formally register the model")
):
    print(f"Selected File: {conf_file}")
    print(f"Selected GPU: {gpu_id}")
    
    print(f"Model will be registered") if register else print("Model will not be registered")

    with open(f'parameters/{conf_file}.yml', 'r') as file:
        all_params = yaml.safe_load(file)
    
    mlflow.set_experiment("Pre-train")
    mlflow.enable_system_metrics_logging() # enable if need logs
    mlflow.set_system_metrics_sampling_interval(600)
    mlflow.start_run()
    print("MLFlow Started")
    try:
        timer_model = TimertPreTrain(params = all_params, mlflow=mlflow, gpu_id=gpu_id)
        timer_model.preprocessing_time_series()
        timer_model.start_pretrain(client_mlflow=MlflowClient(), register = register)
    finally:
        print("MLFlow Finished")
        mlflow.end_run()

@app.command()
def finetuning(
    conf_file: str = typer.Option(..., help="The configuration file name"),
    gpu_id: str = typer.Option("0", help="The GPU ID to use"),
    enc_name: str = typer.Option(..., help="The name of pretrained model (consult MLflow)"),
    enc_ver: str = typer.Option(..., help="The version of pretrained model (consult MLflow)")
):
    print(f"Selected File: {conf_file}")
    print(f"Selected GPU: {gpu_id}")
    print(f"Model name: {gpu_id}")
    print(f"Selected GPU: {gpu_id}")


    with open(f'parameters/{conf_file}.yml', 'r') as file:
        all_params = yaml.safe_load(file)
    
    encoder = {
        "name": enc_name,
        "version": enc_ver
    }

    mlflow.set_experiment("Fine-tuning")
    mlflow.enable_system_metrics_logging() # enable if need logs
    mlflow.set_system_metrics_sampling_interval(500)
    mlflow.start_run()
    print("MLFlow Started")
    try:
        timert_model = TimertFineTuning(params = all_params, encoder= encoder, mlflow=mlflow, client_mlflow=MlflowClient(), gpu_id=gpu_id)
        timert_model.runn_all_models()
    finally:
        print("MLFlow Finished")
        mlflow.end_run()

if __name__ == "__main__":
    app()
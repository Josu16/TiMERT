import typer
import yaml
import mlflow

from engine.timert_pretrain import TimertPreTrain

app = typer.Typer()


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
    mlflow.set_system_metrics_sampling_interval(90)
    mlflow.start_run()
    print("MLFlow Started")
    try:
        timer_model = TimertPreTrain(params = all_params, mlflow=mlflow, gpu_id=gpu_id)
        timer_model.preprocessing_time_series()
        timer_model.start_pretrain(register = register)
    finally:
        print("MLFlow Finished")
        mlflow.end_run()


@app.command()
def finetuning(
    gpu_id: str = typer.Option("0", help="The GPU ID to use")
):
    print("Under construction")


if __name__ == "__main__":
    app()
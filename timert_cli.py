import typer
import yaml
import mlflow

from engine.timert_pretrain import TimertPreTrain

app = typer.Typer()


@app.command()
def pretrain(
    conf_file: str = typer.Option(..., help="The configuration file name"),
    gpu_id: str = typer.Option("0", help="The GPU ID to use")
):
    print(f"Selected File: {conf_file}")
    print(f"Selected GPU: {gpu_id}")

    with open(f'parameters/{conf_file}.yml', 'r') as file:
        all_params = yaml.safe_load(file)

    print(type(all_params))
    print(all_params)
    
    
    mlflow.start_run()
    
    try:
        timer_model = TimertPreTrain(params = all_params, mlflow=mlflow, gpu_id=gpu_id)
        timer_model.preprocessing_time_series()
        timer_model.start_pretrain()
    finally:
        mlflow.end_run()


@app.command()
def finetuning(
    gpu_id: str = typer.Option("0", help="The GPU ID to use")
):
    print("Under construction")


if __name__ == "__main__":
    app()
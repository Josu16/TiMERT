import typer

from engine.timert_pretrain import TimertPreTrain
from engine.data_models.pretrain_params_model import TimertPreTrainParams
from engine.data_models.global_params_model import TimertGlobalParams

app = typer.Typer()


@app.command()
def pretrain(
    gpu_id: str = typer.Option("0", help="The GPU ID to use")
):
    print(f"Selected GPU: {gpu_id}")
    
    global_params = TimertGlobalParams(
        data_dir = "UCRArchive_2018/",
        max_len = 512,
        seed = 666,
        pretrain_frac = 0.65,
        train_frac = 0.3,
        valid_frac = 0.1,
        test_frac = 0.1,
        is_same_length = True
        )

    model_params = TimertPreTrainParams(
        in_dim = 1,
        out_dim = 512,
        n_layer = 4,
        n_dim = 64,
        n_head = 8,
        norm_first = True,
        is_pos = True,
        is_projector = True,
        project_norm = "LN",
        dropout = 0.0,
        lr = 0.0001,
        batch_size = 128,
        n_epoch = 150,
        n_ckpt = 100,
        )

    timer_model = TimertPreTrain(
        model_params=model_params,
        global_params=global_params,
        gpu_id=gpu_id
        )

    timer_model.preprocessing_time_series()
    timer_model.start_pretrain()

@app.command()
def finetuning(
    gpu_id: str = typer.Option("0", help="The GPU ID to use")
):
    print("Under construction")


if __name__ == "__main__":
    app()
import typer

app = typer.Typer()


@app.command()
def pretrain(
    conf_file: str = typer.Option(..., help="The configuration file to use"),
    gpu_id: str = typer.Option("0", help="The GPU ID to use")
):
    print(f"Selected GPU: {gpu_id}")
    print(f"Config file: {conf_file}")
    from engine import timert_pretrain



if __name__ == "__main__":
    app()
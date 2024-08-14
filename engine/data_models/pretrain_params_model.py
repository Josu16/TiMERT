from dataclasses import dataclass

@dataclass
class TimertPreTrainParams:
    # Encoder
    in_dim: int = 1
    out_dim: int = 128
    n_layer: int = 4
    n_dim: int = 64
    n_head: int = 8
    norm_first: bool = True
    is_pos: bool = True
    is_projector: bool = True
    project_norm: str = "LN"
    dropout: float = 0.0

    # Train
    lr: float = 0.0001
    batch_size: int = 128
    n_epoch: int = 400
    n_ckpt: int = 100
from dataclasses import dataclass

@dataclass
class TimertGlobalParams:
    data_dir: str = "UCRArchive_2018/"
    max_len: int = 512
    seed: int = 666
    pretrain_frac: float = 0.5
    train_frac: float = 0.3
    valid_frac: float = 0.1
    test_frac: float = 0.1
    is_same_length: bool = True
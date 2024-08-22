from engine.core.pretext_tasks.augment_strategy import add_slope, add_spike, add_step, cropping, jittering, mag_warping, masking, shifting, smoothing, time_warping
from engine.core.pretext_tasks.custom_loss.ntx import NTXentLossPoly
from engine.core.pretext_tasks.mae_enc import MAEEncoder
from engine.core.pretext_tasks.timeclr_enc_0 import TimeCLREncoder
import torch.nn as nn
import torch

# Definición de estrategias
class MAELossStrategy:
    def __init__(self):
        self.loss_fun = nn.MSELoss()

    def compute_loss(self, model, data_batch):
        masked_data_batch, mask = model.forward(
            data_batch, normalize=False, to_numpy=False)
        return self.loss_fun(masked_data_batch[mask], data_batch[mask])
        
    def __str__(self):
        return f"Masked Autoencoder Loss Strategy: {self.loss_fun.__str__()}"

class TimeCLRStrategy:
    def __init__(self):
            self.loss_fun = NTXentLossPoly()

    def compute_loss(self, model, data_batch):
        ts_emb_aug_0 = model.forward(
            data_batch, normalize=False, to_numpy=False, is_augment=True)
        ts_emb_aug_1 = model.forward(
            data_batch, normalize=False, to_numpy=False, is_augment=True)
        return self.loss_fun(ts_emb_aug_0, ts_emb_aug_1)
    
    def __str__(self):
        return f"TF-C Loss Strategy: {self.loss_fun.__str__()}"


# Def de la factory, instancía el encoder y la función de pérdida.
class PretrainingFactory:
    @staticmethod
    def get_pretraining_strategy(model_config, encoder):
        method_name = model_config['pretexttask_name']

        if method_name == 'mae':
            mask_percent = float(model_config['mask_percent'])
            model = MAEEncoder(encoder, mask_percent=mask_percent)
            loss_strategy = MAELossStrategy()
        
        elif method_name == 'timeclr':
            aug_bank_ver = int(model_config['aug_bank_ver'])
            if aug_bank_ver == 0:
                aug_bank = [
                    lambda x: torch.tensor(jittering(x.cpu().numpy(), strength=0.1, seed=None)).to(x.device),
                    lambda x: torch.tensor(smoothing(x.cpu().numpy(), max_ratio=0.5, min_ratio=0.01, seed=None)).to(x.device),
                    lambda x: torch.tensor(mag_warping(x.cpu().numpy(), strength=1, seed=None)).to(x.device),
                    lambda x: torch.tensor(add_slope(x.cpu().numpy(), strength=1, seed=None)).to(x.device),
                    lambda x: torch.tensor(add_spike(x.cpu().numpy(), strength=3, seed=None)).to(x.device),
                    lambda x: torch.tensor(add_step(x.cpu().numpy(), min_ratio=0.1, strength=1, seed=None)).to(x.device),
                    lambda x: torch.tensor(cropping(x.cpu().numpy(), min_ratio=0.1, seed=None)).to(x.device),
                    lambda x: torch.tensor(masking(x.cpu().numpy(), max_ratio=0.5, seed=None)).to(x.device),
                    lambda x: torch.tensor(shifting(x.cpu().numpy(), seed=None)).to(x.device),
                    lambda x: torch.tensor(time_warping(x.cpu().numpy(), min_ratio=0.5, seed=None)).to(x.device),
                ]

            model = TimeCLREncoder(encoder, aug_bank)
            loss_strategy = TimeCLRStrategy()
        
        # Otros métodos...
        
        return model, loss_strategy


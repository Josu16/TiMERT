# Masked AutoEncoder
import torch
import torch.nn as nn
import numpy as np
import copy

class MAEEncoder(nn.Module):
    def __init__(self, encoder, mask_percent=0.15):
        r"""
        Masked Autoencoder (MAE) encoder, SSL technique described in BERT workd and explainded for
        time series in paper "Self-Supervised Learning for Time Analysis: Taxonomy, Progress, and
        Prospects".

        Args:
            encoder (Module): The base encoder.
            mask_percent (float, optional): Probability of masking a time step. Default: 0.15.

        Shape:
            - Input: :math:`(N, C_{in}, L_{in})`.
            - Output: :math:`(N, C_{out})`.
        """
        super(MAEEncoder, self).__init__()
        self.pretrain_name = 'mae'
        self.encoder = copy.deepcopy(encoder)
        self.mask_percent = mask_percent

        self.out_dim = self.encoder.out_dim
        self.dummy = nn.Parameter(torch.empty(0))

    def forward(self, ts, normalize=True, to_numpy=False):
        masked_ts, mask = self._mask_ts(ts)
        
        ts_emb = self.encoder.encode(
            masked_ts, normalize=normalize, to_numpy=to_numpy)
        ts_emb = torch.unsqueeze(ts_emb, dim=1)  # redimensionando para que sea comparable
        return ts_emb, mask

    def encode(self, ts, normalize=True, to_numpy=False):
        ts_emb, _ = self.forward(
            ts, normalize=normalize, to_numpy=to_numpy)
        return ts_emb

    def _mask_ts(self, data, mask_percent=0.15):
        mask = torch.full(data.shape, mask_percent, device=data.device)
        mask = torch.bernoulli(mask).bool()
        masked_data = data.clone()
        masked_data[mask] = 0  # Asignar cero a los valores enmascarados
        return masked_data, mask

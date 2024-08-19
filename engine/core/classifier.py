import torch
import torch.nn as nn
from collections import OrderedDict

class CustomTSClassifier(torch.nn.Module):
    def __init__(self, encoder, n_class, in_dim, n_dim = 64, n_layer = 2):
        super(CustomTSClassifier, self).__init__()
        self.encoder = encoder
        self.add_module('encoder', encoder)

        in_dim_ = in_dim
        out_dim_ = n_dim
        layers = OrderedDict()

        for i in range(n_layer - 1):
            layers[f'linear_{i:02d}'] = nn.Linear(in_dim_, out_dim_)
            layers[f'relu_{i:02d}'] = nn.ReLU()
            in_dim_ = out_dim_
            out_dim_ = n_dim

        layers[f'linear_{n_layer - 1:02d}'] = nn.Linear(in_dim_, n_class)
        self.classifier = nn.Sequential(layers)

    def forward(self, ts):
        transformer_rep = self.encoder(
            ts = ts,
            normalize = True,
            to_numpy = False
        )
        
        logits = self.classifier(transformer_rep)
        logits = logits.squeeze(1) # TODO: revisar por qué se mantiene esa dimensión.

        return logits
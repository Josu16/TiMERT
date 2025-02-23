import math
import torch
import torch.nn as nn
from collections import OrderedDict


def _normalize_t(t_, normalize):
    if not torch.is_tensor(t_):
        t_ = torch.from_numpy(t_)
    if len(t_.size()) == 1:
        t_ = torch.unsqueeze(t_, 0)
    if len(t_.size()) == 2:
        t_ = torch.unsqueeze(t_, 1)
    if normalize:
        t_mu = torch.mean(t_, 2, keepdims=True)
        t_sigma = torch.std(t_, 2, keepdims=True)
        t_sigma[t_sigma <= 0] = 1.0
        t_ = (t_ - t_mu) / t_sigma
    return t_



class Transformer(nn.Module):
    def __init__(self, in_dim=1, out_dim=128, n_layer=8, n_dim=64, n_head=8,
                 norm_first=False, is_pos=True, is_projector=True,
                 project_norm=None, dropout=0.0, learnable_pos=False):
        r"""
        Transformer-based time series encoder

        Args:
            in_dim (int, optional): Number of dimension for the input time
                series. Default: 1.
            out_dim (int, optional): Number of dimension for the output
                representation. Default: 128.
            n_layer (int, optional): Number of layer for the transformer
                encoder. Default: 8.
            n_dim (int, optional): Number of dimension for the intermediate
                representation. Default: 64.
            n_head (int, optional): Number of head for the transformer
                encoder. Default: 8.
            norm_first: if ``True``, layer norm is done prior to attention and
                feedforward operations, respectively. Otherwise it's done
                after. Default: ``False`` (after).
            is_pos (bool, optional): If set to ``False``, the encoder will
                not use position encoding. Default: ``True``.
            is_projector (bool, optional): If set to ``False``, the encoder
                will not use additional projection layers. Default: ``True``.
            project_norm (string, optional): If set to ``BN``, the projector
                will use batch normalization. If set to ``LN``, the projector
                will use layer normalization. If set to None, the projector
                will not use normalization. Default: None (no normalization).
            dropout (float, optional): The probability of an element to be
                zeroed for the dropout layers. Default: 0.0.

        Shape:
            - Input: :math:`(N, C_{in}, L_{in})`, :math:`(N, L_{in})`, or
                :math:`(L_{in})`.
            - Output: :math:`(N, C_{out})`.
        """
        super(Transformer, self).__init__()
        assert project_norm in ['BN', 'LN', None]

        self.in_dim = in_dim # dimensión de entrada
        self.out_dim = out_dim # dimensión de la salida
        self.n_dim = n_dim # the dimension of the feedforward network model (default=2048).
        self.is_projector = is_projector 
        self.is_pos = is_pos
        self.max_len = 0
        self.dropout = dropout
        self.learnable_pos = learnable_pos

        self.in_net = nn.Conv1d(
            in_dim, n_dim, 7, stride=2, padding=3, dilation=1)
        self.add_module('in_net', self.in_net)
        transformer = OrderedDict()
        for i in range(n_layer):
            transformer[f'encoder_{i:02d}'] = nn.TransformerEncoderLayer( # se definen todas las capas 
                n_dim, 
                n_head, 
                dim_feedforward=n_dim,
                dropout=dropout, 
                batch_first=True,
                norm_first=norm_first)
        self.transformer = nn.Sequential(transformer)

        self.start_token = nn.Parameter(
            torch.randn(1, n_dim, 1))
        self.register_parameter(
            name='start_token',
            param=self.start_token)

        self.out_net = nn.Linear(n_dim, out_dim)
        self.project_norm = project_norm
        if is_projector:
            if project_norm == 'BN':
                self.projector = nn.Sequential(
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU(),
                    nn.Linear(out_dim, out_dim * 2),
                    nn.BatchNorm1d(out_dim * 2),
                    nn.ReLU(),
                    nn.Linear(out_dim * 2, out_dim)
                )
            elif project_norm == 'LN':
                self.projector = nn.Sequential(
                    nn.ReLU(),
                    nn.LayerNorm(out_dim),
                    nn.Linear(out_dim, out_dim * 2),
                    nn.ReLU(),
                    nn.LayerNorm(out_dim * 2),
                    nn.Linear(out_dim * 2, out_dim)
                )
            else:
                self.projector = nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(out_dim, out_dim * 2),
                    nn.ReLU(),
                    nn.Linear(out_dim * 2, out_dim)
                )
        self.dummy = nn.Parameter(torch.empty(0))

    def forward(self, ts, normalize=True, to_numpy=False):
        device = self.dummy.device
        is_projector = self.is_projector
        is_pos = self.is_pos

        ts = _normalize_t(ts, normalize)
        ts = ts.to(device, dtype=torch.float32)

        ts_emb = self.in_net(ts) ## expande los canales a 64 y reduce d_model a la mitad
        if is_pos:
            n_dim = self.n_dim
            dropout = self.dropout
            ts_len = ts_emb.size()[2]
            if ts_len > self.max_len:
                self.max_len = ts_len
                if self.learnable_pos:
                    self.pos_net = LearnablePositionalEncoding(
                        n_dim, ts_len, dropout=dropout)
                    num_params = sum(p.numel() for p in self.pos_net.parameters())
                    print(f"El PE tiene {num_params} parámetros.")
                else:
                    self.pos_net = PositionalEncoding(
                        n_dim, ts_len, dropout=dropout)
                self.pos_net.to(device)
            ts_emb = self.pos_net(ts_emb)

        start_tokens = self.start_token.expand(ts_emb.size()[0], -1, -1)
        ts_emb = torch.cat((start_tokens, ts_emb, ), dim=2)
        ts_emb = torch.transpose(ts_emb, 1, 2)

        ts_emb = self.transformer(ts_emb)
        ts_emb = ts_emb[:, 0, :]
        ts_emb = self.out_net(ts_emb)

        if is_projector:
            ts_emb = self.projector(ts_emb)

        if to_numpy:
            return ts_emb.cpu().detach().numpy()
        else:
            return ts_emb

    def encode(self, ts, normalize=True, to_numpy=False):
        return self.forward(ts, normalize=normalize, to_numpy=to_numpy)

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024, dropout=0.1):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(1, d_model, max_len))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch_size, embed dim, max_len]
            output: [batch_size, embed dim, max_len]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class PositionalEncoding(nn.Module):
    def __init__(self, n_dim, max_len, dropout=0.0):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len)
        div_term = torch.exp(
            torch.arange(0, n_dim, 2) * (-math.log(10000.0) / n_dim))
        pos_emb = torch.zeros(1, n_dim, max_len) # 1 x 64 x 256

        position = position.unsqueeze(0)
        div_term = div_term.unsqueeze(1)
        pos_emb[0, 0::2, :] = torch.sin(div_term * position)
        pos_emb[0, 1::2, :] = torch.cos(div_term * position)
        self.register_buffer('pos_emb', pos_emb, persistent=False)

    def forward(self, x):
        x = x + self.pos_emb[:, :, :x.size()[2]] ## batch_sizex64x256 + 1x64x256 : pytorch do broadcasting
        return self.dropout(x)


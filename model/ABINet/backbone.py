import torch
import torch.nn as nn

from model.ABINet.resnet import resnet45
from model.ABINet.transformer import (PositionalEncoding,
                        TransformerEncoder,
                         TransformerEncoderLayer)

_default_tfmer_cfg = dict(d_model=512, nhead=8, d_inner=2048,  # 1024
                          dropout=0.1, activation='relu',num_layers=3)


class ResTranformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.resnet = resnet45()

        self.d_model = _default_tfmer_cfg['d_model']
        nhead = _default_tfmer_cfg['nhead']
        d_inner = _default_tfmer_cfg['d_inner']
        dropout =  _default_tfmer_cfg['dropout']
        activation = _default_tfmer_cfg['activation']
        num_layers = _default_tfmer_cfg['num_layers']

        self.pos_encoder = PositionalEncoding(self.d_model, max_len=8*32)
        encoder_layer = TransformerEncoderLayer(d_model=self.d_model, nhead=nhead,
                dim_feedforward=d_inner, dropout=dropout, activation=activation)
        self.transformer = TransformerEncoder(encoder_layer, num_layers)

    def forward(self, images,label_strs):
        # Resnet45 + Transformer Encoder
        # 相当于用一个小的Resnet+Trans 来替代原来大的backbone Resnet
        feature = self.resnet(images,label_strs)
        n, c, h, w = feature.shape
        feature = feature.view(n, c, -1).permute(2, 0, 1)
        feature = self.pos_encoder(feature) # add PE
        feature = self.transformer(feature) # encoder
        feature = feature.permute(1, 2, 0).view(n, c, h, w)
        return feature



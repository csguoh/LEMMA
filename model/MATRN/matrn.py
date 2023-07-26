import torch
import torch.nn as nn
from model.ABINet.abinet import BaseVision,BaseAlignment,BCNLanguage,Model
from model.MATRN.sematic_visual_backbone import BaseSemanticVisual_backbone_feature
import torch.nn.functional as F
import logging

class MATRN(Model):
    def __init__(self, config):
        super().__init__(config)
        self.iter_size = 3
        self.test_bh = None
        self.vision = BaseVision(config)
        self.language = BCNLanguage(config)
        self.semantic_visual = BaseSemanticVisual_backbone_feature(config)
        self.max_length = config.dataset_max_length + 1  # additional stop token
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])
        # load full model--> Vision Language Align
        if config.full_ckpt is not None:
            logging.info(f'Read full ckpt model from {config.full_ckpt}.')
            self.load(config.full_ckpt)


    def forward(self, images,input_lr=False,normalize=True):
        device = images.device
        if images.shape[2] == 16:
            images = F.interpolate(images, scale_factor=2, mode='bicubic', align_corners=True)
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        if normalize:
            images =(images-self.mean[..., None, None] ) /  self.std[..., None, None]

        v_res = self.vision(images)
        a_res = v_res
        for _ in range(self.iter_size):
            tokens = torch.softmax(a_res['logits'], dim=-1)
            lengths = a_res['pt_lengths']
            lengths.clamp_(2, self.max_length)
            l_res = self.language(tokens, lengths)

            lengths_l = l_res['pt_lengths']
            lengths_l.clamp_(2, self.max_length)

            v_attn_input = v_res['attn_scores'].clone().detach()
            l_logits_input = None
            texts_input = None

            a_res = self.semantic_visual(l_res['feature'], v_res['backbone_feature'], lengths_l=lengths_l, v_attn=v_attn_input, l_logits=l_logits_input, texts=texts_input, training=self.training)

        # TODO 和ABINet一样，这里Matrn直接把logits进行了互换
        v_res['logits'] = a_res['logits']

        return v_res




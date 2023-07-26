import logging
import torch
from model.ABINet.transformer import (PositionalEncoding,
                         MultiheadAttention,
                         TransformerDecoder,
                         TransformerDecoderLayer,
                         TransformerEncoderLayer,
                         TransformerEncoder)

from setup import CharsetMapper
from model.ABINet.attention import *
from model.ABINet.backbone import ResTranformer
from model.ABINet.resnet import resnet45
from torch.nn import functional as F


_default_tfmer_cfg = dict(d_model=512, nhead=8, d_inner=2048,  # 1024
                          dropout=0.1, activation='relu')



class Model(nn.Module):
    # vision model 和 LM 的基类--共同的单词表和load dict
    def __init__(self, config):
        super().__init__()
        self.max_length = config.dataset_max_length + 1
        self.charset = CharsetMapper(config.dataset_charset_path, max_length=self.max_length)

    def load(self, source, device=None, strict=True):
        state = torch.load(source, map_location=device)
        self.load_state_dict(state['model'], strict=strict)

    def _get_length(self, logit, dim=-1):
        """ Greed decoder to obtain length from logit"""
        out = (logit.argmax(dim=-1) == self.charset.null_label)
        abn = out.any(dim)
        out = ((out.cumsum(dim) == 1) & out).max(dim)[1]
        out = out + 1  # additional end token
        out = torch.where(abn, out, out.new_tensor(logit.shape[1]))
        return out

    @staticmethod
    def _get_padding_mask(length, max_length):
        length = length.unsqueeze(-1)
        grid = torch.arange(0, max_length, device=length.device).unsqueeze(0)
        return grid >= length

    @staticmethod
    def _get_square_subsequent_mask(sz, device, diagonal=0, fw=True):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz, device=device), diagonal=diagonal) == 1)
        if fw: mask = mask.transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    @staticmethod
    def _get_location_mask(sz, device=None):
        mask = torch.eye(sz, device=device)
        mask = mask.float().masked_fill(mask == 1, float('-inf'))  # 将对角线处的注意力置为-inf
        return mask



class BaseVision(Model):
    def __init__(self, config):
        super().__init__(config)
        self.out_channels = config.vision.d_model if config.vision.d_model is not None else 512

        # ====== Backbone-- Resnet45+Encoder  for  Feature Exaction=============
        if config.vision.backbone == 'transformer':
            self.backbone = ResTranformer(config)
        else: self.backbone = resnet45()
        # =============== get attn map ======================================
        if config.vision.attention == 'position':
            mode = config.model_vision_attention_mode if config.model_vision_attention_mode is not None else 'nearest'
            self.attention = PositionAttention(
                max_length=config.dataset_max_length + 1,  # additional stop token
                mode=mode,
            )
        elif config.vision.attention == 'attention':
            self.attention = Attention(
                max_length=config.dataset_max_length + 1,  # additional stop token
                n_feature=8*32,
            )

        self.cls = nn.Linear(self.out_channels, self.charset.num_classes)

        if config.vision.checkpoint is not None:
            logging.info(f'Read vision model from {config.vision.checkpoint}.')
            self.load(config.vision.checkpoint)


    def forward(self, images, label_strs=None,*args):
        features = self.backbone(images,label_strs)  # (N, E, H, W)
        attn_vecs, attn_scores = self.attention(features)  # (N, T, E), (N, T, H, W)
        logits = self.cls(attn_vecs) # (N, T, C) 对加权后的V过一层linear
        pt_lengths = self._get_length(logits)
        # 这里的backbonefeature由于是小的特征图，因此没法用。 只能用到概率分布的logits（其实用的是它的经过LM的版本）和atten scores
        return {'feature': attn_vecs, 'logits': logits, 'pt_lengths': pt_lengths,
                'attn_scores': attn_scores, 'name': 'vision',
                'backbone_feature': features}


class BCNLanguage(Model):
    def __init__(self, config):
        super().__init__(config)
        d_model = _default_tfmer_cfg['d_model']
        nhead = _default_tfmer_cfg['nhead']
        d_inner = _default_tfmer_cfg['d_inner']
        dropout = _default_tfmer_cfg['dropout']
        activation = _default_tfmer_cfg['activation']
        num_layers = 4
        self.d_model = d_model
        self.detach = config.language.detach if config.language.detach is not None else True
        self.use_self_attn = config.language.use_self_attn if config.language.use_self_attn is not None else False
        self.max_length = config.dataset_max_length + 1  # additional stop token
        self.debug = False

        self.proj = nn.Linear(self.charset.num_classes, d_model, False)
        self.token_encoder = PositionalEncoding(d_model, max_len=self.max_length)
        self.pos_encoder = PositionalEncoding(d_model, dropout=0, max_len=self.max_length)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, d_inner, dropout,
                activation, self_attn=self.use_self_attn, debug=self.debug)
        self.model = TransformerDecoder(decoder_layer, num_layers)

        self.cls = nn.Linear(d_model, self.charset.num_classes)

        if config.language.checkpoint is not None:
            logging.info(f'Read language model from {config.language.checkpoint}.')
            self.load(config.language.checkpoint)

    def forward(self, tokens, lengths):
        """
        Args:
            tokens: (N, T, C) where T is length, N is batch size and C is classes number
            lengths: (N,)
            思路：先对tokens这一分布做特征映射，把他作为decoder的memory。
            再使用PE作为decoder的query，两者经过一个dcoder的输出就是修正后的分布。
            注意的是两个decoder的输入是长度一致的相同模态
        """
        if self.detach: tokens = tokens.detach() # 将概率logits给切断梯度
        embed = self.proj(tokens)  # (N, T, E) 重新把概率分布映射为特征
        embed = embed.permute(1, 0, 2)  # (T, N, E)
        embed = self.token_encoder(embed)  # (T, N, E) 变成transformer的输入形式并在特征上加入PE
        padding_mask = self._get_padding_mask(lengths, self.max_length)

        zeros = embed.new_zeros(*embed.shape)
        qeury = self.pos_encoder(zeros)
        location_mask = self._get_location_mask(self.max_length, tokens.device)
        output = self.model(qeury, embed,
                tgt_key_padding_mask=padding_mask,
                memory_mask=location_mask,
                memory_key_padding_mask=padding_mask)  # (T, N, E)
        output = output.permute(1, 0, 2)  # (N, T, E)

        logits = self.cls(output)  # (N, T, C) 最后修正后的分布
        pt_lengths = self._get_length(logits)

        res =  {'feature': output, 'logits': logits, 'pt_lengths': pt_lengths,'name': 'language'}
        return res


def encoder_layer(in_c, out_c, k=3, s=2, p=1):
    return nn.Sequential(nn.Conv2d(in_c, out_c, k, s, p),
                         nn.BatchNorm2d(out_c),
                         nn.ReLU(True))


def decoder_layer(in_c, out_c, k=3, s=1, p=1, mode='nearest', scale_factor=None, size=None):
    align_corners = None if mode=='nearest' else True
    return nn.Sequential(nn.Upsample(size=size, scale_factor=scale_factor,
                                     mode=mode, align_corners=align_corners),
                         nn.Conv2d(in_c, out_c, k, s, p),
                         nn.BatchNorm2d(out_c),
                         nn.ReLU(True))


class PositionAttention(nn.Module):
    def __init__(self, max_length, in_channels=512, num_channels=64,
                 h=8, w=32, mode='nearest', **kwargs):
        super().__init__()
        self.max_length = max_length # len of alphbet -- 26
        self.k_encoder = nn.Sequential(
            encoder_layer(in_channels, num_channels, s=(1, 2)),
            encoder_layer(num_channels, num_channels, s=(2, 2)),
            encoder_layer(num_channels, num_channels, s=(2, 2)),
            encoder_layer(num_channels, num_channels, s=(2, 2))
        ) # conv - bn - relu
        self.k_decoder = nn.Sequential(
            decoder_layer(num_channels, num_channels, scale_factor=2, mode=mode),
            decoder_layer(num_channels, num_channels, scale_factor=2, mode=mode),
            decoder_layer(num_channels, num_channels, scale_factor=2, mode=mode),
            decoder_layer(num_channels, in_channels, size=(h, w), mode=mode)  )# upsample - conv - bn -relu
        self.pos_encoder = PositionalEncoding(in_channels, dropout=0, max_len=max_length)
        self.project = nn.Linear(in_channels, in_channels) # 对位置编码  过一层Linear

    def forward(self, x, q=None):
        # x is img feat
        # q is text PE
        N, E, H, W = x.size()
        k, v = x, x  # (N, E, H, W)

        # calculate key vector from x
        # mini U-Net
        features = []
        for i in range(0, len(self.k_encoder)):
            k = self.k_encoder[i](k)
            features.append(k)
        for i in range(0, len(self.k_decoder) - 1):
            k = self.k_decoder[i](k)
            k = k + features[len(self.k_decoder) - 2 - i]
        k = self.k_decoder[-1](k)

        # calculate query vector
        if q is None:
            zeros = x.new_zeros((self.max_length, N, E))  # (T, N, E)
            q = self.pos_encoder(zeros)  # (T, N, E)
            q = q.permute(1, 0, 2)  # (N, T, E)
        q = self.project(q)  # (N, T, E)

        # calculate attention
        attn_scores = torch.bmm(q, k.flatten(2, 3))  # (N, T, (H*W))
        attn_scores = attn_scores / (E ** 0.5)
        attn_scores = torch.softmax(attn_scores, dim=-1)

        v = v.permute(0, 2, 3, 1).view(N, -1, E)  # (N, (H*W), E)
        attn_vecs = torch.bmm(attn_scores, v)  # (N, T, E)

        return attn_vecs, attn_scores.view(N, -1, H, W)



class BaseAlignment(Model):
    def __init__(self, config):
        super().__init__(config)
        d_model =  _default_tfmer_cfg['d_model']

        self.loss_weight = 1.0
        self.max_length = config.dataset_max_length + 1
        self.w_att = nn.Linear(2 * d_model, d_model)
        self.cls = nn.Linear(d_model, self.charset.num_classes)

    def forward(self, l_feature, v_feature):
        """
        Args:
            l_feature: (N, T, E) where T is length, N is batch size and d is dim of model
            v_feature: (N, T, E) shape the same as l_feature
            l_lengths: (N,)
            v_lengths: (N,)
        """
        f = torch.cat((l_feature, v_feature), dim=2)
        f_att = torch.sigmoid(self.w_att(f))
        output = f_att * v_feature + (1 - f_att) * l_feature

        logits = self.cls(output)  # (N, T, C)
        pt_lengths = self._get_length(logits)

        return {'logits': logits, 'pt_lengths': pt_lengths, 'loss_weight':self.loss_weight,
                'name': 'alignment'}


class ABINet_model(Model):
    def __init__(self,config):
        super().__init__(config)
        self.vision = BaseVision(config)
        self.language = BCNLanguage(config)
        self.alignment = BaseAlignment(config)
        self.max_length = config.dataset_max_length + 1  # additional stop token
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])


        if config.full_ckpt is not None:
            logging.info(f'Read full ckpt model from {config.full_ckpt}.')
            self.load(config.full_ckpt)


    def logits_to_string(self,output):
        # 输入是文本识别结果的logits，输出是字符串列表
        # {'feature': attn_vecs, 'logits': logits, 'pt_lengths': pt_lengths,'attn_scores': attn_scores, 'name': 'vision',' backbone_feature': features}
        logit = output['logits']#.cpu()
        out = F.softmax(logit, dim=2)
        pt_text, pt_scores, pt_lengths = [], [], []
        for o in out:
            text = self.charset.get_text(o.argmax(dim=1), padding=False, trim=False)
            text = text.split(self.charset.null_char)[0]  # end at end-token 本质上也是实现了trim的功能
            pt_text.append(text)
            pt_scores.append(o.max(dim=1)[0])
            pt_lengths.append(min(len(text) + 1, self.charset.max_length))  # one for end-token
        return pt_text, pt_scores, pt_lengths


    def forward(self, images,label_strs=None,normalize=True):
        device = images.device
        if images.shape[2]==16:
            images = F.interpolate(images, scale_factor=2, mode='bicubic', align_corners=True)
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        if normalize:
            images = (images-self.mean[..., None, None]) / self.std[..., None, None]
        v_res = self.vision(images,label_strs)
        # new here!
        v_tokens = torch.softmax(v_res['logits'], dim=-1)
        v_lengths = v_res['pt_lengths'].clamp_(2, self.max_length)
        l_res = self.language(v_tokens, v_lengths)
        l_feature, v_feature = l_res['feature'], v_res['feature']
        a_res = self.alignment(l_feature, v_feature)
        return a_res














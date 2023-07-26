import cv2
import sys
import time
import torch
import string
import random
import numpy as np
import torch.nn as nn
import torchvision.transforms

from utils import util, ssim_psnr
from setup import CharsetMapper
# ce_loss = torch.nn.CrossEntropyLoss()
import torch
import torch.nn.functional as F
import math, copy
import numpy as np
from torch.autograd import Variable
import torch
import pickle as pkl
import torchvision.transforms as TF

alphabet = '-0123456789abcdefghijklmnopqrstuvwxyz'
standard_alphebet = '-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
standard_dict = {}
for index in range(len(standard_alphebet)):
    standard_dict[standard_alphebet[index]] = index

def load_confuse_matrix():
    f = open('./dataset/confuse.pkl', 'rb')
    data = pkl.load(f)
    f.close()
    number = data[:10]
    upper = data[10:36]
    lower = data[36:]
    end = np.ones((1,62))
    pad = np.ones((63,1))
    rearrange_data = np.concatenate((end, number, lower, upper), axis=0)
    rearrange_data = np.concatenate((pad, rearrange_data), axis=1)
    rearrange_data = 1 / rearrange_data
    rearrange_data[rearrange_data==np.inf] = 1
    rearrange_data = torch.Tensor(rearrange_data).cuda()

    lower_alpha = 'abcdefghijklmnopqrstuvwxyz'
    # upper_alpha = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i in range(63):
        for j in range(63):
            if i != j and standard_alphebet[j] in lower_alpha:
                rearrange_data[i][j] = max(rearrange_data[i][j], rearrange_data[i][j+26])
    rearrange_data = rearrange_data[:37,:37]

    return rearrange_data

weight_table = load_confuse_matrix()

def weight_cross_entropy(pred, gt):
    global weight_table
    batch = gt.shape[0]
    weight = weight_table[gt]
    pred_exp = torch.exp(pred)
    pred_exp_weight = weight * pred_exp
    loss = 0
    for i in range(len(gt)):
        loss -= torch.log(pred_exp_weight[i][gt[i]] / torch.sum(pred_exp_weight, 1)[i])
    return loss / batch

def get_alphabet_len():
    return len(alphabet)

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None, attention_map=None):
    "Compute 'Scaled Dot Product Attention'"
    # [N, W, H, C]
    if attention_map is not None:
        return torch.matmul(attention_map, value), attention_map

    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    else:
        pass

    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, compress_attention=False):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.compress_attention = compress_attention
        self.compress_attention_linear = nn.Linear(h, 1)

    def forward(self, query, key, value, mask=None, attention_map=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attention_map = attention(query, key, value, mask=mask,
                                     dropout=self.dropout, attention_map=attention_map)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x), attention_map


class ResNet(nn.Module):
    def __init__(self, num_in, block, layers):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(num_in, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d((2, 2), (2, 2))

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)

        self.layer1_pool = nn.MaxPool2d((2, 2), (2, 2))
        self.layer1 = self._make_layer(block, 128, 256, layers[0])
        self.layer1_conv = nn.Conv2d(256, 256, 3, 1, 1)
        self.layer1_bn = nn.BatchNorm2d(256)
        self.layer1_relu = nn.ReLU(inplace=True)

        self.layer2_pool = nn.MaxPool2d((2, 2), (2, 2))
        self.layer2 = self._make_layer(block, 256, 256, layers[1])
        self.layer2_conv = nn.Conv2d(256, 256, 3, 1, 1)
        self.layer2_bn = nn.BatchNorm2d(256)
        self.layer2_relu = nn.ReLU(inplace=True)

        self.layer3_pool = nn.MaxPool2d((2, 2), (2, 2))
        self.layer3 = self._make_layer(block, 256, 512, layers[2])
        self.layer3_conv = nn.Conv2d(512, 512, 3, 1, 1)
        self.layer3_bn = nn.BatchNorm2d(512)
        self.layer3_relu = nn.ReLU(inplace=True)

        self.layer4_pool = nn.MaxPool2d((2, 2), (2, 2))
        self.layer4 = self._make_layer(block, 512, 512, layers[3])
        self.layer4_conv2 = nn.Conv2d(512, 1024, 3, 1, 1)
        self.layer4_conv2_bn = nn.BatchNorm2d(1024)
        self.layer4_conv2_relu = nn.ReLU(inplace=True)

    def _make_layer(self, block, inplanes, planes, blocks):

        if inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 3, 1, 1),
                nn.BatchNorm2d(planes), )
        else:
            downsample = None
        layers = []
        layers.append(block(inplanes, planes, downsample))
        for i in range(1, blocks):
            layers.append(block(planes, planes, downsample=None))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.layer1_pool(x)
        x = self.layer1(x)
        x = self.layer1_conv(x)
        x = self.layer1_bn(x)
        x = self.layer1_relu(x)

        # x = self.layer2_pool(x)
        x = self.layer2(x)
        x = self.layer2_conv(x)
        x = self.layer2_bn(x)
        x = self.layer2_relu(x)

        # x = self.layer3_pool(x)
        x = self.layer3(x)
        x = self.layer3_conv(x)
        x = self.layer3_bn(x)
        x = self.layer3_relu(x)

        # x = self.layer4_pool(x)
        x = self.layer4(x)
        x = self.layer4_conv2(x)
        x = self.layer4_conv2_bn(x)
        x = self.layer4_conv2_relu(x)

        return x


class Bottleneck(nn.Module):

    def __init__(self, input_dim):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, input_dim, 1)
        self.bn1 = nn.BatchNorm2d(input_dim)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(input_dim, input_dim, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(input_dim)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.se(out)

        out += residual
        out = self.relu(out)

        return out


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # print(features)
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        self.relu = nn.ReLU()

    def forward(self, x):
        # return F.softmax(self.proj(x))
        return self.proj(x)


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        embed = self.lut(x) * math.sqrt(self.d_model)
        # print("embed",embed)
        # embed = self.lut(x)
        # print(embed.requires_grad)
        return embed


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        self.mask_multihead = MultiHeadedAttention(h=16, d_model=1024, dropout=0.1)
        self.mul_layernorm1 = LayerNorm(features=1024)

        self.multihead = MultiHeadedAttention(h=16, d_model=1024, dropout=0.1, compress_attention=True)
        self.mul_layernorm2 = LayerNorm(features=1024)

        self.pff = PositionwiseFeedForward(1024, 2048)
        self.mul_layernorm3 = LayerNorm(features=1024)

    def forward(self, text, conv_feature, attention_map=None):
        text_max_length = text.shape[1]
        mask = subsequent_mask(text_max_length).cuda()

        result = text
        result = self.mul_layernorm1(result + self.mask_multihead(result, result, result, mask=mask)[0])

        b, c, h, w = conv_feature.shape
        conv_feature = conv_feature.view(b, c, h * w).permute(0, 2, 1).contiguous()
        # result--[bs, Seq, C]  conv_feature--[bs, HW, C]
        word_image_align, attention_map = self.multihead(result, conv_feature, conv_feature, mask=None, attention_map=attention_map)
        result = self.mul_layernorm2(result + word_image_align)
        result = self.mul_layernorm3(result + self.pff(result))

        return result, attention_map


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, downsample):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.se(out)

        if self.downsample != None:
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)

        return out


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.cnn = ResNet(num_in=1, block=BasicBlock, layers=[1, 2, 5, 3])

    def forward(self, input):
        conv_result = self.cnn(input)
        return conv_result


class Transformer(nn.Module):

    def __init__(self):
        super(Transformer, self).__init__()

        word_n_class = get_alphabet_len() # 37
        self.embedding_word = Embeddings(512, word_n_class) # to embedding a word index to a word feature
        self.pe = PositionalEncoding(d_model=512, dropout=0.1, max_len=5000)

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.generator_word = Generator(1024, word_n_class)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, image, text_length, text_input, test=False, attention_map=None):
        # Encoder
        conv_feature = self.encoder(image) # batch, 1024, 8, 32
        text_embedding = self.embedding_word(text_input) # batch, text_max_length, 512
        postion_embedding = self.pe(torch.zeros(text_embedding.shape).cuda()).cuda() # batch, text_max_length, 512
        text_input_with_pe = torch.cat([text_embedding, postion_embedding], 2) # batch, text_max_length, 1024
        batch, seq_len, _ = text_input_with_pe.shape
        # output of Encoder + [PE, text_embedding] --> Decoder
        text_input_with_pe, word_attention_map = self.decoder(text_input_with_pe, conv_feature, attention_map=attention_map)
        word_decoder_result = self.generator_word(text_input_with_pe)
        total_length = torch.sum(text_length).data
        probs_res = torch.zeros(total_length, get_alphabet_len()).type_as(word_decoder_result.data)

        start = 0
        for index, length in enumerate(text_length):
            # print("index, length", index,length)
            length = length.data
            probs_res[start:start + length, :] = word_decoder_result[index, 0:0 + length, :]
            start = start + length

        if not test:
            return probs_res, word_attention_map, None
        else:
            # return word_decoder_result, word_attention_map, text_input_with_pe
            return word_decoder_result



def to_gray_tensor(tensor):
    R = tensor[:, 0:1, :, :]
    G = tensor[:, 1:2, :, :]
    B = tensor[:, 2:3, :, :]
    tensor = 0.299 * R + 0.587 * G + 0.114 * B
    return tensor


def str_filt(str_, voc_type):
    alpha_dict = {
        'digit': string.digits,
        'lower': string.digits + string.ascii_lowercase,
        'upper': string.digits + string.ascii_letters,
        'all': string.digits + string.ascii_letters + string.punctuation
    }
    if voc_type == 'lower':
        str_ = str_.lower()
    for char in str_:
        if char not in alpha_dict[voc_type]:
            str_ = str_.replace(char, '')
    str_ = str_.lower()
    return str_

class TextFocusLossSTT(nn.Module):
    def __init__(self, args):
        super(TextFocusLossSTT, self).__init__()
        self.args = args
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.l1_loss = nn.L1Loss()
        self.english_alphabet = '-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.english_dict = {}
        for index in range(len(self.english_alphabet)):
            self.english_dict[self.english_alphabet[index]] = index

        self.build_up_transformer()

    def build_up_transformer(self):

        transformer = Transformer()
        transformer.load_state_dict(torch.load(self.args.pretrained_trans))
        transformer.eval()
        transformer = nn.DataParallel(transformer, device_ids=range(self.args.ngpu))
        self.transformer = transformer

    def label_encoder(self, label):
        batch = len(label)

        length = [len(i) for i in label]
        length_tensor = torch.Tensor(length).long().cuda()

        max_length = max(length)
        input_tensor = np.zeros((batch, max_length))
        for i in range(batch):
            for j in range(length[i] - 1):
                input_tensor[i][j + 1] = self.english_dict[label[i][j]]

        text_gt = []
        for i in label:
            for j in i:
                text_gt.append(self.english_dict[j])
        text_gt = torch.Tensor(text_gt).long().cuda()

        input_tensor = torch.from_numpy(input_tensor).long().cuda()
        return length_tensor, input_tensor, text_gt


    def forward(self,sr_img, hr_img, label):

        mse_loss = self.mse_loss(sr_img, hr_img)

        if self.args.text_focus:
            label = [str_filt(i, 'lower')+'-' for i in label] # to lower-case
            length_tensor, input_tensor, text_gt = self.label_encoder(label)
            hr_pred, word_attention_map_gt, hr_correct_list = self.transformer(to_gray_tensor(hr_img), length_tensor,
                                                                          input_tensor, test=False)
            sr_pred, word_attention_map_pred, sr_correct_list = self.transformer(to_gray_tensor(sr_img), length_tensor,
                                                                            input_tensor, test=False)
            attention_loss = self.l1_loss(word_attention_map_gt, word_attention_map_pred)
            # recognition_loss = self.l1_loss(hr_pred, sr_pred)
            recognition_loss = weight_cross_entropy(sr_pred, text_gt)
            loss = mse_loss + attention_loss * 10 + recognition_loss * 0.0005
            return loss, mse_loss, attention_loss, recognition_loss
        else:
            attention_loss = -1
            recognition_loss = -1
            loss = mse_loss
            return loss, mse_loss, attention_loss, recognition_loss



class TextFocusLoss(nn.Module):
    def __init__(self, cfg):
        super(TextFocusLoss, self).__init__()
        self.cfg = cfg
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.l1_loss = nn.L1Loss()
        self.ssim = ssim_psnr.SSIM()
        self.kl_Loss = torch.nn.KLDivLoss(reduction='batchmean')
        self.augmentations = TF.Compose(
            [TF.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5)]
        )
        self.build_up_transformer()

    def build_up_transformer(self):
        transformer = Transformer()
        transformer.load_state_dict(torch.load(self.cfg.pretrained_trans))
        transformer.eval()
        transformer = nn.DataParallel(transformer,device_ids=range(self.cfg.ngpu))
        self.transformer = transformer

    def label_encoder(self, label):
        batch = len(label)

        length = [len(i) for i in label]
        length_tensor = torch.Tensor(length).long().cuda()

        max_length = max(length)
        input_tensor = np.zeros((batch, max_length))
        for i in range(batch):
            for j in range(length[i] - 1):
                input_tensor[i][j + 1] = self.english_dict[label[i][j]]

        text_gt = []
        for i in label:
            for j in i:
                text_gt.append(self.english_dict[j])
        text_gt = torch.Tensor(text_gt).long().cuda()

        input_tensor = torch.from_numpy(input_tensor).long().cuda()
        return length_tensor, input_tensor, text_gt


    def forward(self,sr_img, hr_img, label_tensor, distill_res):
        label_id, length_tensor, input_tensor, text_gt = label_tensor
        mse_loss = self.mse_loss(sr_img, hr_img)

        hr_pred, word_attention_map_gt, hr_correct_list = self.transformer(to_gray_tensor(hr_img), length_tensor,
                                                                      input_tensor, test=False)
        sr_pred, word_attention_map_pred, sr_correct_list = self.transformer(to_gray_tensor(sr_img), length_tensor,
                                                                        input_tensor, test=False)
        attention_loss = self.l1_loss(word_attention_map_gt, word_attention_map_pred)
        recognition_loss = weight_cross_entropy(sr_pred, text_gt)

        finetune_loss_v = self.ce_loss(F.softmax(distill_res['img_logits'],-1).view(-1,37),label_id.view(-1))
        finetune_loss_t = self.ce_loss(F.softmax(distill_res['text_logits'],-1).view(-1,37),label_id.view(-1))
        finetune_loss = finetune_loss_v + finetune_loss_t


        loss = (1 * mse_loss) + 0.5 * (attention_loss * 10 + recognition_loss * 0.0005) + finetune_loss * 0.01
        return loss, mse_loss, attention_loss, recognition_loss, finetune_loss

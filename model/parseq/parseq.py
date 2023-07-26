import math
from functools import partial
from itertools import permutations
from typing import Sequence, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from timm.models.helpers import named_apply
from model.parseq.modules import DecoderLayer, Decoder, Encoder, TokenEmbedding
import logging
from model.parseq.parseq_tokenizer import get_parseq_tokenize
from torchvision import transforms


def init_weights(module: nn.Module, name: str = '', exclude: Sequence[str] = ()):
    """Initialize the weights using the typical initialization schemes used in SOTA models."""
    if any(map(name.startswith, exclude)):
        return
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

class PARSeq(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.tokenizer = get_parseq_tokenize()

        img_size= config.img_size
        patch_size= config.patch_size
        embed_dim=config.embed_dim
        enc_depth=config.enc_depth
        enc_num_heads=config.enc_num_heads
        enc_mlp_ratio=config.enc_mlp_ratio

        self.max_label_length = 25
        self.decode_ar = True
        self.refine_iters = 1
        self.bos_id = 95
        self.eos_id = 0
        self.pad_id = 96

        dec_num_heads = config.dec_num_heads
        dec_mlp_ratio=config.dec_mlp_ratio
        dropout = config.dropout
        dec_depth = config.dec_depth
        perm_num = config.perm_num
        perm_mirrored = config.perm_mirrored
        max_label_length = config.max_label_length


        self.encoder = Encoder(img_size, patch_size, embed_dim=embed_dim, depth=enc_depth, num_heads=enc_num_heads,
                               mlp_ratio=enc_mlp_ratio)
        decoder_layer = DecoderLayer(embed_dim, dec_num_heads, embed_dim * dec_mlp_ratio, dropout)
        self.decoder = Decoder(decoder_layer, num_layers=dec_depth, norm=nn.LayerNorm(embed_dim))

        # Perm/attn mask stuff
        self.rng = np.random.default_rng()
        self.max_gen_perms = perm_num // 2 if perm_mirrored else perm_num
        self.perm_forward = True
        self.perm_mirrored =True

        # We don't predict <bos> nor <pad>
        self.head = nn.Linear(embed_dim, len(self.tokenizer) - 2)
        self.text_embed = TokenEmbedding(len(self.tokenizer), embed_dim)#编码上一时刻的字符的embedding

        # +1 for <eos>
        self.pos_queries = nn.Parameter(torch.Tensor(1, max_label_length + 1, embed_dim))#字符embeddin
        self.dropout = nn.Dropout(p=dropout)
        # Encoder has its own init.
        #named_apply(partial(init_weights, exclude=['encoder']), self)
        #nn.init.trunc_normal_(self.pos_queries, std=.02)
        if config.full_ckpt is not None:
            logging.info(f'Read full ckpt parseq model from {config.full_ckpt}.')
            state = torch.load(config.full_ckpt)
            self.load_state_dict(state, strict=True)



    def encode(self, img: torch.Tensor):
        return self.encoder(img)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[Tensor] = None,
               tgt_padding_mask: Optional[Tensor] = None, tgt_query: Optional[Tensor] = None,
               tgt_query_mask: Optional[Tensor] = None):
        N, L = tgt.shape
        # <bos> stands for the null context. We only supply position information for characters after <bos>.
        null_ctx = self.text_embed(tgt[:, :1]) #只有BOS
        tgt_emb = self.pos_queries[:, :L - 1] + self.text_embed(tgt[:, 1:])# 这里不对字符的embed加位置编码，枚举的话就是一点意义也没有。之前所有时间步y的embedding  这里有广播，pos_queries就是位置编码
        tgt_emb = self.dropout(torch.cat([null_ctx, tgt_emb], dim=1))
        if tgt_query is None:
            tgt_query = self.pos_queries[:, :L].expand(N, -1, -1)
        tgt_query = self.dropout(tgt_query)#第二种类型的位置查询，不依赖于上一时刻字符的embedding，目的：找到当前时刻的解码中之前的哪个字符embedding对我的帮助更大
        return self.decoder(tgt_query, tgt_emb, memory, tgt_query_mask, tgt_mask, tgt_padding_mask)#前两个参数在第一级的结果（其中第一个参数是q，第二个是k、v）作为第二级解码的q

    def forward(self, images, tgt=None,permutation=None,is_train=False):
        # ==how to test
        # targets = self.tokenizer.encode(label_str, self.device)
        # logits = model(images,targets,is_train=False)
        # ====how to train
        # targets = self.tokenizer.encode(label_str, self.device)
        # tgt_perms = self.gen_tgt_perms(targets)  # 返回6，max_len的解码顺序。6就是最后枚举的个数K的多少
        # output_dict= model(images, targets, tgt_perms,is_train=False)

        assert (is_train is True and tgt!=None and permutation !=None) \
               or (is_train is False and tgt==None and permutation==None)
        self._device = images.device

        nimg = images.permute(0,2,3,1).cpu().numpy()
        img = np.uint8(nimg * 255)  # 将原来tensor里0-1的数值乘以255，以便转为uint8数据形式，uint8是图片的数值形式。
        import matplotlib.pyplot as plt

        # 无论是test也好还是train也好，都要先做0.5的归一化
        trans = transforms.Normalize(0.5, 0.5)
        images = trans(images)

        if is_train:
            return self.training_step(images,tgt=tgt,tgt_perms=permutation)
        else:
            return self.test_step(images)



    def test_step(self,images):
        # targets_ = tgt[:, 1:]  # Discard <bos>
        # max_length = targets_.shape[1] - 1
        if images.shape[2] == 16:
            images = F.interpolate(images, scale_factor=2, mode='bicubic', align_corners=True)

        max_length = None
        testing = not self.training
        max_length = self.max_label_length if max_length is None else min(max_length, self.max_label_length)
        bs = images.shape[0]
        num_steps = max_length + 1
        memory = self.encode(images)#2,384,128

        pos_queries = self.pos_queries[:, :num_steps].expand(bs, -1, -1)#每个batch的query是一样的，在batch上重复
        tgt_mask = query_mask = torch.triu(torch.full((num_steps, num_steps), float('-inf'), device=self._device), 1)
        # 全负矩阵的上三角矩阵就是AR所用到的mask
        if self.decode_ar:
            tgt_in = torch.full((bs, num_steps), self.pad_id, dtype=torch.long, device=self._device)
            tgt_in[:, 0] = self.bos_id

            logits = []
            for i in range(num_steps):
                j = i + 1  # next token index
                tgt_out = self.decode(tgt_in[:, :j], memory, tgt_mask[:j, :j], tgt_query=pos_queries[:, i:j],
                                      tgt_query_mask=query_mask[i:j, :j])
                p_i = self.head(tgt_out)
                logits.append(p_i)
                if j < num_steps:
                    tgt_in[:, j] = p_i.squeeze().argmax(-1)
                    if testing and (tgt_in == self.eos_id).any(dim=-1).all():
                        break

            logits = torch.cat(logits, dim=1)#N,8(max_len),95 在测试时是遇到终止符就结束
        else:
            tgt_in = torch.full((bs, 1), self.bos_id, dtype=torch.long, device=self._device)
            tgt_out = self.decode(tgt_in, memory, tgt_query=pos_queries)
            logits = self.head(tgt_out)

        if self.refine_iters:
            query_mask[torch.triu(torch.ones(num_steps, num_steps, dtype=torch.bool, device=self._device), 2)] = 0
            bos = torch.full((bs, 1), self.bos_id, dtype=torch.long, device=self._device)
            for i in range(self.refine_iters):
                tgt_in = torch.cat([bos, logits[:, :-1].argmax(-1)], dim=1)
                tgt_padding_mask = ((tgt_in == self.eos_id).int().cumsum(-1) > 0)  # mask tokens beyond the first EOS token.
                tgt_out = self.decode(tgt_in, memory, tgt_mask, tgt_padding_mask,
                                      tgt_query=pos_queries, tgt_query_mask=query_mask[:, :tgt_in.shape[1]])
                logits = self.head(tgt_out)


        return {'logits':logits}

    def gen_tgt_perms(self, tgt):
        max_num_chars = tgt.shape[1] - 2
        if max_num_chars == 1:
            return torch.arange(3, device=self._device).unsqueeze(0)
        perms = [torch.arange(max_num_chars, device=self._device)] if self.perm_forward else []
        max_perms = math.factorial(max_num_chars)
        if self.perm_mirrored:
            max_perms //= 2
        num_gen_perms = min(self.max_gen_perms, max_perms)
        if max_num_chars < 5:
            if max_num_chars == 4 and self.perm_mirrored:
                selector = [0, 3, 4, 6, 9, 10, 12, 16, 17, 18, 19, 21]
            else:
                selector = list(range(max_perms))
            perm_pool = torch.as_tensor(list(permutations(range(max_num_chars), max_num_chars)), device=self._device)[selector]
            if self.perm_forward:
                perm_pool = perm_pool[1:]
            perms = torch.stack(perms)
            if len(perm_pool):
                i = self.rng.choice(len(perm_pool), size=num_gen_perms - len(perms), replace=False)
                perms = torch.cat([perms, perm_pool[i]])
        else:#extend就保证了perms里面必定含有顺序解码的枚举序列
            perms.extend([torch.randperm(max_num_chars, device=self._device) for _ in range(num_gen_perms - len(perms))])
            perms = torch.stack(perms)
        if self.perm_mirrored:
            comp = perms.flip(-1)
            perms = torch.stack([perms, comp]).transpose(0, 1).reshape(-1, max_num_chars)#把正反两面放到相邻的位置上去，为什么？？
        bos_idx = perms.new_zeros((len(perms), 1))
        eos_idx = perms.new_full((len(perms), 1), max_num_chars + 1)
        perms = torch.cat([bos_idx, perms + 1, eos_idx], dim=1)
        if len(perms) > 1:
            perms[1, 1:] = max_num_chars + 1 - torch.arange(max_num_chars + 1, device=self._device)
        return perms

    def generate_attn_masks(self, perm):
        sz = perm.shape[0]
        mask = torch.zeros((sz, sz), device=self._device)
        for i in range(sz):
            query_idx = perm[i]
            masked_keys = perm[i + 1:]
            mask[query_idx, masked_keys] = float('-inf')#横坐标是跳步的，每一行掩模哪一个也是随机跳步的
        content_mask = mask[:-1, :-1].clone()
        mask[torch.eye(sz, dtype=torch.bool, device=self._device)] = float('-inf')  # mask "self"
        query_mask = mask[1:, :-1]
        return content_mask, query_mask#实际中只用到了query_mask,直接从第一个字符开始问，同时不用结束符来聚合（去掉第一行和最后一列）

    def training_step(self, images,tgt,tgt_perms):
        #TODO 这个label要在外面生成，之后就可以分布式训练了
        # images, labels = batch
        # tgt = self.tokenizer.encode(labels, self._device)# 加上EOS的token，以及padding，一个批次的长度和最长的实例长度相同
        memory = self.encode(images)
        # TODO 这里决定把教师和学生都使用这个training的函数
        # tgt_perms = self.gen_tgt_perms(tgt)#返回6，max_len的解码顺序。6就是最后枚举的个数K的多少
        tgt_in = tgt[:, :-1]# 对原始的标签掐头去尾就得到了回归模型的输入和输出
        tgt_out = tgt[:, 1:]
        tgt_padding_mask = (tgt_in == self.pad_id) | (tgt_in == self.eos_id)
        loss = 0
        loss_numel = 0
        n = (tgt_out != self.pad_id).sum().item()#一个批次中的有效字符的个数
        for i, perm in enumerate(tgt_perms):
            tgt_mask, query_mask = self.generate_attn_masks(perm)
            out = self.decode(tgt_in, memory, tgt_mask, tgt_padding_mask, tgt_query_mask=query_mask)
            logits = self.head(out).flatten(end_dim=1)#[NT, C]
            loss += n * F.cross_entropy(logits, tgt_out.flatten(), ignore_index=self.pad_id)
            loss_numel += n
            if i == 1:
                tgt_out = torch.where(tgt_out == self.eos_id, self.pad_id, tgt_out)
                n = (tgt_out != self.pad_id).sum().item()
        loss /= loss_numel#最后的损失时每个时间步的损失，比batch-mean还要小

        self.log('loss', loss)
        return loss

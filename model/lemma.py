import math
import torch
import torch.nn.functional as F
from torch import nn
from model.Position_aware_module import PositionAwareModule,Location_enhancement_Multimodal_alignment
from .tps_spatial_transformer import TPSSpatialTransformer
from .stn_head import STNHead
from torchvision import transforms
SHUT_BN = False

def showPIL(img,batch_id=0):
    img = img[batch_id,:3,...]*255
    img = torch.as_tensor(img.detach().cpu(),dtype=torch.uint8).numpy()
    img = transforms.functional.to_pil_image(img.transpose((1,2,0)))
    img.show()


class AffineModulate(nn.Module):
    def __init__(self, channel=64):
        super(AffineModulate, self).__init__()
        self.BN = nn.BatchNorm2d(channel)
        self.conv1x1_1 = nn.Sequential(
            nn.Conv2d(channel*2, channel, 1),
            nn.BatchNorm2d(channel),
            nn.PReLU(),
            nn.Conv2d(channel, channel, 1)
        )
        self.conv1x1_2 = nn.Sequential(
            nn.Conv2d(channel*2, channel, 1),
            nn.BatchNorm2d(channel),
            nn.PReLU(),
            nn.Conv2d(channel, channel, 1)
        )
        self.conv1x1_3 = nn.Sequential(
            nn.Conv2d(channel*2, channel, 1),
            nn.BatchNorm2d(channel),
            nn.PReLU(),
            nn.Conv2d(channel, channel, 1)
        )
        self.channel_attn = nn.Sequential(
            # Linear
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(16, 64), groups=64),  # 全局深度可分离
            nn.Conv2d(channel, channel, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 1),
            nn.Sigmoid()
        )
        self.aggregation = nn.Sequential(
            nn.Conv2d(channel,channel,1),
            nn.BatchNorm2d(channel),
            nn.PReLU(),
            nn.Conv2d(channel,channel,1)
        )
        self.BN2 = nn.BatchNorm2d(channel)

    def forward(self,image_feature,tp_map):
        tp_map = self.BN(tp_map)
        cat_feature = torch.cat([image_feature,tp_map],dim=1)
        x1,x2,x3 = self.conv1x1_1(cat_feature),self.conv1x1_2(cat_feature),self.conv1x1_3(cat_feature)
        x1 = self.channel_attn(x1)
        x2 = self.aggregation(x1*x2)
        return self.BN2(x3+x2)


class LEMMA(nn.Module):
    def __init__(self,
                 scale_factor=2,
                 width=128,
                 height=32,
                 STN=False,
                 srb_nums=5,
                 mask=True,
                 hidden_units=32,
                 word_vec_d=300,
                 text_emb=37,
                 out_text_channels=64,
                 feature_rotate=False,
                 rotate_train=3.,
                 cfg=None):
        super(LEMMA, self).__init__()
        in_planes = 3
        if mask:
            in_planes = 4
        assert math.log(scale_factor, 2) % 1 == 0
        upsample_block_num = int(math.log(scale_factor, 2))
        self.block1 = nn.Sequential(
            nn.Conv2d(in_planes, 2 * hidden_units, kernel_size=9, padding=4), # 256 out feature
            nn.PReLU()
        )
        self.srb_nums = srb_nums
        for i in range(srb_nums):
            setattr(self, 'block%d' % (i + 2), RecurrentResidualBlockAffine(2 * hidden_units, out_text_channels))
        self.feature_rotate = feature_rotate
        self.rotate_train = rotate_train
        if not SHUT_BN:
            setattr(self, 'block%d' % (srb_nums + 2),
                    nn.Sequential(
                        nn.Conv2d(2 * hidden_units, 2 * hidden_units, kernel_size=3, padding=1),
                        nn.BatchNorm2d(2 * hidden_units)
                    ))
        else:
            setattr(self, 'block%d' % (srb_nums + 2),
                    nn.Sequential(
                        nn.Conv2d(2 * hidden_units, 2 * hidden_units, kernel_size=3, padding=1),
                        # nn.BatchNorm2d(2 * hidden_units)
                    ))

        block_ = [UpsampleBLock(2 * hidden_units, 2) for _ in range(upsample_block_num)]
        block_.append(nn.Conv2d(2 * hidden_units, in_planes, kernel_size=9, padding=4))
        setattr(self, 'block%d' % (srb_nums + 3), nn.Sequential(*block_))
        self.tps_inputsize = [height // scale_factor, width // scale_factor]
        tps_outputsize = [height // scale_factor, width // scale_factor]
        num_control_points = 20
        tps_margins = [0.05, 0.05]
        self.stn = STN
        if self.stn:
            self.tps = TPSSpatialTransformer(
                output_image_size=tuple(tps_outputsize),
                num_control_points=num_control_points,
                margins=tuple(tps_margins))

            self.stn_head = STNHead(
                in_planes=in_planes,
                num_ctrlpoints=num_control_points,
                activation='none',
                input_size=self.tps_inputsize)

        self.block_range = [k for k in range(2, self.srb_nums + 2)]
        self.position_prior = PositionAwareModule(cfg.PositionAware)
        # We implement both Location Enhancement Module and Multi-modal Alignment Module here
        self.guidanceGen = Location_enhancement_Multimodal_alignment(cfg.PositionAware)


    def forward(self, x):
        if self.stn and self.training:
            _, ctrl_points_x = self.stn_head(x)
            x, _ = self.tps(x, ctrl_points_x)

        x_ = x.clone()[:,:3,:,:].detach()
        x_ = F.interpolate(x_,scale_factor=2,mode='bicubic',align_corners=True)
        pos_prior = self.position_prior(x_) # 'attn_map, 'text_feature', 'text_logits', 'pt_lengths'

        block = {'1': self.block1(x)}
        padding_feature = block['1']
        tp_map,pr_weights = self.guidanceGen(pos_prior,padding_feature)
        for i in range(self.srb_nums + 1):
            if i + 2 in self.block_range:
                block[str(i + 2)] = getattr(self, 'block%d' % (i + 2))(block[str(i + 1)], tp_map)
            else:
                block[str(i + 2)] = getattr(self, 'block%d' % (i + 2))(block[str(i + 1)])

        block[str(self.srb_nums + 3)] = getattr(self, 'block%d' % (self.srb_nums + 3)) \
            ((block['1'] + block[str(self.srb_nums + 2)]))
        output = torch.tanh(block[str(self.srb_nums + 3)])
        self.block = block
        return output, pos_prior


class RecurrentResidualBlock(nn.Module):
    def __init__(self, channels):
        super(RecurrentResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.gru1 = GruBlock(channels, channels)
        # self.prelu = nn.ReLU()
        self.prelu = mish()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.gru2 = GruBlock(channels, channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        residual = self.gru1(residual.transpose(-1, -2)).transpose(-1, -2)

        return self.gru2(x + residual)




class RecurrentResidualBlockAffine(nn.Module):
    def __init__(self, channels,text_channel=None):
        # channls = 64
        super(RecurrentResidualBlockAffine, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.gru1 = GruBlock(channels, channels)
        self.prelu = mish()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.gru2 = GruBlock(channels, channels) # + text_channels
        self.affine = AffineModulate(channel=channels)
        self.BN = nn.BatchNorm2d(channels)


    def forward(self, x, tp_map):
        residual = self.conv1(x)
        if not SHUT_BN:
            residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        if not SHUT_BN:
            residual = self.bn2(residual)
        # AdaFM
        residual = self.affine(residual,tp_map)
        # SRB
        residual = self.gru1(residual.transpose(-1, -2)).transpose(-1, -2)

        return self.gru2(self.BN(x + residual))




class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)

        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        # self.prelu = nn.ReLU()
        self.prelu = mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class mish(nn.Module):
    def __init__(self, ):
        super(mish, self).__init__()
        self.activated = True

    def forward(self, x):
        if self.activated:
            x = x * (torch.tanh(F.softplus(x)))
        return x


class GruBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GruBlock, self).__init__()
        assert out_channels % 2 == 0
        self.gru = nn.GRU(out_channels, out_channels // 2, bidirectional=True, batch_first=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        b = x.size()
        x = x.view(b[0] * b[1], b[2], b[3])
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = x.view(b[0], b[1], b[2], b[3])
        x = x.permute(0, 3, 1, 2)
        return x

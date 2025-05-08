import torch
from torch import nn
from models.model.attion3D import DAModule
from models.model.RVM_unet import MambaLayer

class SElayer_3D(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SElayer_3D, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume


def interweave_tensors(refimg_fea, targetimg_fea):
    B, C, H, W = refimg_fea.shape
    interwoven_features = refimg_fea.new_zeros([B, 2 * C, H, W])
    interwoven_features[:, ::2, :, :] = refimg_fea  # interval=2,from 0
    interwoven_features[:, 1::2, :, :] = targetimg_fea  # interval=2, from 1
    interwoven_features = interwoven_features.contiguous()  # 使用contiguous()针对x进行变化，感觉上就是我们认为的深拷贝
    return interwoven_features


def conv3d_lrelu(in_planes, out_planes, kernel_size, stride, pad, dilation=(1, 1, 1)):
    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size, stride, pad, dilation=dilation, bias=False),
                         nn.InstanceNorm3d(out_planes),
                         nn.SiLU(True))

class BuildCostVolume(nn.Module):
    def __init__(self, channels_features, num_groups, volume_size):
        super(BuildCostVolume, self).__init__()
        self.num_groups = num_groups
        self.volume_size = volume_size
        self.mb = MambaLayer(channels_features, channels_features//8)
        self.att = DAModule(channels_features//8, reduction_ratio=2, pool_type=['avg', 'max'], no_spatial_depth=False)
    def forward(self, featuresL, featuresR):
        B, C, H, W = featuresL.shape
        volume = featuresL.new_zeros([B, self.num_groups, self.volume_size, H, W])
        for i in range(self.volume_size):
            if i > 0:
                # x = groupwise_correlation(featuresL[:, :, :, i:], featuresR[:, :, :, :-i], self.num_groups)
                x = interweave_tensors(featuresL[:, :, :, i:], featuresR[:, :, :, :-i])
                x1 = x.unsqueeze(2)
                cost = self.mb(x1)
                # cost = self.pad(cost)
                cost = self.att(cost)
                cost = cost.squeeze(2)
                volume[:, :, i, :, i:] = cost
            else:
                # x = groupwise_correlation(featuresL, featuresR,self.num_groups)
                x = interweave_tensors(featuresL, featuresR)  # left and right features interlacing
                x1 = x.unsqueeze(2)
                cost = self.mb(x1)
                cost = self.att(cost)
                cost = cost.squeeze(2)
                volume[:, :, i, :, :] = cost
        volume = volume.contiguous()
        return volume


class FusionCostVolume(nn.Module):
    def __init__(self, in_size, out_size):
        super(FusionCostVolume, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose3d(in_size, out_size, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.GroupNorm(out_size // 4, out_size),
            nn.LeakyReLU(True),
            conv3d_lrelu(out_size, out_size, 3, 1, 1))

    def forward(self, low_cost_volume, high_cost_volume):
        low_cost_volume = self.net(low_cost_volume)
        cost_volume = torch.cat([low_cost_volume, high_cost_volume], dim=1)
        return cost_volume

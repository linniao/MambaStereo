# fixme: 3D Attention Module
import torch.nn as nn
import torch
import torch.nn.functional as F


# fixme: Channel Attention
# fixme: Flatten for Channel Attention
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        # MLP
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        channel_att_raw = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool3d(x, kernel_size=(x.size(2), x.size(3), x.size(4)),
                                        stride=(x.size(2), x.size(3), x.size(4)))  # [B, C, 1, 1, 1]
                channel_att_raw = self.mlp(avg_pool)  # [B, C]
            elif pool_type == 'max':
                max_pool = F.max_pool3d(x, kernel_size=(x.size(2), x.size(3), x.size(4)),
                                        stride=(x.size(2), x.size(3), x.size(4)))  # [B, C, 1, 1, 1]
                channel_att_raw = self.mlp(max_pool)  # [B, C]

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        scale = torch.sigmoid(channel_att_sum)  # [B, C]
        scale = scale.unsqueeze(2).unsqueeze(3).unsqueeze(3).expand_as(x)  # [B, C, X.D, X.H, X.W]
        return x * scale


class ChannelPool(nn.Module):
    def __init__(self):
        super(ChannelPool, self).__init__()

    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


# fixme: Feature Extraction for Spatial Attention
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm3d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class SpatialDepthGate(nn.Module):
    def __init__(self):
        super(SpatialDepthGate, self).__init__()
        kernel_size = 7
        self.channel_pool = ChannelPool()
        self.channel_conv = BasicConv(2, 1, kernel_size=(1, kernel_size, kernel_size), stride=1,
                                      padding=(0, (kernel_size - 1) // 2, (kernel_size - 1) // 2),
                                      relu=False)  # 后面为了减少参数量可以考虑将它换为可分离卷积
        self.depth_conv = BasicConv(1, 1, kernel_size=(kernel_size, 1, 1), stride=1,
                                    padding=((kernel_size - 1) // 2, 0, 0), relu=False)
        self.overall_conv = BasicConv(1, 1, kernel_size=(kernel_size, kernel_size, kernel_size), stride=1,
                                      padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):  # x [B, C. X.D, X.H, X.W]
        compress = self.channel_pool(x)  # [B, 2, X.D, X.H, X.W]  [B, 32, 12, 16,20]
        compress = self.channel_conv(compress)  # [B, 1, 12 ,16, 20]
        compress = self.depth_conv(compress)  # [B, 1, 12 ,16, 20]
        compress = self.overall_conv(compress)  # [1, 1, 12, 16, 20]
        scale = torch.sigmoid(compress)  # [1, 1, 12, 16, 20]
        return x * scale


class DAModule(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_type=['avg', 'max'], no_spatial_depth=False):
        super(DAModule, self).__init__()
        self.gate_channels = gate_channels
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_type)
        self.no_spatial_depth = no_spatial_depth
        #add skcale
        self.norm1 = nn.LayerNorm(gate_channels)
        self.skip_scale = nn.Parameter(torch.ones(1))

        if not no_spatial_depth:
            self.SpatialDepthGate = SpatialDepthGate()

    def forward(self, x):
        B, C = x.shape[:2]
        assert C == self.gate_channels
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        # print(x_flat.shape)
        x_norm1 = self.norm1(x_flat)
        x_norm1= x_norm1.transpose(-1, -2).reshape(B, self.gate_channels, *img_dims)
        x = self.ChannelGate(x)
        if not self.no_spatial_depth:
            x = self.SpatialDepthGate(x)
        #add scale
        x = x + self.skip_scale * x_norm1
        return x


if __name__ == '__main__':
    input1 = torch.randn(1, 48, 32, 32, 48)

    eam = DAModule(48, reduction_ratio=8, pool_type=['avg', 'max'], no_spatial_depth=False)
    output = eam(input1)
    print(output.shape)
# self.attention_block_3 = DAModule(4, reduction_ratio=2, pool_type=['avg', 'max'], no_spatial_depth=False)

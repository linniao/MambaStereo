import torch.nn.functional as F

import torch
import torch.nn as nn


class senet(nn.Module):
    def __init__(self, channel, ratio=4):
        super(senet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, False),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel, False),
            nn.Sigmoid(),

        )

    def forward(self, x):
        b, c, h, w = x.size()
        avg = self.avg_pool(x).view(b, c)
        fc = self.fc(avg).view(b, c, 1, 1)
        return x * fc


def conv3d_lrelu(in_planes, out_planes, kernel_size, stride, pad, dilation=(1, 1, 1)):
    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size, stride, pad, dilation=dilation, bias=False),
                         nn.InstanceNorm3d(out_planes),
                         nn.SiLU(True))


def conv2d_lrelu(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size, stride, pad, dilation, bias=False),
                         nn.InstanceNorm2d(out_planes),
                         nn.SiLU(True))


class warp_err(nn.Module):
    def __init__(self, channel):
        super(warp_err, self).__init__()
        self.conv1 = conv2d_lrelu(channel + 1, channel + 1, 3, 1, 1, dilation=1)
        self.conv2 = conv2d_lrelu(channel + 1, channel + 1, 3, 1, 2, dilation=2)
        self.conv3 = conv2d_lrelu(channel + 1, channel + 1, 3, 1, 4, dilation=4)
        self.conv4 = nn.Sequential(conv2d_lrelu((channel + 1) * 3, (channel + 1) * 3 // 2, 1, 1, 0, 1),
                                   conv2d_lrelu((channel + 1) * 3 // 2, (channel + 1) * 3 // 2, 3, 1, 1, 1),
                                   conv2d_lrelu((channel + 1) * 3 // 2, (channel + 1) * 3, 1, 1, 0, 1),
                                   senet((channel + 1) * 3)
                                   )
        self.conv5 = nn.Sequential(conv2d_lrelu((channel + 1) * 3, (channel + 1) * 3 // 2, 1, 1, 0, 1),
                                   conv2d_lrelu((channel + 1) * 3 // 2, (channel + 1) * 3 // 2, 3, 1, 1, 1),
                                   conv2d_lrelu((channel + 1) * 3 // 2, 1, 1, 1, 0, 1),
                                   conv2d_lrelu(1, 1, 3, 1, 1, 1))

    def forward(self, L, R, disp_est):
        B, C, H, W = R.shape
        # 创建网格坐标
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float().cuda()
        # 添加视差位移
        vgrid = grid.clone()
        vgrid[:, :1, :, :] = vgrid[:, :1, :, :] - disp_est
        # 归一化到[-1, 1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        R_warped = F.grid_sample(R, vgrid, align_corners=True)
        erropmap = L - R_warped
        # mask = torch.ones(R.size(), requires_grad=True)
        # mask = nn.functional.grid_sample(mask, vgrid)
        # mask[mask < 0.999] = 0
        # mask[mask > 0] = 1
        # erropmap = mask * erropmap

        err_dis = torch.cat([erropmap, disp_est], dim=1)

        err_dis_1 = self.conv1(err_dis)
        err_dis_2 = self.conv2(err_dis)
        err_dis_3 = self.conv3(err_dis)
        err_dis_all = torch.cat([err_dis_1, err_dis_2, err_dis_3], dim=1)
        dis_ref = self.conv4(err_dis_all)
        dis_ref = self.conv5(dis_ref)
        final_dis = disp_est + dis_ref
        final_dis = final_dis.squeeze(1)

        return final_dis


# if __name__ == '__main__':
#     input1 = torch.randn([2, 32, 64, 128]).cuda()
#     input2 = torch.randn([2, 32, 64, 128]).cuda()
#     disp_est = torch.randn([2, 64, 128]).cuda()
#     model = warp_err(32).cuda()
#     model(input1, input2, disp_est.unsqueeze(1))

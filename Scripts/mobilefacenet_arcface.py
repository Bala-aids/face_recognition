import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k, s, p, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)

    def forward(self, x):
        return self.prelu(self.bn(self.conv(x)))


class DepthWise(nn.Module):
    def __init__(self, in_c, out_c, s):
        super().__init__()
        self.depthwise = ConvBlock(in_c, in_c, 3, s, 1, groups=in_c)
        self.pointwise = ConvBlock(in_c, out_c, 1, 1, 0)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MobileFaceNet(nn.Module):
    def __init__(self, embedding_size=128):
        super().__init__()

        self.conv1 = ConvBlock(3, 64, 3, 2, 1)

        self.features = nn.Sequential(
            DepthWise(64, 64, 1),
            DepthWise(64, 64, 2),
            DepthWise(64, 128, 1),
            DepthWise(128, 128, 2),
            DepthWise(128, 128, 1),
            DepthWise(128, 128, 1),
            DepthWise(128, 256, 2),
            DepthWise(256, 256, 1),
            DepthWise(256, 256, 1),
            DepthWise(256, 256, 1),
            DepthWise(256, 256, 1),
            DepthWise(256, 256, 1),
            DepthWise(256, 256, 1),
            DepthWise(256, 512, 2),
            DepthWise(512, 512, 1),
        )

        self.conv2 = nn.Conv2d(512, 512, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(512)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, embedding_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = self.bn(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.normalize(x)
        return x

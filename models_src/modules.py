import torch.nn as nn
from torch import nn as nn


class ConvBnAct(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, padding, stride):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size=kernel_size, padding=padding, stride=stride)  # 316
        self.bn = nn.BatchNorm2d(out_size)
        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.do = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return self.do(x)

    def to_list(self):
        return [self.conv, self.bn, self.act, self.do]


class UpConvBnRelu(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, padding, stride, use_act=True):
        super(UpConvBnRelu, self).__init__()
        self.conv = nn.ConvTranspose2d(in_size, out_size, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(out_size)
        self.act = nn.ReLU() if use_act else nn.Identity()
        self.do = nn.Dropout(0.5) if use_act else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return self.do(x)


def double_conv_down(in_channels: int, out_channels: int, stride1=1, stride2=1, padding1=0, padding2=0, both=True):
    first = [nn.Conv2d(in_channels, out_channels, (2, 2), padding=padding1, stride=(stride1, stride1)),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(inplace=True),
             nn.Dropout(0.5), ]
    second = [nn.Conv2d(out_channels, out_channels, (2, 2), padding=padding2, stride=(stride2, stride2)),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True),
              nn.Dropout(0.5)
              ]
    modules = [*first, *second] if both else [*first]
    return nn.Sequential(
        *modules
    )


def double_conv_up(in_channels: int, out_channels: int, stride1=1, stride2=1, padding1=0, padding2=0, both=True):
    first = [
        nn.ConvTranspose2d(in_channels, out_channels, (2, 2), padding=(padding1, padding1), stride=(stride1, stride1)),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
    ]
    second = [nn.Conv2d(out_channels, out_channels, (3, 3), padding=(padding2, padding2), stride=(stride2, stride2)),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True),
              nn.Dropout(0.5)
              ]
    modules = [*first, *second] if both else [*first]
    return nn.Sequential(
        *modules
    )

import torch
import torch.nn as nn

from models_src.modules import double_conv_down, double_conv_up
from util.tools import init_weights

init = 16


class Generator(nn.Module):
    def __init__(self, in_class, n_class):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_class)
        self.dconv_down1 = double_conv_down(in_class, init, stride1=2, padding1=1)
        self.dconv_down2 = double_conv_down(init, 2*init, stride1=2, padding1=1)
        self.dconv_down3 = double_conv_down(2*init, 4*init, stride1=1, padding1=1)
        self.dconv_up2 = double_conv_up(2*init + 4*init, 2*init, stride1=2, padding1=0, stride2=1,
                                        padding2=1)
        self.dconv_up3 = double_conv_up(2*init + init, init, stride1=2, padding1=0, stride2=1,
                                        padding2=1)

        self.conv_last = nn.Conv2d(init, n_class, (1, 1))
        self.act = nn.Tanh()
        self.apply(init_weights)

    def forward(self, x):
        x = self.norm(x)
        conv1 = self.dconv_down1(x)
        conv2 = self.dconv_down2(conv1)
        x = self.dconv_down3(conv2)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up3(x)
        out = self.conv_last(x)

        return self.act(out)

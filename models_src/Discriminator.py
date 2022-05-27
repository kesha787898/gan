import torch.nn as nn

from models_src.modules import ConvBnAct
from util import config
from util.tools import init_weights

init = 8


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.norm = nn.BatchNorm2d(4)
        self.conv1 = ConvBnAct(4, init * 2, kernel_size=2, padding=1, stride=2)  # 316
        self.conv2 = ConvBnAct(init * 2, init * 16, kernel_size=2, padding=0, stride=2)  # 79
        self.conv3 = ConvBnAct(init * 16, init * 2, kernel_size=2, padding=0, stride=2)  # 158
        self.conv4 = ConvBnAct(init * 2, config.discr_out[0], kernel_size=2, padding=0, stride=1)  # 158
        self.act = nn.Sigmoid()
        self.apply(init_weights)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.act(x)

    def to_list(self):
        return [self.norm,
                *self.conv1.to_list(),
                *self.conv2.to_list(),
                *self.conv3.to_list(),
                *self.conv4.to_list(),
                self.act]

    def to_seq(self):
        return nn.Sequential(*self.to_list())

import torch

from models_src.Gan import GAN
from util.config import inp_shape, discr_out

gan = GAN()
d = gan.discriminator
g = gan.generator

print(d(torch.randn((1, 4, *inp_shape))).shape)
assert d(torch.randn((1, 4, *inp_shape))).shape == (1, *discr_out)

print(g(torch.randn((1, 1, *inp_shape))).shape)
assert g(torch.randn((1, 1, *inp_shape))).shape == (1, 3, *inp_shape)



import torch

from models_src.Discriminator import Discriminator
from util.config import inp_shape, discr_out
from models_src.generator import Generator
from util.rec_field import receptive_field

d = Discriminator()
g = Generator(1, 3)
#print(receptive_field(d.to_seq(), (4, 200, 200), device='cpu'))

print(d(torch.randn((1, 4, *inp_shape))).shape)
assert d(torch.randn((1, 4, *inp_shape))).shape == (1, *discr_out)

print(g(torch.randn((1, 1, *inp_shape))).shape)
assert g(torch.randn((1, 1, *inp_shape))).shape == (1, 3, *inp_shape)

# print(Generator()(torch.randn((1, 1, *inp_shape))).shape)
# assert Generator()(torch.randn((1, 1, *inp_shape))).shape == (1, 3, *inp_shape)

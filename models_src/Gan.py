import gc
import os

import cv2
import torch
from pytorch_lightning import LightningModule
from torchvision import transforms

import numpy as np

from models_src.ttt import NLayerDiscriminator, UnetGenerator
from util import config
from util.config import inp_shape
from util.tools import skel

class GAN(LightningModule):
    def __init__(self):
        super().__init__()

        self.generator = UnetGenerator(1, 3, 3, ngf=12)
        self.discriminator = NLayerDiscriminator(4, ndf=36,n_layers=3)
        print(self.generator)
        print(self.discriminator)
        self.loss = torch.nn.BCELoss()
        self.last_gen_loss = 0
        self.last_discr_loss = 0
        self.n = 0
        self.last_g_loss = 0
        self.last_d_loss = 0

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_idx, optimizer_idx):
        torch.cuda.empty_cache()
        gc.collect()
        edge_tensor, real_tensor = batch
        # train generator
        predicted_image = self.generator(edge_tensor)
        if optimizer_idx == 0:
            # if self.last_d_loss > coef * self.last_g_loss and self.n > 4:
            #    return None
            self.discriminator.eval()
            self.discriminator.requires_grad_(False)
            self.generator.train()
            self.discriminator.requires_grad_(True)

            img = predicted_image.detach().cpu()
            edge = np.concatenate([edge_tensor[0].detach().cpu()] * 3)
            prediction = img[0]
            if config.debug or batch_idx % 10 == 0:
                self.logger.experiment.add_image("images",
                                                 np.concatenate(
                                                     [edge, prediction, real_tensor[0].detach().cpu()],
                                                     axis=1),
                                                 self.n)
            pred_tensor = self.generator(edge_tensor)
            loss_gen = self.loss(self.discriminator(torch.cat([edge_tensor, pred_tensor], dim=1)),
                                 (1 - torch.zeros(
                                     [len(edge_tensor),
                                      *config.discr_out])).to(config.device))
            loss = loss_gen
            self.last_discr_loss = loss
            self.log("loss_gen_total", float(loss), prog_bar=True)
            self.log("loss_gen", float(loss_gen), prog_bar=True)
            self.logger.experiment.add_scalar("loss_gen_total", float(loss), self.n)
            self.logger.experiment.add_scalar("loss_gen", float(loss_gen), self.n)
            self.last_g_loss = float(loss)
            return loss

        # train discriminator
        if optimizer_idx == 1:
            # Todo подумать
            # if self.last_g_loss > coef * self.last_d_loss and self.n > 4:
            #    return None
            self.discriminator.train()
            self.discriminator.requires_grad_(True)
            self.generator.eval()
            self.generator.requires_grad_(False)
            real_x = torch.cat([edge_tensor, real_tensor], dim=1)
            fake_x = torch.cat([edge_tensor, self.generator(edge_tensor)], dim=1)
            real_y = (1 - torch.zeros(
                [len(real_tensor), *config.discr_out])
                      ).to(config.device)
            fake_y = (torch.zeros(
                [len(real_tensor), *config.discr_out])
            ).to(config.device)
            x = torch.cat([real_x, fake_x], dim=0)
            y = torch.cat([real_y, fake_y], dim=0)
            loss = self.loss(self.discriminator(x), y)
            self.last_gen_loss = loss
            self.log("loss_discr", float(loss), prog_bar=True)
            self.logger.experiment.add_scalar("loss_discr", float(loss), self.n)
            self.last_d_loss = float(loss)
            self.n += 1
            return loss

    def on_epoch_end(self) -> None:
        test_images = os.listdir("D:/gan/eval")
        for idx, img_name in enumerate(test_images):
            self.generator.eval()
            image = cv2.imread(os.path.join("D:/gan/eval", img_name))
            image = cv2.resize(image, inp_shape, interpolation=cv2.INTER_AREA)
            image_edges = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_edges = skel(image_edges)
            transform = transforms.Compose(
                [
                    # transforms.ToPILImage(),
                    # transforms.Resize((128, 128)),
                    transforms.ToTensor()])
            x = transform(image_edges)
            edge_tensor = (x > 0.5).to(torch.float32).cuda()

            predicted_image = self.generator(edge_tensor.unsqueeze(0))
            img = predicted_image.detach().cpu()
            prediction = img
            edge = np.concatenate([edge_tensor.detach().cpu()] * 3)
            res = np.concatenate(
                [edge, prediction.detach().cpu()[0]],
                axis=1)
            self.logger.experiment.add_image("eval_imgs",
                                             res,
                                             self.current_epoch * len(test_images) + idx)

    def configure_optimizers(self):
        lr_g = 1e-3
        lr_d = 6e-4

        opt_g = torch.optim.AdamW(self.generator.parameters(), lr=lr_g, weight_decay=3e-3)
        opt_d = torch.optim.SGD(self.discriminator.parameters(), lr=lr_d, weight_decay=3e-3)

        return [opt_g,
                opt_d]#, \
               #[torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, 50),
                #torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, 50)]

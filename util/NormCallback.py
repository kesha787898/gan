import torch
from pytorch_lightning import Callback


class NormCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.norms_d = []
        self.norms_g = []

        self.i_g = 0
        self.i_d = 0

    def on_before_backward(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", loss: torch.Tensor) -> None:
        super().on_before_backward(trainer, pl_module, loss)

        for p in list(filter(lambda p: p.grad is not None, pl_module.generator.parameters())):
            self.norms_g.append(p.grad.data.norm(2).item())
        for p in list(filter(lambda p: p.grad is not None, pl_module.discriminator.parameters())):
            self.norms_d.append(p.grad.data.norm(2).item())
        if self.norms_g and max(self.norms_g) > 0:
            trainer.model.logger.experiment.add_scalar("gen_max_norm", max(self.norms_g), self.i_g)
            trainer.model.logger.experiment.add_scalar("gen_mean_norm", sum(self.norms_g) / len(self.norms_g), self.i_g)
            self.i_g += 1
            self.norms_g = []
        if self.norms_d and max(self.norms_d) > 0:
            trainer.model.logger.experiment.add_scalar("discr_max_norm", max(self.norms_d), self.i_d)
            trainer.model.logger.experiment.add_scalar("discr_mean_norm", sum(self.norms_d) / len(self.norms_d),
                                                       self.i_d)
            self.i_d += 1
            self.norms_d = []

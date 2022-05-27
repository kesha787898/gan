from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from models_src.Gan import GAN
from util import config
from util.NormCallback import NormCallback
from util.config import AVAIL_GPUS, BATCH_SIZE, X_PATH, Y_PATH
from util.data import PixToPixDataset
import torch

if config.device == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
if __name__ == '__main__':
    train_dataset = PixToPixDataset(X_PATH, Y_PATH)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=config.num_workers,
                                  pin_memory=config.device == 'cuda')

    model = GAN()
    checkpoint_callback = ModelCheckpoint(dirpath="chkpts", every_n_epochs=1)
    logger = TensorBoardLogger("tb_logs", name="my_model")
    trainer = Trainer(gpus=AVAIL_GPUS, max_epochs=1000000, progress_bar_refresh_rate=1, logger=logger,
                      callbacks=[checkpoint_callback, NormCallback()])
    trainer.fit(model, train_dataloader)
    trainer.save_checkpoint("example.ckpt")

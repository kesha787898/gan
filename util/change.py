from torch.utils.data import DataLoader
from tqdm import tqdm

from util.config import X_PATH, Y_PATH
from util.data import PixToPixDataset
import cv2
import numpy as np


if __name__ == '__main__':
    train_dataset = PixToPixDataset(X_PATH)

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0,
                                  pin_memory=True)
    itirator = iter(train_dataloader)
    for idx in tqdm(range(len((train_dataloader)))):
        x, y = itirator.__next__()
        img = np.concatenate([x[0].detach().cpu() * 255] * 3).transpose(1, 2, 0)
        # cv2.imshow(f"data_dir/data_out/{idx + 1}.png", img)
        # cv2.waitKey(0)
        cv2.imwrite(f"{Y_PATH}/{idx + 1}.png", img)

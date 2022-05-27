from torchvision.transforms import transforms

from models_src import Gan
import cv2
import numpy as np
img_name = r"D:\gan\eval\8.png"
path_to_chkpt = r"D:\gan\chkpts\epoch=999-step=39999.ckpt"

g = Gan.GAN()
g.load_from_checkpoint(path_to_chkpt)
mdl = g.generator

image = cv2.imread(img_name)
image=cv2.dilate(image, np.ones((3, 3)))
transform = transforms.ToTensor()

# Convert the image to PyTorch tensor
tensor = transform(image)[0].unsqueeze(0).unsqueeze(1)

pred = mdl(tensor)
img = pred[0].detach().cpu().numpy().transpose(1, 2, 0)
cv2.imshow("img", img)
cv2.waitKey(0)

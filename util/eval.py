from torchvision.transforms import transforms

from models_src import Gan
import cv2
import numpy as np

from util.config import inp_shape


class ImageTrnsformer():
    def __init__(self, chkpt_path):
        g = Gan.GAN()
        g.load_from_checkpoint(chkpt_path)
        self.mdl = g.generator

    def predict_image(self, image):
        image = cv2.dilate(image, np.ones((3, 3)))
        image = cv2.resize(image, inp_shape, interpolation=cv2.INTER_AREA)
        transform = transforms.ToTensor()
        tensor = transform(image)[0].unsqueeze(0).unsqueeze(1)
        pred = self.mdl(tensor)
        return pred[0].detach().cpu().numpy().transpose(1, 2, 0)


img_name = r"D:\gan\eval\photo_2022-02-26_18-53-59.jpg"
path_to_chkpt = r"D:\gan\chkpts\epoch=15-step=15.ckpt"
tf = ImageTrnsformer(path_to_chkpt)
image = cv2.imread(img_name)
img = tf.predict_image(image)
cv2.imshow("img", img)
cv2.waitKey(0)

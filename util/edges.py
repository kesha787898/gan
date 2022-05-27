import cv2 as cv
import numpy as np

prototxt = r"D:\gan\models\deploy.prototxt"
caffemodel = r"D:\gan\models\hed_pretrained_bsds.caffemodel"


class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = (inputShape[2] - targetShape[2]) // 2
        self.xstart = (inputShape[3] - targetShape[3]) // 2
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.ystart:self.yend, self.xstart:self.xend]]

cv.dnn_registerLayer('Crop', CropLayer)
net = cv.dnn.readNet(cv.samples.findFile(prototxt), cv.samples.findFile(caffemodel))


def get_edges(frame) -> np.array:
    inp = cv.dnn.blobFromImage(frame, scalefactor=1.0, size=(500, 500),
                               mean=(104.00698793, 116.66876762, 122.67891434),
                               swapRB=False, crop=False)
    net.setInput(inp)
    out = net.forward()
    out = out[0, 0]
    out = cv.resize(out, (frame.shape[1], frame.shape[0]))
    return out


if __name__ == '__main__':
    img = cv.imread(r"D:\gan\data\train\Qma1aZPn7iS1vxkfip6kjGjbA5EUPDaunsApJJ8mUt8pyT.jpg")
    cv.imshow("frame", img)
    cv.waitKey(0)
    cv.imshow("frame", get_edges(img))
    cv.waitKey(0)
#CatBoostClassifier(num_trees=config.catboost_trees,
#                                    task_type="GPU" if config.device == "cuda" else "CPU")
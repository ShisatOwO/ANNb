import sys
import PIL
import numpy as np

from PIL import Image
from sklearn.datasets import load_digits

from annb.layers import *
from annb.actfunc import *
from annb.optimizer import *
from annb.n_manager import NManager


digits = load_digits()


def show_img(img: [[float, ], ], scaling: int = 10) -> None:
    im = img.copy()
    img = np.zeros((len(im), len(im), 3), dtype=np.uint8)
    for x in range(len(im)):
        for y in range(len(im)):
            img[x, y] = [255 - ((im[x][y]/16)*255), ]*3
    img = Image.fromarray(img, "RGB")
    w, h = img.size
    img = img.transform((w*scaling, h*scaling), Image.EXTENT, (0, 0, w, h))
    img.show()


#TODO: ADAM optimizer implentieren


def main(*args, **kwargs) -> int:
    l1_dense = LayerInfo(DenseLayer(64, 64), Relu())
    l2_dense = LayerInfo(DenseLayer(64, 15), Relu())
    l3_dense = LayerInfo(DenseLayer(15, 7), Relu())
    l4_dense = LayerInfo(DenseLayer(7, 10), Softmax())

    n_manager = NManager(l1_dense, l4_dense)
    n_manager.add_layer(l2_dense)
    n_manager.add_layer(l3_dense)

    n_manager.add_trainer(BatchSGD(0.9, 5e-4, 0.2))

    data = digits.images.reshape((len(digits.images), -1))
    data = data[:1700]
    reference = digits.target[:1700]

    n_manager.train(50000, data, reference)

    """datax = np.array_split(data, 17)
    referencex = np.array_split(reference, 17)

    for dat, ref in zip(datax, referencex):
        n_manager.train(10000, dat, ref)"""

    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1::]))

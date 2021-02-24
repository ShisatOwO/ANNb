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


def main(*args, **kwargs) -> int:
    l1_dense = LayerInfo(DenseLayer(64, 64), Relu())
    l2_dense = LayerInfo(DenseLayer(64, 15), Relu())
    l3_dense = LayerInfo(DenseLayer(15, 7), Relu())
    l4_dense = LayerInfo(DenseLayer(7, 10), Softmax())

    n_manager = NManager(l1_dense, l4_dense)
    n_manager.add_layer(l2_dense)
    n_manager.add_layer(l3_dense)

    n_manager.add_trainer(BatchSGD(0.2, 5e-4, 0.1))

    data = digits.images.reshape((len(digits.images), -1))
    data = data[:1700]
    reference = digits.target[:1700]

    n_manager.train(2500, data, reference)

    test = digits.images.reshape((len(digits.images), -1))
    test_dat = test[1700:]
    test_img = digits.images[1700:]
    test_ref = digits.target[1700:]

    x = input("PLease Choose one out of 96 samples: ")
    while x != "exit":
        x = int(x)
        if x >= 0 and 96 >= x:
            show_img(test_img[x])
            print(f"Showing the network a {test_ref[x]}...")
            n_manager.guess(test_dat[x])
            print(f"The network thinks that this is a {np.argmax(n_manager.out)}")
            print(f"\n")
            print(n_manager.out)
            print(l3_dense.layer.out.copy())
        x = input("\n\nPlease choose one out of 96 samples: ")

    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1::]))

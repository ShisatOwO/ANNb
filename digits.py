import sys
import PIL
import numpy as np

from PIL import Image
from sklearn.datasets import load_digits

from annb.layers import *
from annb.actfunc import *
from annb.optimizer import *
from annb.n_manager import NManager


# Der Datensatz
digits = load_digits()


def show_img(img: [[float, ], ], scaling: int = 10) -> None:
    """Methode zum Anzeigen der Bilder"""
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
    # Erstellung der Einzelnen Layer
    l1_dense = LayerInfo(DenseLayer(64, 64), Relu())    # Input Layer
    l2_dense = LayerInfo(DenseLayer(64, 15), Relu())    # Hidden Layer 1
    l3_dense = LayerInfo(DenseLayer(15, 7), Relu())     # Hidden Layer 2
    l4_dense = LayerInfo(DenseLayer(7, 10), Softmax())  # Output Layer

    n_manager = NManager(l1_dense, l4_dense)    # Initialisierung des Netzwerk Managers mit dem In- und Output Layer
    n_manager.add_layer(l2_dense)               # Hidden Layer 1 an den Netzwerk Mananger geben
    n_manager.add_layer(l3_dense)               # Hidden Layer 1 an den Netzwerk Mananger geben

    n_manager.add_trainer(BatchSGD(0.2, 5e-4, 0.1))     # Einen Trainer für Das netzwerk erstellen

    # Den Datensatz vorbereiten. Form wird vom 8x8 zu 1x64 geändert
    data = digits.images.reshape((len(digits.images), -1))
    test = digits.images.reshape((len(digits.images), -1))

    # Zum Trainieren möchte ich nur 1700 von den 1797 Bildern benutzen
    data = data[:1700]
    reference = digits.target[:1700]

    # Die restlichen 97 Bilder können zum Testen verwendet werden.
    test_dat = test[1700:]
    test_img = digits.images[1700:]
    test_ref = digits.target[1700:]

    # Das Netzwerk trainieren.
    n_manager.train(2500, data, reference)

    # Das Netzwerk ist jetzt trainiert. nun können dem Netzwerk die Test Bilder gezeigt werden.
    # Dazu wird der Benutzer aufgefordert, ein Bild aus dem Test Datensetz auszuwählen
    x = input("Please choose one out of 96 imgaes to show it to the network: ")
    while x != "exit":
        x = int(x)
        if 0 <= x <= 96:
            show_img(test_img[x])
            print(f"Showing the network a {test_ref[x]}...")
            n_manager.guess(test_dat[x])
            print(f"The network thinks that this is a {np.argmax(n_manager.out)}")
            print(f"\nOutput of the Output Layer")
            print(n_manager.out)
            print("\n")
        x = input("\n\nPlease choose one out of 96 imgaes to show it to the network: ")

    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1::]))

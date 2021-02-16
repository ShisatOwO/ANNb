import sys
import nnfs

from nnfs.datasets import spiral_data
from annb.network_manager import NetworkManager
from annb.layers import *
from annb.func import *
from annb.activation_functions import *

nnfs.init()


def main(*args, **kwargs) -> int:
    x, y = spiral_data(100, 3)

    l_in = LayerInfo(DenseLayer(2, 3), Relu())
    l_out = LayerInfo(DenseLayer(3, 3), Softmax())
    n_manager = NetworkManager(l_in, l_out)

    n_manager.guess(x)
    #loss = np.mean(n_manager.calc_loss(x, y))
    print(n_manager.out[:5:])
    #print("Loss:\t", loss)
    #print("Acc:\t", n_manager.calc_acc(y))

    return 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1::]))


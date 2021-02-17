import sys
import nnfs

from nnfs.datasets import spiral_data
from annb.network_manager import NetworkManager
from annb.layers import *
from annb.func import *
from annb.activation_functions import *


def main(*args, **kwargs) -> int:
    # Training Data
    x, y = spiral_data(100, 3)

    # Layer erstellen
    l1_inp_dense = LayerInfo(DenseLayer(2, 3), Relu())
    l2_out_dense = LayerInfo(DenseLayer(3, 3), Softmax())

    # Network manager mit Layern initialisieren
    n_manager = NetworkManager(l1_inp_dense, l2_out_dense)

    # Ein forward pass durch netzwerk +
    loss = np.mean(n_manager.calc_loss(x, y))

    # Backward pass
    """l2_out_dense.layer.pass_backward(n_manager.out, l2_out_dense.act_func, y)
    l1_inp_dense.layer.pass_backward(l2_out_dense.layer.dinp, l1_inp_dense.act_func)"""

    n_manager.back_prop(y)

    print(n_manager.out[:5])
    print(loss)
    print(l1_inp_dense.layer.dweights, "\n")
    print(l1_inp_dense.layer.dbiases, "\n")
    print(l2_out_dense.layer.dweights, "\n")
    print(l2_out_dense.layer.dbiases, "\n")



    return 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1::]))


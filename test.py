import sys
import nnfs

from nnfs.datasets import spiral_data

from annb.n_manager import NManager
from annb.layers import *
from annb.actfunc import *
from annb.optimizer import *


nnfs.init()


def main(*args, **kwargs) -> int:
    # Test traing data
    inp, ref = spiral_data(100, 3)

    # Setup
    d1_in = LayerInfo(DenseLayer(2, 512), Relu())
    lh1 = LayerInfo(DenseLayer(512, 10), Relu())
    d2_out = LayerInfo(DenseLayer(10, 3), Softmax())
    nmanager = NManager(d1_in, d2_out)
    nmanager.add_layer(lh1)
    nmanager.guess(inp)
    old_acc = nmanager.calc_acc(ref)

    # Training
    for i in range(6000+1):
        nmanager.guess(inp)
        loss = nmanager.calc_gradient(ref)
        nmanager.optimize(BatchSGD())
        print(f"Iteration:\t {i}")
        print(f"Acc:\t {nmanager.calc_acc(ref)} ")
        print(f"Loss:\t {np.mean(loss)}\n")

    print(f"Acc improvement: {nmanager.calc_acc(ref) - old_acc}")

    return 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1::]))

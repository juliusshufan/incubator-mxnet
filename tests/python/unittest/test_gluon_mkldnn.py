import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.test_utils import assert_almost_equal
from common import setup_module, with_seed
import numpy as np
import random
from nose.tools import raises


def check_layer_forward(net, x):
    x_hybrid = x.copy()
    x.attach_grad()
    x_hybrid.attach_grad()
    net.collect_params().initialize()
    with mx.autograd.record():
        out1 = net(x)
    out1.backward()
    net.hybridize()
    with mx.autograd.record():
        out2 = net(x_hybrid)
        print out2
    out2.backward()
    mx.test_utils.assert_almost_equal(x.grad.asnumpy(), x_hybrid.grad.asnumpy(), rtol=1e-5, atol=1e-6)
    mx.test_utils.assert_almost_equal(out1.asnumpy(), out2.asnumpy(), rtol=1e-5, atol=1e-6)


@with_seed()
def test_reshape_conv():
    class Net(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.conv0 = nn.Conv2D(64, (3, 3))

        def hybrid_forward(self, F, x):
            x_reshape = x.reshape((0, 0, 448, 112))
            out = self.conv0(x_reshape)
            return out
    x = mx.nd.random.uniform(shape=(32, 3, 224, 224))
    net = Net()
    check_layer_forward(net, x)


@with_seed()
def test_reshape_conv_reshape_conv():
    class Net(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.conv0 = nn.Conv2D(64, (3, 3))
                self.conv1 = nn.Conv2D(256, (3, 3))

        def hybrid_forward(self, F, x):
            x_reshape = x.reshape((0, 0, 448, 112))
            y = self.conv0(x_reshape)
            y_reshape = y.reshape((0, 0, 223, 220))
            out = self.conv1(y_reshape)
            return out
    x = mx.nd.random.uniform(shape=(32, 3, 224, 224))
    net = Net()
    check_layer_forward(net, x)


@with_seed()
def test_slice_conv():
    class Net(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.conv0 = nn.Conv2D(64, (3, 3))

        def hybrid_forward(self, F, x):
            x_slice = x.slice(begin=(0, 2, 0, 0), end=(32, 5, 224, 224))
            out = self.conv0(x_slice)
            return out
    x = mx.nd.random.uniform(shape=(32, 6, 224, 224))
    net = Net()
    check_layer_forward(net, x)


@with_seed()
def test_slice_conv_slice_conv():
    class Net(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.conv0 = nn.Conv2D(64, (3, 3))
                self.conv1 = nn.Conv2D(256, (3, 3))

        def hybrid_forward(self, F, x):
            x_slice = x.slice(begin=(0, 2, 0, 0), end=(32, 5, 224, 224))
            y = self.conv0(x_slice)
            y_slice = y.slice(begin=(0, 32, 0, 0), end=(32, 64, 222, 222))
            out = self.conv1(y_slice)
            return out
    x = mx.nd.random.uniform(shape=(32, 6, 224, 224))
    net = Net()
    check_layer_forward(net, x)


@with_seed()
def test_slice_conv_reshape_conv():
    class Net(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.conv0 = nn.Conv2D(64, (3, 3))
                self.conv1 = nn.Conv2D(256, (3, 3))

        def hybrid_forward(self, F, x):
            x_slice = x.slice(begin=(0, 0, 1, 1), end=(32, 3, 225, 225))
            y = self.conv0(x_slice)
            y_reshape = y.reshape((0, 0, 444, 111))
            out = self.conv1(y_reshape)
            return out

    x = mx.nd.random.uniform(shape=(32, 3, 299, 299))
    net = Net()
    check_layer_forward(net, x)


def test_reshape_conv_slice_conv():
    """
    This test will test gluon Conv2d computation on mkldnn with ndarray reshape and slice
    """
    class Net(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.conv0 = nn.Conv2D(64, (3, 3))
                self.conv1 = nn.Conv2D(256, (3, 3))

        def hybrid_forward(self, F, x):
            x_reshape = x.reshape((0, 0, 448, 112))
            y = self.conv0(x_reshape)
            y_slice = y.slice(begin=(0, 32, 0, 0), end=(32, 64, 446, 110))
            out = self.conv1(y_slice)
            return out
    x = mx.nd.random.uniform(shape=(32, 6, 224, 224))
    net = Net()
    check_layer_forward(net, x)


@with_seed()
def test_reshape_dense():
    class Net(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                channel0 = random.randint(1, 1000)
                self.dense0 = nn.Dense(channel0)

        def hybrid_forward(self, F, x):
            x_reshape = x.reshape((8, 64, 600, -1))
            out = self.dense0(x)
            return out

    x = mx.nd.random.uniform(shape=(16, 128, 300, 300))
    net = Net()
    check_layer_forward(net, x)


@with_seed()
def test_slice_dense():
    class Net(gluon.HybridBlock):
        def __init__(self, slice, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                channel0 = random.randint(1, 1000)
                self.dense0 = nn.Dense(channel0)
                self.slice = slice

        def hybrid_forward(self, F, x):
            x_slice = x.slice(begin=tuple(self.slice[0]),
                              end=tuple(self.slice[1]))
            out = self.dense0(x_slice)
            return out

    x = mx.nd.random.uniform(shape=(16, 128, 300, 300))
    slice = [[0, 64, 50, 0], [8, 128, 300, 300]]
    net = Net(slice)
    check_layer_forward(net, x)


@with_seed()
def test_slice_dense_slice_dense():
    class Net(gluon.HybridBlock):
        def __init__(self, slice, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                channel0 = 50
                channel1 = random.randint(1, 1000)
                self.dense0 = nn.Dense(channel0)
                self.dense1 = nn.Dense(channel1)
                self.slice = slice

        def hybrid_forward(self, F, x):
            x_slice = x.slice(begin=tuple(self.slice[0]), end=tuple(self.slice[1]))
            y = self.dense0(x_slice)
            y_slice = y.slice(begin=(4, 0), end=(-1, 10))
            out = self.dense1(y_slice)
            return out

    x = mx.nd.random.uniform(shape=(16, 128, 300, 300))
    slice = [[0, 64, 50, 0], [8, 128, 300, 300]]
    net = Net(slice)
    check_layer_forward(net, x)


@with_seed()
def test_reshape_dense_reshape_dense():
    class Net(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                channel0 = random.randint(1, 1000)
                channel1 = random.randint(1, 1000)
                self.dense0 = nn.Dense(channel0)
                self.dense1 = nn.Dense(channel1)

        def hybrid_forward(self, F, x):
            x_reshape = x.reshape((8, 64, 600, -1))
            y = self.dense0(x_reshape)
            y_reshape = y.reshape((1, -1))
            out = self.dense1(y_reshape)
            return out

    x = mx.nd.random.uniform(shape=(16, 128, 300, 300))
    net = Net()
    check_layer_forward(net, x)


@with_seed()
def test_slice_dense_reshape_dense():
    class Net(gluon.HybridBlock):
        def __init__(self, slice, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                channel0 = random.randint(1, 1000)
                channel1 = random.randint(1, 1000)
                self.dense0 = nn.Dense(channel0)
                self.dense1 = nn.Dense(channel1)
                self.slice = slice

        def hybrid_forward(self, F, x):
            x_slice = x.slice(begin=tuple(self.slice[0]), end=tuple(self.slice[1]))
            y = self.dense0(x_slice)
            y_reshape = y.reshape((1, -1))
            out = self.dense1(y_reshape)
            return out

    x = mx.nd.random.uniform(shape=(16, 128, 300, 300))
    slice = [[0, 64, 50, 0], [8, 128, 300, 300]]
    net = Net(slice)
    check_layer_forward(net, x)


@with_seed()
def test_reshape_dense_slice_dense():
    class Net(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                channel0 = 800
                channel1 = random.randint(1, 1000)
                self.dense0 = nn.Dense(channel0)
                self.dense1 = nn.Dense(channel1)

        def hybrid_forward(self, F, x):
            x_reshape = x.reshape((8, 64, 600, -1))
            y = self.dense0(x_reshape)
            y_slice = y.slice(begin=(0, 500), end=(8, 628))
            out = self.dense1(y_slice)
            return out

    x = mx.nd.random.uniform(shape=(16, 128, 300, 300))
    net = Net()
    check_layer_forward(net, x)


@with_seed()
def test_reshape_batchnorm():
    class Net(gluon.HybridBlock):
        def __init__(self, shape, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.conv0 = nn.Conv2D(128, (1, 1))
                self.bn0 = nn.BatchNorm()
                self.reshape = shape

        def hybrid_forward(self, F, x):
            x_in = self.conv0(x)
            x_reshape = x_in.reshape(self.reshape)
            out = self.bn0(x_reshape)
            return out

    x = mx.nd.random.uniform(shape=(16, 128, 256, 256))
    shape = (32, 512, 128, -1)
    net = Net(shape)
    check_layer_forward(net, x)


@with_seed()
def test_slice_batchnorm():
    class Net(gluon.HybridBlock):
        def __init__(self, slice, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.conv0 = nn.Conv2D(128, (1, 1))
                self.bn0 = nn.BatchNorm(3)
                self.slice = slice

        def hybrid_forward(self, F, x):
            x_in = self.conv0(x)
            x_slice = x_in.slice(begin=tuple(self.slice[0]),
                              end=tuple(self.slice[1]))
            out = self.bn0(x_slice)
            return out

    x = mx.nd.random.uniform(shape=(16, 128, 256, 256))
    slice = [[0, 64, 50, 0], [8, 128, 256, 256]]
    net = Net(slice)
    check_layer_forward(net, x)


@with_seed()
def test_slice_batchnorm_slice_batchnorm():
    class Net(gluon.HybridBlock):
        def __init__(self, slice, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.conv0 = nn.Conv2D(128, (1, 1))
                self.bn0 = nn.BatchNorm(3)
                self.bn1 = nn.BatchNorm(1)
                self.slice = slice

        def hybrid_forward(self, F, x):
            x_in = self.conv0(x)
            x_slice = x_in.slice(begin=tuple(self.slice[0][0]), end=tuple(self.slice[0][1]))
            y = self.bn0(x_slice)
            y_slice = y.slice(begin=tuple(self.slice[1][0]), end=tuple(self.slice[1][1]))
            out = self.bn1(y_slice)
            return out

    x = mx.nd.random.uniform(shape=(16, 128, 256, 256))
    slice = [[[0, 64, 50, 0], [8, 128, 200, 256]], [[4, 50, 0, 128], [7, -1, -1, -1]]]
    net = Net(slice)
    check_layer_forward(net, x)


@with_seed()
def test_reshape_batchnorm_reshape_batchnorm():
    class Net(gluon.HybridBlock):
        def __init__(self, shape, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.conv0 = nn.Conv2D(128, (1, 1))
                self.bn0 = nn.BatchNorm(0)
                self.bn1 = nn.BatchNorm(2)
                self.reshape = shape

        def hybrid_forward(self, F, x):
            x_in = self.conv0(x)
            x_reshape = x_in.reshape(self.reshape[0])
            y = self.bn0(x_reshape)
            y_reshape = y.reshape(self.reshape[1])
            out = self.bn1(y_reshape)
            return out

    x = mx.nd.random.uniform(shape=(16, 128, 256, 512))
    shape = [(8, 256, 128, -1), (32, 128, 512, -1)]
    net = Net(shape)
    check_layer_forward(net, x)


@with_seed()
def test_slice_batchnorm_reshape_batchnorm():
    class Net(gluon.HybridBlock):
        def __init__(self, shape, slice, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.conv0 = nn.Conv2D(128, (1, 1))
                self.bn0 = nn.BatchNorm(0)
                self.bn1 = nn.BatchNorm(2)
                self.reshape = shape
                self.slice = slice

        def hybrid_forward(self, F, x):
            x_in = self.conv0(x)
            x_slice = x_in.slice(begin=tuple(self.slice[0]), end=tuple(self.slice[1]))
            y = self.bn0(x_slice)
            y_reshape = y.reshape(self.reshape)
            out = self.bn1(y_reshape)
            return out

    x = mx.nd.random.uniform(shape=(16, 128, 256, 256))
    slice = [[0, 64, 50, 0], [8, 128, 200, 256]]
    shape = (1, 128, 256, -1)
    net = Net(shape, slice)
    check_layer_forward(net, x)


@with_seed()
def test_reshape_batchnorm_slice_batchnorm():
    class Net(gluon.HybridBlock):
        def __init__(self, shape, slice, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.conv0 = nn.Conv2D(128, (1, 1))
                self.bn0 = nn.BatchNorm(2)
                self.bn1 = nn.BatchNorm(0)
                self.reshape = shape
                self.slice = slice

        def hybrid_forward(self, F, x):
            x_in = self.conv0(x)
            x_reshape = x_in.reshape(self.reshape)
            y = self.bn0(x_reshape)
            y_slice = y.slice(begin=tuple(self.slice[0]), end=tuple(self.slice[1]))
            out = self.bn1(y_slice)
            return out

    x = mx.nd.random.uniform(shape=(16, 128, 256, 256))
    slice = [[0, 0, 50, 0], [8, 1, -1, 100]]
    shape = (128, 1, 256, -1)
    net = Net(shape, slice)
    check_layer_forward(net, x)

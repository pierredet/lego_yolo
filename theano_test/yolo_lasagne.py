import cPickle as pkl

import lasagne
from lasagne.layers import InputLayer, NonlinearityLayer, ReshapeLayer
# The layer below requires GPU, replace Conv2DDNNLayer by COnv2DLayer
# and remove the flip_filter option
from lasagne.layers import ConcatLayer, batch_norm
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax
from lasagne.nonlinearities import LeakyRectify as leaky

def build_model(path=None):
    image_w = 416
    net = {}
    net['input'] = InputLayer((1, 3, image_w, image_w))
    net['0conv'] = batch_norm(ConvLayer(
        net['input'], 32, 3, pad=1,
        nonlinearity=leaky(leakiness=0.1)))

    net['1pool'] = PoolLayer(net['0conv'], 2, stride=2, mode='max')

    net['2conv'] = batch_norm(ConvLayer(
        net['1pool'], 64, 3, pad=1, nonlinearity=leaky(leakiness=0.1),
        flip_filters=False))

    net['3pool'] = PoolLayer(net['2conv'], 2, stride=2, mode='max')

    net['4conv'] = batch_norm(ConvLayer(
        net['3pool'], 128, 3, pad=1, nonlinearity=leaky(leakiness=0.1) ,
        flip_filters=False))
    net['5conv'] = batch_norm(ConvLayer(net['4conv'], 64, 1, pad=0,
                                        nonlinearity=leaky(leakiness=0.1) ,
                                        flip_filters=False))
    net['6conv'] = batch_norm(ConvLayer(net['5conv'], 128, 3, pad=1, nonlinearity=leaky(leakiness=0.1)))

    net['7pool'] = PoolLayer(net['6conv'], 2, stride=2, mode='max')

    net['8conv'] = batch_norm(ConvLayer(net['7pool'], 256, 3, pad=1, nonlinearity=leaky(leakiness=0.1)))
    net['9conv'] = batch_norm(ConvLayer(net['8conv'], 128, 1, pad=0, nonlinearity=leaky(leakiness=0.1)))
    net['10conv'] = batch_norm(ConvLayer(net['9conv'], 256, 3, pad=1, nonlinearity=leaky(leakiness=0.1)))

    net['11pool'] = PoolLayer(net['10conv'], 2, stride=2, pad=0, mode='max')

    net['12conv'] = batch_norm(ConvLayer(net['11pool'], 512, 3, pad=1, nonlinearity=leaky(leakiness=0.1)))
    net['13conv'] = batch_norm(ConvLayer(net['12conv'], 256, 1, pad=0, nonlinearity=leaky(leakiness=0.1)))
    net['14conv'] = batch_norm(ConvLayer(net['13conv'], 512, 3, pad=1, nonlinearity=leaky(leakiness=0.1)))
    net['15conv'] = batch_norm(ConvLayer(net['14conv'], 256, 1, pad=0, nonlinearity=leaky(leakiness=0.1)))
    net['16conv'] = batch_norm(ConvLayer(net['15conv'], 512, 3, pad=1, nonlinearity=leaky(leakiness=0.1)))

    net['17pool'] = PoolLayer(net['16conv'], 2, stride=2, mode='max')

    net['18conv'] = batch_norm(ConvLayer(net['17pool'], 1024, 3, pad=1, nonlinearity=leaky(leakiness=0.1)))
    net['19conv'] = batch_norm(ConvLayer(net['18conv'], 512, 1, pad=0, nonlinearity=leaky(leakiness=0.1)))
    net['20conv'] = batch_norm(ConvLayer(net['19conv'], 1024, 3, pad=1, nonlinearity=leaky(leakiness=0.1)))
    net['21conv'] = batch_norm(ConvLayer(net['20conv'], 512, 1, pad=0, nonlinearity=leaky(leakiness=0.1)))
    net['22conv'] = batch_norm(ConvLayer(net['21conv'], 1024, 3, pad=1, nonlinearity=leaky(leakiness=0.1)))
    net['23conv'] = batch_norm(ConvLayer(net['22conv'], 1024, 3, pad=1, nonlinearity=leaky(leakiness=0.1)))
    net['24conv'] = batch_norm(ConvLayer(net['23conv'], 1024, 3, pad=1, nonlinearity=leaky(leakiness=0.1)))

    shp = net['16conv'].output_shape
    net['25reshape'] = ReshapeLayer(net['16conv'],(shp[0],shp[1]*4,shp[2]/2,shp[3]/2))
    net['26merge'] = ConcatLayer((net['25reshape'], net['24conv']))

    net['27conv'] = batch_norm(ConvLayer(net['26merge'], 1024, 3, pad=1, nonlinearity=leaky(leakiness=0.1)))
    net['28conv'] = ConvLayer(net['27conv'], 425, 1, pad=0, nonlinearity=leaky(leakiness=0.1))



    # import operator
    # for key,layer in sorted(net.items(), key=operator.itemgetter(0)):
    #    print key, layer.output_shape
    #30 detection TODO figure out the last layer


    if path is not None:
        values = pkl.load(open(path))
        lasagne.layers.set_all_param_values(net['28conv'], values)
    return net

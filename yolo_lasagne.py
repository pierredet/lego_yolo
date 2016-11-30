import cPickle as pkl

import lasagne
from lasagne.layers import InputLayer, NonlinearityLayer, ReshapeLayer
# The layer below requires GPU, replace Conv2DDNNLayer by COnv2DLayer
# and remove the flip_filter option
from lasagne.layers import ConcatLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax

def build_model(path=None):
    image_w = 416
    net = {}
    net['input'] = InputLayer((1, 3, image_w, image_w))
    net['0conv'] = ConvLayer(net['input'], 32, 3, pad=1, nonlinearity=lasagne.nonlinearities.LeakyRectify(leakiness=0.1) , flip_filters=False)

    net['1pool'] = PoolLayer(net['0conv'], 2, stride=2, mode='max')

    net['2conv'] = ConvLayer(net['1pool'], 64, 3, pad=1, nonlinearity=lasagne.nonlinearities.LeakyRectify(leakiness=0.1) , flip_filters=False)

    net['3pool'] = PoolLayer(net['2conv'], 2, stride=2, mode='max')

    net['4conv'] = ConvLayer(net['3pool'], 128, 3, pad=1, nonlinearity=lasagne.nonlinearities.LeakyRectify(leakiness=0.1) , flip_filters=False)
    net['5conv'] = ConvLayer(net['4conv'], 64, 1, pad=0, nonlinearity=lasagne.nonlinearities.LeakyRectify(leakiness=0.1) , flip_filters=False)
    net['6conv'] = ConvLayer(net['5conv'], 128, 3, pad=1, nonlinearity=lasagne.nonlinearities.LeakyRectify(leakiness=0.1) , flip_filters=False)

    net['7pool'] = PoolLayer(net['6conv'], 2, stride=2, mode='max')

    net['8conv'] = ConvLayer(net['7pool'], 256, 3, pad=1, nonlinearity=lasagne.nonlinearities.LeakyRectify(leakiness=0.1) , flip_filters=False)
    net['9conv'] = ConvLayer(net['8conv'], 128, 1, pad=0, nonlinearity=lasagne.nonlinearities.LeakyRectify(leakiness=0.1) , flip_filters=False)
    net['10conv'] = ConvLayer(net['9conv'], 256, 3, pad=1, nonlinearity=lasagne.nonlinearities.LeakyRectify(leakiness=0.1) , flip_filters=False)

    net['11pool'] = PoolLayer(net['10conv'], 2, stride=2, pad=0, mode='max')

    net['12conv'] = ConvLayer(net['11pool'], 512, 3, pad=1, nonlinearity=lasagne.nonlinearities.LeakyRectify(leakiness=0.1) , flip_filters=False)
    net['13conv'] = ConvLayer(net['12conv'], 256, 1, pad=0, nonlinearity=lasagne.nonlinearities.LeakyRectify(leakiness=0.1) , flip_filters=False)
    net['14conv'] = ConvLayer(net['13conv'], 512, 3, pad=1, nonlinearity=lasagne.nonlinearities.LeakyRectify(leakiness=0.1) , flip_filters=False)
    net['15conv'] = ConvLayer(net['14conv'], 256, 1, pad=0, nonlinearity=lasagne.nonlinearities.LeakyRectify(leakiness=0.1) , flip_filters=False)
    net['16conv'] = ConvLayer(net['15conv'], 512, 3, pad=1, nonlinearity=lasagne.nonlinearities.LeakyRectify(leakiness=0.1) , flip_filters=False)

    net['17pool'] = PoolLayer(net['16conv'], 2, stride=2, mode='max')

    net['18conv'] = ConvLayer(net['17pool'], 1024, 3, pad=1, nonlinearity=lasagne.nonlinearities.LeakyRectify(leakiness=0.1) , flip_filters=False)
    net['19conv'] = ConvLayer(net['18conv'], 512, 1, pad=0, nonlinearity=lasagne.nonlinearities.LeakyRectify(leakiness=0.1) , flip_filters=False)
    net['20conv'] = ConvLayer(net['19conv'], 1024, 3, pad=1, nonlinearity=lasagne.nonlinearities.LeakyRectify(leakiness=0.1) , flip_filters=False)
    net['21conv'] = ConvLayer(net['20conv'], 512, 1, pad=0, nonlinearity=lasagne.nonlinearities.LeakyRectify(leakiness=0.1) , flip_filters=False)
    net['22conv'] = ConvLayer(net['21conv'], 1024, 3, pad=1, nonlinearity=lasagne.nonlinearities.LeakyRectify(leakiness=0.1) , flip_filters=False)
    net['23conv'] = ConvLayer(net['22conv'], 1024, 3, pad=1, nonlinearity=lasagne.nonlinearities.LeakyRectify(leakiness=0.1) , flip_filters=False)
    net['24conv'] = ConvLayer(net['23conv'], 1024, 3, pad=1, nonlinearity=lasagne.nonlinearities.LeakyRectify(leakiness=0.1) , flip_filters=False)

    shp = net['16conv'].output_shape
    net['25reshape'] = ReshapeLayer(net['16conv'],(shp[0],shp[1]*4,shp[2]/2,shp[3]/2))
    net['26merge'] = ConcatLayer((net['25reshape'], net['24conv']))

    net['27conv'] = ConvLayer(net['26merge'], 1024, 3, pad=1, nonlinearity=lasagne.nonlinearities.LeakyRectify(leakiness=0.1) , flip_filters=False)
    net['28conv'] = ConvLayer(net['27conv'], 425, 1, pad=0, nonlinearity=lasagne.nonlinearities.LeakyRectify(leakiness=0.1) , flip_filters=False)

    # import operator
    # for key,layer in sorted(net.items(), key=operator.itemgetter(0)):
    #    print key, layer.output_shape
    #30 detection TODO figure out the last layer


    if path is not None:
        # values = pkl.load(open('vgg19_normalized.pkl'))['param values']
        # lasagne.layers.set_all_param_values(net['pool5'], values)
        # TODO weight loading
    return net

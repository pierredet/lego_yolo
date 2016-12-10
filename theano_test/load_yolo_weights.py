import numpy as np
import cPickle as pkl
import sys
from yolo_lasagne import build_model
#from yolo_test import build_model
import lasagne


def blob_size(shape):
    res = 4
    for s in shape:
        res *= s
    return res

def rec_filter(li):
    """
    recognise batch norm in the shape vector and rorder it to read it in the right order
    """
    if li == []:
        return []
    elif len(li)<5:
        return li
    else:
        if li[1]==li[2]==li[3]==li[4] and len(li[1])==1:
            return li[1:5]+[li[0]]+rec_filter(li[5:])
        else:
            return [li[0]]+ rec_filter(li[1:])

def rec_filter2(li):
    if li == []:
        return []
    elif len(li)<5:
        return li
    else:
        if li[0].shape==li[1].shape==li[2].shape==li[3].shape and len(li[1].shape)==1:
            return [li[4]]+li[1:3]+[np.sqrt(1/(0.0001+li[3]))]+[li[0]]+rec_filter2(li[5:])
        else:
            return [li[0]]+ rec_filter2(li[1:])



def main(path):
    weights = []
    net = build_model()
    # print lasagne.layers.get_all_layers(net['28conv'])
    wei_shapes = lasagne.layers.get_all_param_values(net['28conv'] )
    shapes = map(lambda w: w.shape, wei_shapes)
    shapes = rec_filter(shapes)
    print np.sum(map(lambda s: blob_size(s),shapes))
    netWeightsFloat = np.fromfile(path, dtype=np.float32)
    netWeights = netWeightsFloat[4:]

    count = 0
    for i,shape in enumerate(shapes):
        if i == len(shapes)-1:
            import pdb; pdb.set_trace()
        weights.append(netWeights[count:count+blob_size(shape)/4].reshape(
            shape))
        count += blob_size(shape)/4
    weights = rec_filter2(weights)
    #print "shape correspondance",shapes ==  map(lambda w: w.shape, weights)
    print count
    # Filter viz
    """import PIL.Image
    w = weights[0].reshape(weights[0].shape[0]*weights[0].shape[1],weights[0].shape[2],weights[0].shape[3])
    for i in range(weights[0].shape[0]):
        wmin = float(w[i].min())
        wmax = float(w[i].max())
        w[i] *= (255.0/float(wmax-wmin))
        w[i] += abs(wmin)*(255.0/float(wmax-wmin))
        W = np.zeros((30,30))
        for x in (0,1,2):
            for y in (0,1,2):
                W[10*x:10*x+10,10*y:10*y+10] = w[i,x,y]
        PIL.Image.fromarray(W).convert("RGBA").save(str(i)+".png") """
    with open("weights.pkl", 'wb') as file:
        pkl.dump(weights,file)

if __name__ == '__main__':
    if len(sys.argv) is not 2:
        print("expect as one and only arg the path to yolo.weights !")
    else:
        main(sys.argv[1])

    """ shapes = [(32, 3, 3), (64, 3, 3), (128, 3, 3), (64, 1, 1), (128, 3, 3),
              (256, 3, 3), (128, 1, 1), (256, 3, 3), (512, 3, 3), (256, 1, 1),
              (512, 3, 3), (1024, 3, 3), (512, 1, 1), (1024, 3, 3),
              (1024, 3, 3), (1024, 3, 3), (1024, 3, 3), (425, 1, 1)]"""

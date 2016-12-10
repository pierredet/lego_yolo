import theano
import theano.tensor as T
import lasagne

import numpy as np
from yolo_lasagne import build_model
import PIL.Image


net = build_model("weights.pkl")
input_im_theano = T.tensor4()
# outputs = lasagne.layers.get_output(net['28conv'], input_im_theano)
outputs = lasagne.layers.get_output(net['28conv'], input_im_theano, deterministic=True)
forward = theano.function([input_im_theano], outputs)#, deterministic=True)
img =np.loadtxt("input").reshape((3,416,416)).astype('float32')

"""img = np.array(PIL.Image.open("input.jpg"))
img = np.swapaxes(np.swapaxes(img, 1, 2), 0, 1)

    # Convert RGB to BGR
img = img[::-1, :, :]"""
inp = np.zeros((1,3,416,416),dtype=np.float32)

inp[0] = img

out = forward(inp)
# import pdb; pdb.set_trace()
out2 =np.loadtxt("29").reshape((425, 13, 13)).astype('float32')
# out2 =np.loadtxt("0").reshape((32,416,416)).astype('float32')
print np.linalg.norm(out[0]-out2)
import pdb; pdb.set_trace()

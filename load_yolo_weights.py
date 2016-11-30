import numpy as np
import cPickle as pkl
import sys


def blob_size(shape):
    res = 4
    for s in shape:
        res *= s
    return res

def main(path):
    weights = []
    with open(path, 'rb') as file:
        shapes = [(32, 3, 3), (64, 3, 3), (128, 3, 3), (64, 1, 1), (128, 3, 3),
                  (256, 3, 3), (128, 1, 1), (256, 3, 3), (512, 3, 3), (256, 1, 1),
                  (512, 3, 3), (1024, 3, 3), (512, 1, 1), (1024, 3, 3),
                  (1024, 3, 3), (1024, 3, 3), (1024, 3, 3), (425, 1, 1)]
        count = 0
        for shape in shapes:
            count += blob_size(shape)
            bytestr =file.read(blob_size(shape))
            weights.append(np.fromstring(bytestr,dtype=np.float32).reshape(
                shape))
    print count
    with open("weights.pkl", 'wb') as file:
        pkl.dump(weights,file)

if __name__ == '__main__':
    if len(sys.argv) is not 2:
        print("expect as one and only arg the path to yolo.weights !")
    else:
        main(sys.argv[1])

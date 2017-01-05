import argparse
import time
import os

import tensorflow as tf
import numpy as np

from image_processing import postprocess, preprocess
from load import load_graph
from utils.read_config import read_cfg


def test(cfg_file, graph_filename, img_path):
    """
    Run a forward pass on an image
    """
    ann_path, val_ann_path, ckpt_path, labels, exclusive, batch, epoch, lr, \
        save_iter, meta = read_cfg(cfg_file)

    this_inp = preprocess(img_path, meta['inp_size'])
    expanded = np.expand_dims(this_inp, 0)
    graph = load_graph(graph_filename)

    # We access the input and output nodes
    inp = graph.get_tensor_by_name(u'prefix/input:0')
    keep_prob = graph.get_tensor_by_name(u'prefix/input_1:0')  # FIXME ?

    net_out = graph.get_tensor_by_name(u'prefix/29_fully_connected:0')

    with tf.Session(graph=graph) as sess:
        start = time.time()
        out = sess.run(net_out[0], {inp: expanded, keep_prob: 1})

        stop = time.time()

        print('Total time = {}s'.format(stop - start))

        postprocess(out, img_path, meta)


def parse_args():
    parser = argparse.ArgumentParser("Testing script for the localisation CNN")
    parser.add_argument("cfg", help="path of the .cfg config file")
    parser.add_argument("pb_file", help="path of the .pb weight file")
    parser.add_argument("img_path", help="path of a test input img")
    return parser.parse_args()

if __name__ == "__main__":
    # define the number of images to generate and run the script
    args = parse_args()
    test(args.cfg, args.pb_file, args.img_path)

import argparse
import time
import os

import tensorflow as tf
import numpy as np

from image_processing import postprocess, preprocess
from load import load_testing
from utils.read_config import read_cfg


def test(cfg_file, img_path):
    """
    Run a forward pass on an image
    """
    ann_path, ckpt_path, labels, exclusive, batch, epoch, lr, save_iter, \
        meta = read_cfg(cfg_file)

    this_inp = preprocess(img_path, meta['inp_size'])
    expanded = np.expand_dims(this_inp, 0)

    sess = tf.Session()

    net_out, [inp, keep_prob] = load_testing(sess, [448, 448, 3],
                                             ckpt_folder="saved")
    start = time.time()
    out = sess.run(net_out[0], {inp: expanded, keep_prob: 1})

    stop = time.time()

    print('Total time = {}s'.format(stop - start))

    postprocess(out, img_path, meta)


def parse_args():
    parser = argparse.ArgumentParser("Testing script for the localisation CNN")
    parser.add_argument("cfg", help="path of the .cfg config file")
    parser.add_argument("img_path", help="path of a test input img")
    return parser.parse_args()

if __name__ == "__main__":
    # define the number of images to generate and run the script
    args = parse_args()
    test(args.cfg, args.img_path)

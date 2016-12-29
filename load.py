import tensorflow as tf
import sys
from loss import loss
from tiny_yolo import create_graph
import cPickle as pkl


def graph_construction(sess, meta, pkl_path=None):
    # create a graph
    net_out, all_vars, [inp, keep_prob] = create_graph(
        [None] + meta['inp_size'])
    placeholders, loss_layer = loss(net_out, meta)
    placeholders['input'] = inp
    placeholders['keep_prob'] = keep_prob

    sess.run(tf.global_variables_initializer())

    # weight transfer_learning
    if pkl_path:
        print "let's transfer the weights from ", pkl_path
        weights = []
        with open(pkl_path, 'rf') as f:
            weights = pkl.load(f)
        for weight, i in enumerate(weights):
            if i != len(weights)-1:
                all_vars[i].assign(weight).eval(session=sess)

    return placeholders, loss_layer, net_out

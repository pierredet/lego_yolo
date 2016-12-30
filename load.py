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
    
    # construct the training loop
    optimizer = tf.train.RMSPropOptimizer(meta['lr'])
    gradients = optimizer.compute_gradients(loss_layer)
    train_op = optimizer.apply_gradients(gradients)

    sess.run(tf.global_variables_initializer())

    # weight transfer_learning
    if pkl_path:
        print "let's transfer the weights from ", pkl_path
        weights = []
        with open(pkl_path, 'rf') as f:
            weights = pkl.load(f)
        for i, weight in enumerate(weights):
            if i < len(weights)-2:
                all_vars[i].assign(weight).eval(session=sess)

    return placeholders, loss_layer, train_op

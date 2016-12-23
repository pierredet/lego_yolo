import tensorflow as tf
import sys
from loss import loss


def lrelu(x, leak=0.1, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def add_fully_connected_layer(net_out, meta):
    nb_neurons_in = int(net_out.get_shape()[1])
    nb_neurons_out = 735  # TODO not sure how to compute this from meta
    # TODO see loss.py
    
    W = tf.Variable(tf.truncated_normal([nb_neurons_in, nb_neurons_out],
                                       stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[nb_neurons_out]))
    out = lrelu(tf.nn.xw_plus_b(net_out, W, b, name="transfer_learning"))

    # dropout
    keep_prob = tf.placeholder(tf.float32)
    out_drop = tf.nn.dropout(out, keep_prob)
    return out_drop, W, b


def graph_construction(sess, ckpt_path, meta):
    new_saver = tf.train.import_meta_graph(ckpt_path)
    new_saver.restore(sess, tf.train.latest_checkpoint('./darkflow/ckpt'))
    net_out = tf.get_collection('net_out')[0]  # fetch the previous fc

    net_out, W, b = add_fully_connected_layer(net_out, meta)

    # all_vars = tf.global_variables()
    placeholders, loss_layer = loss(net_out, meta)
    sess.run(tf.variables_initializer([W,b], name='init'))
    return placeholders, loss_layer

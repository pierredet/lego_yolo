import tensorflow as tf
import sys
from loss import loss


def graph_construction(sess, ckpt_path, meta):
    new_saver = tf.train.import_meta_graph(ckpt_path)
    new_saver.restore(sess, tf.train.latest_checkpoint('./darkflow/ckpt'))
    net_out = tf.get_collection('net_out')[0]

    # all_vars = tf.global_variables()
    placeholders, loss_layer = loss(net_out, meta)


    return placeholders, loss_layer

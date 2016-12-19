import tensorflow as tf
import sys
from loss import loss


def graph_construction(sess, ckpt_path, meta):
    new_saver = tf.train.import_meta_graph(ckpt_path)
    new_saver.restore(sess, tf.train.latest_checkpoint('./darkflow/ckpt'))
    # truc = tf.get_collection('loss')[0]
    import pdb; pdb.set_trace()
    all_vars = tf.global_variables()
    placeholders, loss_layer = loss(all_vars[-1], meta)


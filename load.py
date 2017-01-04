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


def load_training(sess, meta, ckpt_folder="saved"):
    if len(os.listdir(ckpt_folder)) > 0:

        new_saver = tf.train.import_meta_graph(
            tf.train.latest_checkpoint(ckpt_folder) + ".meta")

        # taking trace of first index
        first_step = int(
            tf.train.latest_checkpoint(ckpt_folder).split('-')[-1])
        print("load from checkpoint {:d}".format(first_step))
        # recreate tf graph
        placeholders, loss_op, train_op = graph_construction(sess, meta,
                                                             pkl_path=None)
        # restore variable values
        new_saver.restore(sess, tf.train.latest_checkpoint(ckpt_folder))
    else:
        first_step = 0
        # load values from pkl for transfer learning
        placeholders, loss_op, train_op = graph_construction(
            sess, meta, pkl_path="weight.pkl")
    return placeholders, loss_op, train_op, first_step


def load_testing(sess, inp_size, ckpt_folder="saved"):
    """
    Create and load a graph without training ops
    """
    # create a graph
    net_out, all_vars, [inp, keep_prob] = create_graph(
        [None] + inp_size)
    sess.run(tf.global_variables_initializer())

    new_saver = tf.train.import_meta_graph(
        tf.train.latest_checkpoint(ckpt_folder) + ".meta")

    # taking trace of first index
    step = int(
        tf.train.latest_checkpoint(ckpt_folder).split('-')[-1])
    print("load from checkpoint {:d}".format(step))

    # restore variable values
    new_saver.restore(sess, tf.train.latest_checkpoint(ckpt_folder))
    return net_out, [inp, keep_prob]

import sys
import os
import cPickle as pkl

import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.contrib.layers.python.layers.optimizers import optimize_loss

from loss import loss
from tiny_yolo import create_graph


def graph_construction(sess, meta, pkl_path=None):
    # create a graph
    net_out, all_vars, [inp, keep_prob] = create_graph(
        [None] + meta['inp_size'])
    placeholders, loss_layer = loss(net_out, meta)
    placeholders['input'] = inp
    placeholders['keep_prob'] = keep_prob

    batch = tf.Variable(0)

    learning_rate = tf.train.exponential_decay(
        meta['lr'],                # Base learning rate.
        batch * meta['batch'],  # Current index into the dataset.
        30000,          # Decay step.
        0.95,                # Decay rate.
        staircase=True)
    # # add a summary
    tf.summary.scalar('learning_rate', learning_rate)
    #
    # # construct the training loop
    # optimizer = tf.train.RMSPropOptimizer(learning_rate)
    # gradients = optimizer.compute_gradients(loss_layer)
    # train_op = optimizer.apply_gradients(gradients, global_step=batch)

    train_op = optimize_loss(loss_layer,
                             batch,
                             learning_rate,
                             "RMSProp")

    sess.run(tf.global_variables_initializer())

    # weight transfer_learning
    if pkl_path:
        print "let's transfer the weights from ", pkl_path
        weights = []
        with open(pkl_path, 'rf') as f:
            weights = pkl.load(f)
        for i, weight in enumerate(weights):
            if i < len(weights) - 2:
                all_vars[i].assign(weight).eval(session=sess)

    return placeholders, loss_layer, train_op


def load_training(sess, meta, ckpt_folder="saved"):
    # if len(os.listdir(ckpt_folder)) > 0:
    #
    #     # recreate tf graph
    #     placeholders, loss_op, train_op = graph_construction(
    #         sess, meta, pkl_path=None)
    #
    #     new_saver = tf.train.import_meta_graph(
    #         tf.train.latest_checkpoint(ckpt_folder) + ".meta")
    #
    #     # taking trace of first index
    #     first_step = int(
    #         tf.train.latest_checkpoint(ckpt_folder).split('-')[-1])
    #     print("load from checkpoint {:d}".format(first_step))
    #     # restore variable values
    #     new_saver.restore(sess, tf.train.latest_checkpoint(ckpt_folder))
    # else:
    first_step = 0
    # load values from pkl for transfer learning
    placeholders, loss_op, train_op = graph_construction(
        sess, meta, pkl_path="weight.pkl")
    return placeholders, loss_op, train_op, first_step


def freeze_graph(model_folder):
    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_folder)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    absolute_model_folder = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_folder + "/frozen_model.pb"

    # Before exporting our graph, we need to precise what is our output node
    # This is how TF decides what part of the Graph he has to keep and what part it can dump
    # NOTE: this variables is plural, because you can have multiple output
    # nodes
    output_node_names = "29_fully_connected"

    # We clear the devices, to allow TensorFlow to control on the loading
    # where it wants operations to be calculated
    clear_devices = True

    # We import the meta graph and retrive a Saver
    saver = tf.train.import_meta_graph(
        input_checkpoint + '.meta', clear_devices=clear_devices)

    # We retrieve the protobuf graph definition
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    # We start a session and restore the graph weights
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constant
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            input_graph_def,  # The graph_def is used to retrieve the nodes
            # The output node names are used to select the usefull nodes
            output_node_names.split(",")
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))


def load_graph(frozen_graph_filename):
    # First we need to load the protobuf file from the disk and parse it to
    # retrieve the Unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a
    # graph_def into the current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph

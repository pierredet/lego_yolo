from batch import shuffle, batch
from loss import loss
from load import load_training

from utils.pascal_voc_clean_xml import parse_to_pkl
from utils.read_config import read_cfg

import cPickle as pkl
import sys
import os
import numpy as np

import tensorflow as tf


def launch_training(cfg_file):
    ann_path, val_ann_path, ckpt_path, labels, exclusive, batch, epoch, lr,\
        save_iter, meta = read_cfg(cfg_file)

    # First check if there is a corresponding annotation
    # parse to your config
    ann_parsed = os.path.join(ann_path, cfg_file.split('.')[0] + '.pkl')
    if os.path.exists(ann_parsed):
        f = open(ann_parsed, 'rb')
        data = pkl.load(f)[0]
    else:
        data = parse_to_pkl(labels, ann_parsed, ann_path, exclusive=exclusive)

    # Similar thing for validation loss
    val_ann_parsed = os.path.join(val_ann_path,
                                  cfg_file.split('.')[0] + '.pkl')
    if os.path.exists(val_ann_parsed):
        f = open(val_ann_parsed, 'rb')
        val = pkl.load(f)[0]
    else:
            val = parse_to_pkl(labels, val_ann_parsed, val_ann_path,
                               exclusive=exclusive)

    sess = tf.Session()

    placeholders, loss_op, train_op, first_step = load_training(
        sess, meta, ckpt_folder="saved")
    saver = tf.train.Saver()
    merged = tf.summary.merge_all()
    train_writer = tf.train.SummaryWriter("out" + '/train',
                                          sess.graph)
    test_writer = tf.train.SummaryWriter("out" + '/test',
                                         sess.graph)
    # actual training loop
    batches = shuffle(data, batch, epoch, meta, ann_path)
    val_batches = shuffle(val, batch, epoch, meta, val_ann_path)
    loss_mva = None
    total = int()  # total number of batches
    for i, packet in enumerate(batches):
        if i == 0:
            total = packet
            total_val = next(val_batches)
            args = [lr, batch]
            args += [epoch, save_iter]
            print "training params", args
            continue

        x_batch, datum = packet
        datum['input'] = x_batch
        datum['keep_prob'] = 0.5
        if i == 1:
            assert set(list(datum)) == set(list(placeholders)), \
                'Feed and placeholders of loss op mismatched'
        feed_pair = [(placeholders[k], datum[k]) for k in datum]
        feed_dict = {holder: val for (holder, val) in feed_pair}

        summary, _, loss = sess.run([merged, train_op, loss_op], feed_dict)

        train_writer.add_summary(summary, i)

        if loss_mva is None:
            loss_mva = loss
        loss_mva = .9 * loss_mva + .1 * loss
        # counter from now on
        step_now = first_step + i
        args = [step_now, loss, loss_mva]
        print 'step {} - loss {} - moving ave loss {}'.format(*args)
        if i % (save_iter / batch) == 0 or i == total:
            ckpt = os.path.join("saved", '{}-{}'.format(
                ann_path.split('/')[-1], step_now))
            print 'Checkpoint at step {}'.format(step_now)
            saver.save(sess, ckpt)

        if i % 100 == 0:
            # validation step
            x_batch, datum = next(val_batches)
            datum['input'] = x_batch
            datum['keep_prob'] = 1
            if i/100 == 0:
                assert set(list(datum)) == set(list(placeholders)), \
                    'Feed and placeholders of loss op mismatched'
            feed_pair = [(placeholders[k], datum[k]) for k in datum]
            feed_dict = {holder: val for (holder, val) in feed_pair}

            summary, loss = sess.run([merged, loss_op], feed_dict)

            test_writer.add_summary(summary, i)


if __name__ == '__main__':
    launch_training(sys.argv[1])

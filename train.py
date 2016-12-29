from batch import shuffle, batch
from loss import loss
from load import graph_construction

from utils.pascal_voc_clean_xml import parse_to_pkl

import cPickle as pkl
import sys
import os
import ConfigParser
import numpy as np

import tensorflow as tf


def launch_training(cfg_file):
    cfg = ConfigParser.ConfigParser()
    cfg.read(cfg_file)

    ann_path = cfg.get('general', 'ann_path')
    ckpt_path = cfg.get('general', 'ckpt_path')
    labels = cfg.get('general', 'labels').split()
    exclusive = cfg.getboolean('general', 'exclusive')

    batch = cfg.getint('net', 'batch')
    epoch = cfg.getint('general', 'epoch')
    lr = cfg.getfloat('general', 'learning_rate')
    save_iter = cfg.getint('general', 'save_iter')

    # load detection metaparameter from the config file
    meta = {}
    for key, value in cfg.items('detection'):
        if '.' in value:
            meta[key] = float(value)
        else:
            meta[key] = int(value)
    meta['inp_size'] = [cfg.getint('net', 'height'), cfg.getint('net', 'width'),
                        cfg.getint('net', 'channels')]
    meta['labels'] = labels
    meta['lr'] = lr
    meta['model'] = ann_path.split('/')[-1]
    meta['ann_path'] = ann_path

    # First check if there is a corresponding annotation
    # parse to your config
    ann_parsed = os.path.join(ann_path, cfg_file.split('.')[0] + '.pkl')
    if os.path.exists(ann_parsed):
        f = open(ann_parsed, 'rb')
        data = pkl.load(f)[0]
    else:
        data = parse_to_pkl(labels, ann_parsed, ann_path, exclusive=exclusive)

    sess = tf.Session()
    placeholders, loss, net_out = graph_construction(sess, meta,
                                                     pkl_path="weight.pkl")

    # build train_op
    optimizer = tf.train.RMSPropOptimizer(meta['lr'])
    gradients = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(gradients)

    # actual training loop
    batches = shuffle(data, batch, epoch, meta)
    loss_mva = None
    total = int()  # total number of batches
    for i, packet in enumerate(batches):
        if i == 0:
            total = packet
            args = [lr, batch]
            args += [epoch, save_iter]
            print "training params", args
            continue

        x_batch, datum = packet
        datum['input'] = x_batch
        import pdb; pdb.set_trace()
        sess.run([net_out],{placeholder['input']: x_batch})
        if i == 1:
            assert set(list(datum)) == set(list(placeholders)), \
                'Feed and placeholders of loss op mismatched'
        feed_pair = [(placeholders[k], datum[k]) for k in datum]
        feed_dict = {holder: val for (holder, val) in feed_pair}
        # for k in self.feed:
        #     feed_dict[k] = self.feed[k]
        import pdb; pdb.set_trace()

        _, loss, summary = sess.run([train_op, loss], feed_dict)

        train_writer.add_summary(summary, i)

        loss = fetched[1]
        # for f in fetched[2:]:
        #     print np.sum(f)
        # assert 0
        if loss_mva is None:
            loss_mva = loss
        loss_mva = .9 * loss_mva + .1 * loss
        # counter from now on
        step_now = self.FLAGS.load + i
        args = [step_now, loss, loss_mva]
        print 'step {} - loss {} - moving ave loss {}'.format(*args)
        if i % (save_iter / batch) == 0 or i == total:
            ckpt = os.path.join(self.FLAGS.backup, '{}-{}'.format(
                model, step_now))
            print 'Checkpoint at step {}'.format(step_now)
            self.saver.save(sess, ckpt)


if __name__ == '__main__':
    launch_training(sys.argv[1])

from batch import shuffle, bat:wqch
from loss import loss
from load import load

from utils.pascal_voc_clean_xml import parse_to_pkl

import cPickle as pkl
import sys
import ConfigParser


def launch_training(cfg_file):
    cfg = ConfigParser.ConfigParser()
    cfg.read(cfg_file)
    
    ann_path = cfg.get('ann_path')
    labels = cfg.get('labels').split()
    exclusive = cfg.getboolean('exclusive')
    
    batch = cfg.getint('batch')
    epoch = cfg.getint('epoch')
    lr = cfg.getfloat('learning_rate')
    save_iter = cfg.getint('save_iter')
    
    # First check if there is a corresponding annotation
    # parse to your config
    ann_parsed = os.path.join(ann, cfg_file.split('.')[0] + '.pkl')
    if os.exists(ann_parsed):
        f = open(ann_parsed, 'rb')
        data = pkl.load(f)
    else:
        data = parse_to_pkl(labels, ann_parsed, ann_path, exclusive=exclusive)
    
    # TODO load from checkpoint here
    # define train_op loss sess
    
    # actual training loop
    batches = shuffle(data, batch, epoch, meta)
    loss_mva = None; total = int() # total number of batches
    for i, packet in enumerate(batches):
	if i == 0: 
            total = packet;
	    args = [lr, batch]
	    args += [epoch, save]
	    print "training params", args	
	    continue

       x_batch, datum = packet

       if i == 1: \
           assert set(list(datum)) == set(list(self.placeholders)), \
           'Feed and placeholders of loss op mismatched'

       feed_pair = [(self.placeholders[k], datum[k]) for k in datum]
       feed_dict = {holder:val for (holder,val) in feed_pair}
       for k in self.feed:
           feed_dict[k] = self.feed[k]
       feed_dict[self.inp] = x_batch

		fetches = [train_op, loss] 
		fetches += self.framework.fetch #todo wtf ?
		fetched = sess.run(fetches, feed_dict)
		loss = fetched[1]
		# for f in fetched[2:]:
		# 	print np.sum(f)
		# assert 0
		if loss_mva is None:
                    loss_mva = loss
		loss_mva = .9 * loss_mva + .1 * loss
		# counter from now on
                step_now = self.FLAGS.load + i
		args = [step_now, loss, loss_mva]
		print 'step {} - loss {} - moving ave loss {}'.format(*args)
		if i % (save_iter/batch) == 0 or i == total:
			ckpt = os.path.join(self.FLAGS.backup, '{}-{}'.format(model, step_now))
			print 'Checkpoint at step {}'.format(step_now)
			self.saver.save(sess, ckpt)
            
       

if __name__ = '__main__':


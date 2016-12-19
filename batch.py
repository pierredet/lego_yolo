import numpy as np
from image_processing import preprocess


def shuffle(data, batch, epoch, meta):
    """
    data, parsed annotations (stored in a pkl)
    Call the specific framework to parse annotations, then use the parsed
    object to yield minibatches. minibatches should be preprocessed before
    yielding to be appropriate placeholders for model's loss evaluation.
    """
    size = len(data)

    print 'Dataset of {} instance(s)'.format(size)
    if batch > size:
        print "batch bigger than size !"
    batch_per_epoch = int(size / batch)
    total = epoch * batch_per_epoch
    yield total

    for i in range(epoch):
        print 'EPOCH {}'.format(i + 1)
        shuffle_idx = perm(np.arange(size))
        for b in range(batch_per_epoch):
            end_idx = (b + 1) * batch
            start_idx = b * batch
            # two yieldee
            x_batch = list()
            feed_batch = dict()

            for j in range(start_idx, end_idx):
                real_idx = shuffle_idx[j]
                chunk = data[real_idx]
                inp, feedval = batch(chunk, meta)
                if inp is None:
                    continue

                x_batch += [np.expand_dims(inp, 0)]
                for key in feedval:
                    if key not in feed_batch:
                        feed_batch[key] = [feedval[key]]
                        continue
                    feed_batch[key] = np.concatenate(
                        [feed_batch[key], [feedval[key]]])

            x_batch = np.concatenate(x_batch, 0)
            yield (x_batch, feed_batch)


def batch(chunk, meta):
    """
    Takes a chunk of parsed annotations
    and a dictionnary of the data-set specific meta parameters
    returns value for placeholders of net's
    input & loss layer correspond to this chunk
    """
    # meta = self.meta
    S, B = meta['side'], meta['num']
    C, labels = meta['classes'], meta['labels']

    # preprocess
    jpg = chunk[0]
    w, h, allobj_ = chunk[1]
    allobj = deepcopy(allobj_)
    path = os.path.join(meta['ann_path'], jpg)
    img = preprocess(path, meta, allobj)

    # Calculate regression target
    cellx = 1. * w / S
    celly = 1. * h / S
    for obj in allobj:
        centerx = .5 * (obj[1] + obj[3])  # xmin, xmax
        centery = .5 * (obj[2] + obj[4])  # ymin, ymax
        cx = centerx / cellx
        cy = centery / celly
        if cx >= S or cy >= S:
            return None, None
        obj[3] = float(obj[3] - obj[1]) / w
        obj[4] = float(obj[4] - obj[2]) / h
        obj[3] = np.sqrt(obj[3])
        obj[4] = np.sqrt(obj[4])
        obj[1] = cx - np.floor(cx)  # centerx
        obj[2] = cy - np.floor(cy)  # centery
        obj += [int(np.floor(cy) * S + np.floor(cx))]

    # show(im, allobj, S, w, h, cellx, celly) # unit test

    # Calculate placeholders' values
    probs = np.zeros([S * S, C])
    confs = np.zeros([S * S, B])
    coord = np.zeros([S * S, B, 4])
    proid = np.zeros([S * S, C])
    conid = np.ones([S * S, B])
    cooid = np.zeros([S * S, B, 4])
    prear = np.zeros([S * S, 4])
    for obj in allobj:
        probs[obj[5], :] = [0.] * C
        probs[obj[5], labels.index(obj[0])] = 1.
        proid[obj[5], :] = [1] * C
        coord[obj[5], :, :] = [obj[1:5]] * B
        prear[obj[5], 0] = obj[1] - obj[3]**2 * .5 * S  # xleft
        prear[obj[5], 1] = obj[2] - obj[4]**2 * .5 * S  # yup
        prear[obj[5], 2] = obj[1] + obj[3]**2 * .5 * S  # xright
        prear[obj[5], 3] = obj[2] + obj[4]**2 * .5 * S  # ybot
        confs[obj[5], :] = [1.] * B
        # conid[obj[5], :] = [1.] * B
        cooid[obj[5], :, :] = [[1.] * 4] * B

    # Finalise the placeholders' values
    upleft = np.expand_dims(prear[:, 0:2], 1)
    botright = np.expand_dims(prear[:, 2:4], 1)
    wh = botright - upleft
    area = wh[:, :, 0] * wh[:, :, 1]
    upleft = np.concatenate([upleft] * B, 1)
    botright = np.concatenate([botright] * B, 1)
    areas = np.concatenate([area] * B, 1)

    # value for placeholder at input layer
    inp_feed_val = img
    # value for placeholder at loss layer
    loss_feed_val = {
        'probs': probs, 'confs': confs, 'coord': coord,
        'proid': proid, 'conid': conid, 'cooid': cooid,
        'areas': areas, 'upleft': upleft, 'botright': botright
    }

    return inp_feed_val, loss_feed_val

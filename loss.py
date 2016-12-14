"""
file: /yolo/train.py
includes: parse(), batch(), and loss()
together they support the pipeline:
    annotation -> minibatch -> loss evaluation -> training
namely,
loss() basically build the loss layer of the net, namely,
            returns the corresponding placeholders for feed values of this loss layer
            as well as loss & train_op built from these placeholders and net.out
"""
import tensorflow.contrib.slim as slim
import cPickle as pickle
import tensorflow as tf
import numpy as np
import os

from copy import deepcopy


def loss(net_out, m):
    """
    Takes net.out and placeholders value
    returned in batch() func above,
    to build train_op and loss
    """
    # meta
    sprob = m['class_scale']
    sconf = m['object_scale']
    snoob = m['noobject_scale']
    scoor = m['coord_scale']
    S, B, C = m['side'], m['num'], m['classes']
    SS = S * S # number of grid cells

    print '{} loss hyper-parameters:'.format(m['model'])
    print '\tside    = {}'.format(m['side'])
    print '\tbox     = {}'.format(m['num'])
    print '\tclasses = {}'.format(m['classes'])
    print '\tscales  = {}'.format([sprob, sconf, snoob, scoor])

    size1 = [None, SS, C]
    size2 = [None, SS, B]

    # return the below placeholders
    _probs = tf.placeholder(tf.float32, size1)
    _confs = tf.placeholder(tf.float32, size2)
    _coord = tf.placeholder(tf.float32, size2 + [4])
    # weights term for L2 loss
    _proid = tf.placeholder(tf.float32, size1)
    _conid = tf.placeholder(tf.float32, size2)
    _cooid = tf.placeholder(tf.float32, size2 + [4])
    # material for loss calculation
    _areas = tf.placeholder(tf.float32, size2)
    _upleft = tf.placeholder(tf.float32, size2 + [2])
    _botright = tf.placeholder(tf.float32, size2 + [2])

    placeholders = {
        'probs':_probs, 'confs':_confs, 'coord':_coord,
        'proid':_proid, 'conid':_conid, 'cooid':_cooid,
        'areas':_areas, 'upleft':_upleft, 'botright':_botright
    }

    # Extract the coordinate prediction from net.out
    coords = net_out[:, SS * (C + B):]
    coords = tf.reshape(coords, [-1, SS, B, 4])
    wh = tf.pow(coords[:,:,:,2:4], 2) * S # unit: grid cell
    area_pred = wh[:,:,:,0] * wh[:,:,:,1] # unit: grid cell^2
    centers = coords[:,:,:,0:2] # [batch, SS, B, 2]
    floor = centers - (wh * .5) # [batch, SS, B, 2]
    ceil  = centers + (wh * .5) # [batch, SS, B, 2]

    # calculate the intersection areas
    intersect_upleft   = tf.maximum(floor, _upleft)
    intersect_botright = tf.minimum(ceil , _botright)
    intersect_wh = intersect_botright - intersect_upleft
    intersect_wh = tf.maximum(intersect_wh, 0.0)
    intersect = tf.mul(intersect_wh[:,:,:,0], intersect_wh[:,:,:,1])

    # calculate the best IOU, set 0.0 confidence for worse boxes
    iou = tf.div(intersect, _areas + area_pred - intersect)
    best_box = tf.equal(iou, tf.reduce_max(iou, [2], True))
    best_box = tf.to_float(best_box)
    confs = tf.mul(best_box, _confs)

    # take care of the weight terms
    weight_con = snoob * (1. - confs) + sconf * confs
    conid = tf.mul(_conid, weight_con)
    weight_coo = tf.concat(3, 4 * [tf.expand_dims(confs, -1)])
    cooid = tf.mul(_cooid, scoor * weight_coo)
    proid = sprob * _proid

    # flatten 'em all
    probs = slim.flatten(_probs)
    proid = slim.flatten(proid)
    confs = slim.flatten(confs)
    conid = slim.flatten(conid)
    coord = slim.flatten(_coord)
    cooid = slim.flatten(cooid)
    true = tf.concat(1, [probs, confs, coord])
    wght = tf.concat(1, [proid, conid, cooid])

    print 'Building {} loss'.format(m['model'])
    loss = tf.pow(net_out - true, 2)
    loss = tf.mul(loss, wght)
    loss = tf.reduce_sum(loss, 1)
    loss = .5 * tf.reduce_mean(loss)
    
    # make the loss retrievable
    tf.add_to_collection("loss", loss)
    
    # adding a summary for Tensorboard
    tf.summary.scalar('training_loss', loss)
    tf.summary.scalar('confs_mean', tf.reduce_mean(confs))
    return placeholders, loss


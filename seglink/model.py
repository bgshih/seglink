import math
import tensorflow as tf
import numpy as np
import logging

import ops
import utils
import data
import model_cnn

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('pos_scale_diff_threshold', 1.7, '')
tf.app.flags.DEFINE_float('neg_scale_diff_threshold', 2.0, '')

# constants
OFFSET_DIM = 6

N_LOCAL_LINKS = 8
N_CROSS_LINKS = 4
N_SEG_CLASSES = 2
N_LNK_CLASSES = 2

MATCH_STATUS_POS = 1
MATCH_STATUS_NEG = -1
MATCH_STATUS_IGNORE = 0
POS_LABEL = 1
NEG_LABEL = 0

N_DET_LAYERS = 6


class SegLinkDetector():
  def __init__(self):
    self.vgg16 = model_cnn.SsdVgg16()

    # anchor box sizes for all feature layers
    self.anchor_sizes = [11.84210526, 23.68421053, 45., 90., 150., 285.]
    logging.info('Anchor sizes: {}'.format(self.anchor_sizes))

  def _detection_classifier(self, maps, ksize, cross_links=False, scope=None):
    """
    Create a SegLink detection classifier on a feature layer
    """
    with tf.variable_scope(scope):
      seg_depth = N_SEG_CLASSES
      if cross_links:
        lnk_depth = N_LNK_CLASSES * (N_LOCAL_LINKS + N_CROSS_LINKS)
      else:
        lnk_depth = N_LNK_CLASSES * N_LOCAL_LINKS
      reg_depth = OFFSET_DIM
      map_depth = maps.get_shape()[3].value
      seg_maps = ops.conv2d(maps, map_depth, seg_depth, ksize, 1, 'SAME', scope='conv_cls')
      lnk_maps = ops.conv2d(maps, map_depth, lnk_depth, ksize, 1, 'SAME', scope='conv_lnk')
      reg_maps = ops.conv2d(maps, map_depth, reg_depth, ksize, 1, 'SAME', scope='conv_reg')
    return seg_maps, lnk_maps, reg_maps

  def _build_cnn(self, images):
    vgg16_outputs = self.vgg16.build_model(images, scope='vgg16')

    # conv4_3, output maps: 64x64x512 (given 512x512 input)
    conv4_3 = vgg16_outputs['conv4_3']
    # l2-normalize and scale along the depth
    conv4_3_scaled = tf.nn.l2_normalize(conv4_3, 3, name='conv4_3_normed')
    conv4_3_scale = tf.get_variable('conv4_3_scale',
                                    shape=[512],
                                    initializer=tf.constant_initializer(value=20.0, dtype=tf.float32),
                                    trainable=True)
    tf.add_to_collection('biases', conv4_3_scale) # do not apply weight decay
    conv4_3_scaled = tf.multiply(conv4_3_scaled,
                                 tf.reshape(conv4_3_scale, [1,1,1,512],
                                 name='conv4_3_norm_scaled'))

    # fc7, 32x32x512
    fc7 = vgg16_outputs['fc7']

    # conv6, 16x16x512
    conv8_1 = ops.conv_relu(fc7, 1024, 256, 1, 1, scope='conv8_1')
    conv8_2 = ops.conv_relu(conv8_1, 256, 512, 3, 2, scope='conv8_2')

    # conv7, 8x8x512
    conv9_1 = ops.conv_relu(conv8_2, 512, 128, 1, 1, scope='conv9_1')
    conv9_2 = ops.conv_relu(conv9_1, 128, 256, 3, 2, scope='conv9_2')

    # conv8, 4x4x512
    conv10_1 = ops.conv_relu(conv9_2, 256, 128, 1, 1, scope='conv10_1')
    conv10_2 = ops.conv_relu(conv10_1, 128, 256, 3, 2, scope='conv10_2')

    # conv11, 2x2x256
    conv11 = ops.conv_relu(conv10_2, 256, 256, 3, 2, scope='conv11')

    outputs = {
      'conv4_3': conv4_3_scaled,
      'fc7': fc7,
      'conv8_2': conv8_2,
      'conv9_2': conv9_2,
      'conv10_2': conv10_2,
      'conv11': conv11,
    }
    return outputs


  def build_model(self, images, scope=None):
    """
    Construct the main body of a SegLink model
    """
    # NOTE: this scope must be set properly to successfully load the pretrained model
    with tf.variable_scope(scope or 'ssd'):
      cnn_outputs = self._build_cnn(images)

      det_1 = self._detection_classifier(cnn_outputs['conv4_3'], 3,
                                         cross_links=False, scope='det_1')
      det_2 = self._detection_classifier(cnn_outputs['fc7'], 3,
                                         cross_links=True, scope='det_2')
      det_3 = self._detection_classifier(cnn_outputs['conv8_2'], 3,
                                         cross_links=True, scope='det_3')
      det_4 = self._detection_classifier(cnn_outputs['conv9_2'], 3,
                                         cross_links=True, scope='det_4')
      det_5 = self._detection_classifier(cnn_outputs['conv10_2'], 3,
                                         cross_links=True, scope='det_5')
      det_6 = self._detection_classifier(cnn_outputs['conv11'], 3,
                                         cross_links=True, scope='det_6')
      outputs = [det_1, det_2, det_3, det_4, det_5, det_6]
      return outputs

  def _cls_mining(self, scores, status, hard_neg_ratio=3.0, scope=None):
    """
    Positive classification loss and hard negative classificatin loss
    ARGS
      scores: [n, n_classes]
      status: int [n] node or link matching status
    RETURNS
      pos_loss: []
      n_pos: int []
      hard_neg_loss: []
      n_hard_neg: []
    """
    with tf.variable_scope(scope or 'cls_mining'):
      # positive classification loss
      pos_mask = tf.equal(status, MATCH_STATUS_POS)
      pos_scores = tf.boolean_mask(scores, pos_mask)
      n_pos = tf.shape(pos_scores)[0]
      pos_labels = tf.fill([n_pos], POS_LABEL)
      pos_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=pos_scores, labels=pos_labels))

      # hard negative classification loss
      neg_mask = tf.equal(status, MATCH_STATUS_NEG)
      neg_scores = tf.boolean_mask(scores, neg_mask)
      n_neg = tf.shape(neg_scores)[0]
      n_hard_neg = tf.cast(n_pos, tf.float32) * hard_neg_ratio
      n_hard_neg = tf.minimum(n_hard_neg, tf.cast(n_neg, tf.float32))
      n_hard_neg = tf.cast(n_hard_neg, tf.int32)
      neg_prob = tf.nn.softmax(neg_scores)[:, NEG_LABEL]
      # find the k examples with the least negative probabilities
      _, hard_neg_indices = tf.nn.top_k(-neg_prob, k=n_hard_neg)
      hard_neg_scores = tf.gather(neg_scores, hard_neg_indices)
      hard_neg_labels = tf.fill([n_hard_neg], NEG_LABEL)
      hard_neg_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=hard_neg_scores, labels=hard_neg_labels))

      return pos_loss, n_pos, hard_neg_loss, n_hard_neg

  def build_loss(self, all_maps, gt_rboxes, gt_counts, image_size, scope=None):
    with tf.variable_scope(scope or 'loss'):
      # status is different from labels
      node_status_all = []
      link_status_all = []
      gt_offsets_all = []

      # encode local groundtruth
      node_status_below = tf.constant([[[0]]], dtype=tf.int32)
      match_indices_below = tf.constant([[[0]]], dtype=tf.int32)
      for i in range(N_DET_LAYERS):
        det_layer_name = 'det_{}'.format(i+1)
        map_size = tf.shape(all_maps[i][0])[1:]
        node_status, link_status, gt_offsets, match_indices = \
          ops.encode_groundtruth(gt_rboxes,
                                 gt_counts,
                                 map_size,
                                 image_size,
                                 node_status_below,
                                 match_indices_below,
                                 anchor_size=self.anchor_sizes[i],
                                 pos_scale_diff_thresh=FLAGS.pos_scale_diff_threshold,
                                 neg_scale_diff_thresh=FLAGS.neg_scale_diff_threshold,
                                 cross_links=(i > 0))
        gt_offsets_scaled = gt_offsets / data.OFFSET_VARIANCE
        node_status_all.append(node_status)
        link_status_all.append(link_status)
        gt_offsets_all.append(gt_offsets_scaled)
        # for next iteration
        node_status_below = node_status
        match_indices_below = match_indices

      # node classification loss
      with tf.variable_scope('node_cls_loss'):
        node_status_flat = tf.concat([tf.reshape(o, [-1]) for o in node_status_all], axis=0)
        node_scores_flat = tf.concat([tf.reshape(o[0], [-1, N_SEG_CLASSES]) for o in all_maps], axis=0)
        node_pos_loss, n_pos_nodes, node_hardneg_loss, _ = \
          self._cls_mining(node_scores_flat,
                           node_status_flat,
                           hard_neg_ratio=FLAGS.hard_neg_ratio,
                           scope='node_cls_mining')
        node_normalizer = tf.maximum(1.0, tf.cast(n_pos_nodes, tf.float32))
        node_cls_loss = tf.truediv(node_pos_loss + node_hardneg_loss,
                                  node_normalizer, name='node_cls_loss')

      # link classification loss
      with tf.variable_scope('link_cls_loss'):
        link_status_flat = tf.concat([tf.reshape(o, [-1]) for o in link_status_all], axis=0)
        link_scores_flat = tf.concat([tf.reshape(o[1], [-1, N_LNK_CLASSES]) for o in all_maps], axis=0)
        link_pos_loss, n_pos_links, link_hardneg_loss, _ = \
          self._cls_mining(link_scores_flat, link_status_flat,
                           hard_neg_ratio=FLAGS.hard_neg_ratio,
                           scope='link_cls_mining')
        link_normalizer = tf.maximum(1.0, tf.cast(n_pos_links, tf.float32))
        link_cls_loss = tf.truediv(link_pos_loss + link_hardneg_loss,
                                   link_normalizer,
                                   name='link_cls_loss')

      # regression loss
      with tf.variable_scope('offset_loss'):
        gt_offsets_flat = tf.concat([tf.reshape(o, [-1, OFFSET_DIM]) for o in gt_offsets_all], axis=0)
        offset_pos_mask = tf.equal(node_status_flat, MATCH_STATUS_POS)
        gt_offsets_pos = tf.boolean_mask(gt_offsets_flat, offset_pos_mask)
        offsets_flat = tf.concat([tf.reshape(o[2], [-1, OFFSET_DIM]) for o in all_maps], axis=0)
        offsets_pos = tf.boolean_mask(offsets_flat, offset_pos_mask)
        offset_loss = tf.truediv(tf.reduce_sum(ops.smooth_l1_loss(offsets_pos, gt_offsets_pos)),
                                 node_normalizer)

      # detection loss (node classification + link classification + regression losses)
      detection_loss = tf.add_n([node_cls_loss, link_cls_loss, offset_loss],
                                name='detection_loss')

      # regularization loss
      with tf.variable_scope('regularization'):
        weight_l2_losses = [tf.nn.l2_loss(o) for o in tf.get_collection('weights')]
        weight_decay_loss = tf.multiply(tf.add_n(weight_l2_losses),
                                        FLAGS.weight_decay,
                                        name='weight_decay_loss')

      # total loss
      total_loss = tf.add_n([detection_loss, weight_decay_loss], name='total_loss')

      return total_loss

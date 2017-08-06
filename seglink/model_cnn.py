import sys, os
import tensorflow as tf
import numpy as np

import ops
import utils

FLAGS = tf.app.flags.FLAGS


class SsdVgg16():
  def __init__(self):
    self.outputs = {}

  def _vgg_conv_relu(self, x, n_in, n_out, scope, fc7=False, trainable=True):
    with tf.variable_scope(scope):
      if fc7 == False:
        conv = ops.conv2d(x, n_in, n_out, 3, trainable=trainable, relu=True)
      else:
        conv = ops.conv2d(x, n_in, n_out, 1, trainable=trainable, relu=True)
    return conv

  def _vgg_atrous_conv_relu(self, x, n_in, n_out, scope):
    with tf.variable_scope(scope):
      conv = ops.atrous_conv2d(x, n_in, n_out, 3, 6,
                               weight_init=FLAGS.weight_init_method, relu=True)
    return conv

  def _vgg_max_pool(self, x, scope, pool5=False):
    with tf.variable_scope(scope):
      if not pool5:
        pool = ops.max_pool(x, 2, 2, 'SAME')
      else:
        pool = ops.max_pool(x, 3, 1, 'SAME')
    return pool

  def build_model(self, images, scope=None):
    with tf.variable_scope(scope or 'vgg16'):
      # conv stage 1
      relu1_1 = self._vgg_conv_relu(images, 3, 64, 'conv1_1', trainable=False)
      relu1_2 = self._vgg_conv_relu(relu1_1, 64, 64, 'conv1_2', trainable=False)
      pool1 = self._vgg_max_pool(relu1_2, 'pool1')
      # conv stage 2
      relu2_1 = self._vgg_conv_relu(pool1, 64, 128, 'conv2_1', trainable=False)
      relu2_2 = self._vgg_conv_relu(relu2_1, 128, 128, 'conv2_2', trainable=False)
      pool2 = self._vgg_max_pool(relu2_2, 'pool2')
      # layers below pool2 are freezed
      pool2 = tf.stop_gradient(pool2)
      # conv stage 3
      relu3_1 = self._vgg_conv_relu(pool2, 128, 256, 'conv3_1')
      relu3_2 = self._vgg_conv_relu(relu3_1, 256, 256, 'conv3_2')
      relu3_3 = self._vgg_conv_relu(relu3_2, 256, 256, 'conv3_3')
      pool3 = self._vgg_max_pool(relu3_3, 'pool3')
      # conv stage 4
      relu4_1 = self._vgg_conv_relu(pool3, 256, 512, 'conv4_1')
      relu4_2 = self._vgg_conv_relu(relu4_1, 512, 512, 'conv4_2')
      relu4_3 = self._vgg_conv_relu(relu4_2, 512, 512, 'conv4_3') # => 38 x 38
      pool4 = self._vgg_max_pool(relu4_3, 'pool4')
      # conv stage 5
      relu5_1 = self._vgg_conv_relu(pool4, 512, 512, 'conv5_1')
      relu5_2 = self._vgg_conv_relu(relu5_1, 512, 512, 'conv5_2')
      relu5_3 = self._vgg_conv_relu(relu5_2, 512, 512, 'conv5_3')
      # pool5 has ksize 3 and stride 1
      pool5 = self._vgg_max_pool(relu5_3, 'pool5', pool5=True) # => 19 x 19
      # atrous_conv6 (fc6)
      relu_fc6 = self._vgg_atrous_conv_relu(pool5, 512, 1024, 'fc6') # => 19 x 19
      relu_fc7 = self._vgg_conv_relu(relu_fc6, 1024, 1024, 'fc7', fc7=True) # => 19 x 19

      outputs = {
        'conv4_3': relu4_3,
        'fc7': relu_fc7
      }
      return outputs

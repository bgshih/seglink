import os
import shutil
import uuid
import tensorflow as tf
import math

FLAGS = tf.app.flags.FLAGS

LIB_NAME = 'seglink'


def load_oplib(lib_name):
  """
  Load TensorFlow operator library.
  """
  # use absolute path so that ops.py can be called from other directory
  lib_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'lib{0}.so'.format(lib_name))
  # duplicate library with a random new name so that
  # a running program will not be interrupted when the original library is updated
  lib_copy_path = '/tmp/lib{0}_{1}.so'.format(str(uuid.uuid4())[:8], LIB_NAME)
  shutil.copyfile(lib_path, lib_copy_path)
  oplib = tf.load_op_library(lib_copy_path)
  return oplib


def _nn_variable(name, shape, init_method, collection=None, **kwargs):
  """
  Create or reuse a variable
  ARGS
    name: variable name
    shape: variable shape
    init_method: 'zero', 'kaiming', 'xavier', or (mean, std)
    collection: if not none, add variable to this collection
    kwargs: extra paramters passed to tf.get_variable
  RETURN
    var: a new or existing variable
  """
  if init_method == 'zero':
    initializer = tf.constant_initializer(0.0)
  elif init_method == 'kaiming':
    if len(shape) == 4: # convolutional filters
      kh, kw, n_in = shape[:3]
      init_std = math.sqrt(2.0 / (kh * kw * n_in))
    elif len(shape) == 2: # linear weights
      n_in, n_out = shape
      init_std = math.sqrt(1.0 / n_out)
    else:
      raise 'Unsupported shape'
    initializer = tf.truncated_normal_initializer(0.0, init_std)
  elif init_method == 'xavier':
    if len(shape) == 4:
      initializer = tf.contrib.layers.xavier_initializer_conv2d()
    else:
      initializer = tf.contrib.layers.xavier_initializer()
  elif isinstance(init_method, tuple):
    assert(len(init_method) == 2)
    initializer = tf.truncated_normal_initializer(init_method[0], init_method[1])
  else:
    raise 'Unsupported weight initialization method: ' + init_method

  var = tf.get_variable(name, shape=shape, initializer=initializer, **kwargs)
  if collection is not None:
    tf.add_to_collection(collection, var)

  return var


def conv2d(x, n_in, n_out, ksize, stride=1, padding='SAME',
           weight_init=None, bias=True, relu=False, scope=None,
           **kwargs):
  weight_init = weight_init or FLAGS.weight_init_method
  trainable = kwargs.get('trainable', True)
  with tf.variable_scope(scope or 'conv2d'):
    # convolution
    kernel = _nn_variable('weight', [ksize,ksize,n_in,n_out], weight_init,
                          collection='weights' if trainable else None,
                          **kwargs)
    y = tf.nn.conv2d(x, kernel, [1,stride,stride,1], padding=padding)
    # add bias
    if bias == True:
      bias = _nn_variable('bias', [n_out], 'zero',
                          collection='biases' if trainable else None,
                          **kwargs)
      y = tf.nn.bias_add(y, bias)
    # apply ReLU
    if relu == True:
      y = tf.nn.relu(y)
  return y


def conv_relu(*args, **kwargs):
  kwargs['relu'] = True
  if 'scope' not in kwargs:
    kwargs['scope'] = 'conv_relu'
  return conv2d(*args, **kwargs)


def atrous_conv2d(x, n_in, n_out, ksize, dilation, padding='SAME',
                  weight_init=None, bias=True,
                  relu=False, scope=None, **kwargs):
  weight_init = weight_init or FLAGS.weight_init_method
  trainable = kwargs.get('trainable', True)
  with tf.variable_scope(scope or 'atrous_conv2d'):
    # atrous convolution
    kernel = _nn_variable('weight', [ksize,ksize,n_in,n_out], weight_init,
                          collection='weights' if trainable else None,
                          **kwargs)
    y = tf.nn.atrous_conv2d(x, kernel, dilation, padding=padding)
    # add bias
    if bias == True:
      bias = _nn_variable('bias', [n_out], 'zero',
                          collection='biases' if trainable else None,
                          **kwargs)
      y = tf.nn.bias_add(y, bias)
    # apply ReLU
    if relu == True:
      y = tf.nn.relu(y)
    return y


def avg_pool(x, ksize, stride, padding='SAME', scope=None):
  with tf.variable_scope(scope or 'avg_pool'):
    y = tf.nn.avg_pool(x, [1,ksize,ksize,1], [1,stride,stride,1], padding)
  return y


def max_pool(x, ksize, stride, padding='SAME', scope=None):
  with tf.variable_scope(scope or 'max_pool'):
    y = tf.nn.max_pool(x, [1,ksize,ksize,1], [1,stride,stride,1], padding)
  return y


# def linear(x, n_in, n_out, bias=True, scope=None):
#   with tf.variable_scope(scope or 'linear'):
#     weight_init_std = math.sqrt(1.0 / n_out)
#     weight = tf.get_variable('weight', shape=[n_in,n_out],
#       initializer=tf.truncated_normal_initializer(0.0, weight_init_std))
#     tf.add_to_collection('weights', weight)
#     y = tf.matmul(x, weight)
#     if bias == True:
#       bias = tf.get_variable('bias', shape=[n_out],
#         initializer=tf.constant_initializer(0.0))
#       tf.add_to_collection('biases', bias)
#       y = y + bias
#   return y


# def mlp(x, n_in, n_hidden, n_out, activation=tf.nn.relu, scope=None):
#   with tf.variable_scope(scope or 'mlp'):
#     y = linear(x, n_in, n_hidden, scope='Linear1')
#     y = activation(y)
#     y = linear(y, n_hidden, n_out, scope='Linear2')
#   return y


def score_loss(gt_labels, match_scores, n_classes):
  """
  Classification loss
  ARGS
    gt_labels: int32 [n]
    match_scores: [n, n_classes]
  RETURN
    loss
  """
  embeddings = tf.one_hot(tf.cast(gt_labels, tf.int64), n_classes, 1.0, 0.0)
  losses = tf.nn.softmax_cross_entropy_with_logits(match_scores, embeddings)
  return tf.reduce_sum(losses)


def smooth_l1_loss(offsets, gt_offsets, scope=None):
  """
  Smooth L1 loss between offsets and encoded_gt
  ARGS
    offsets: [m?, 5], predicted offsets for one example
    gt_offsets: [m?, 5], correponding groundtruth offsets
  RETURN
    loss: scalar
  """
  with tf.variable_scope(scope or 'smooth_l1_loss'):
    gt_offsets = tf.stop_gradient(gt_offsets)
    diff = tf.abs(offsets - gt_offsets)
    lesser_mask = tf.cast(tf.less(diff, 1.0), tf.float32)
    larger_mask = 1.0 - lesser_mask
    losses = (0.5 * tf.square(diff)) * lesser_mask + (diff - 0.5) * larger_mask
    return tf.reduce_sum(losses, 1)


oplib = load_oplib(LIB_NAME)

# map C++ operators to python objects
sample_crop_bbox = oplib.sample_crop_bbox
encode_groundtruth = oplib.encode_groundtruth
# decode_local_rboxes = oplib.decode_local_rboxes
decode_segments_links = oplib.decode_segments_links
combine_segments = oplib.combine_segments
clip_rboxes = oplib.clip_rboxes
polygons_to_rboxes = oplib.polygons_to_rboxes
detection_mask = oplib.detection_mask
project_polygons = oplib.project_polygons

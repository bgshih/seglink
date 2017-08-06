import os, sys, re, logging
import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


def setup_logger(log_file_path):
  """
  Setup a logger that simultaneously output to a file and stdout
  ARGS
    log_file_path: string, path to the logging file
  """
  # logging settings
  log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
  root_logger = logging.getLogger()
  root_logger.setLevel(logging.DEBUG)
  # file handler
  log_file_handler = logging.FileHandler(log_file_path)
  log_file_handler.setFormatter(log_formatter)
  root_logger.addHandler(log_file_handler)
  # stdout handler
  log_stream_handler = logging.StreamHandler(sys.stdout)
  log_stream_handler.setFormatter(log_formatter)
  root_logger.addHandler(log_stream_handler)

  logging.info('Log file is %s' % log_file_path)


def log_flags(FLAGS):
  """
  Log all variables defined in FLAGS and their values.
  """
  param_list = []
  for k, v in FLAGS.__dict__['__flags'].items():
    param_list.append("'--{}': {}".format(k, v))
  logging.info('Parameters:\n' + '\n'.join(param_list))


def log_git_version():
  """
  Log git version number and uncommitted changes if any.
  """
  import subprocess
  version_str = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()
  logging.info('Git commit is {}'.format(version_str))
  changes = str(subprocess.check_output(['git', 'diff']))
  logging.info('Uncommitted chanages:\n' + changes)


def summarize_activations(tensors, collections=['detailed_summaries'], tower_name='tower'):
  if not isinstance(tensors, list):
    tensors = [tensors]
  for x in tensors:
    tensor_name = re.sub('%s_[0-9]*/' % tower_name, '', x.op.name)
    logging.info('Register activation summary for %s' % tensor_name)
    tf.summary.histogram(tensor_name + '/activations', x, collections=collections)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x), collections=collections)


def summarize_losses(losses, collections=['brief_summaries']):
  if not isinstance(losses, list):
    losses = [losses]

  loss_ema = tf.train.ExponentialMovingAverage(0.9, name='ema')
  loss_ema_update = loss_ema.apply(losses)

  with tf.control_dependencies([loss_ema_update]):
    for l in losses:
      tf.summary.scalar(l.op.name +' (raw)', l, collections=collections)
      tf.summary.scalar(l.op.name, loss_ema.average(l), collections=collections)


def print_tensor_summary(tensor, tag=None, n_print=21):
  tensor_min = tf.reduce_min(tensor)
  tensor_max = tf.reduce_max(tensor)
  tensor_avg = tf.reduce_mean(tensor)
  tensor_zero_fraction = tf.nn.zero_fraction(tensor)
  tensor_shape = tf.shape(tensor)
  tag = tag or tensor.name
  tensor = tf.Print(tensor,
                    [tensor_min, tensor_max, tensor_avg, tensor_zero_fraction, tensor_shape, tensor],
                    message=(tag + ' Min, max, mean, sparsity, shape, value:'),
                    summarize=n_print)
  return tensor


def mkdir_if_not_exist(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)


def rboxes_to_polygons(rboxes):
  """
  Convert rboxes to polygons
  ARGS
    `rboxes`: [n, 5]
  RETURN
    `polygons`: [n, 8]
  """
  theta = rboxes[:,4:5]
  cxcy = rboxes[:,:2]
  half_w = rboxes[:,2:3] / 2.
  half_h = rboxes[:,3:4] / 2.
  v1 = np.hstack([np.cos(theta) * half_w, np.sin(theta) * half_w])
  v2 = np.hstack([-np.sin(theta) * half_h, np.cos(theta) * half_h])
  p1 = cxcy - v1 - v2
  p2 = cxcy + v1 - v2
  p3 = cxcy + v1 + v2
  p4 = cxcy - v1 + v2
  polygons = np.hstack([p1, p2, p3, p4])
  return polygons


def rboxes_to_bboxes(rboxes):
  """
  Calculate the bounding boxes of rboxes
  ARGS
    `rboxes`: [n, 5]
  RETURN
    `bboxes`: [n, 4]
  """
  polygons = _rboxes_to_polygons(rboxes)
  xmin = np.min(polygons[:,::2], axis=1, keepdims=True)
  ymin = np.min(polygons[:,1::2], axis=1, keepdims=True)
  xmax = np.max(polygons[:,::2], axis=1, keepdims=True)
  ymax = np.max(polygons[:,1::2], axis=1, keepdims=True)
  bboxes = np.hstack([xmin, ymin, xmax, ymax])
  return bboxes

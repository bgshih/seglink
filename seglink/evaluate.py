import os, sys, math, time, logging, random
import tensorflow as tf
import numpy as np
import visualizations
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import joblib

import model
import data
import utils
import ops

FLAGS = tf.app.flags.FLAGS
# logging
tf.app.flags.DEFINE_string('log_dir', '', 'Directory for saving checkpoints and log files')
tf.app.flags.DEFINE_string('log_prefix', '', 'Log file name prefix')
# testing
tf.app.flags.DEFINE_string('image_resize_method', 'fixed', 'Image resizing method. "fixed" or "dynamic"')
tf.app.flags.DEFINE_string('test_model', '', 'Checkpoint for testing')
tf.app.flags.DEFINE_string('test_dataset', '', 'Test dataset path')
tf.app.flags.DEFINE_integer('test_batch_size', 32, 'Test batch size')
tf.app.flags.DEFINE_integer('num_test', 500, 'Number of test images')
tf.app.flags.DEFINE_float('node_threshold', 0.5, 'Confidence threshold for nodes')
tf.app.flags.DEFINE_float('link_threshold', 0.5, 'Confidence threshold for links')
tf.app.flags.DEFINE_integer('save_vis', 0, 'Save visualization results')
tf.app.flags.DEFINE_string('vis_save_dir', '', 'Visualization save directory')
tf.app.flags.DEFINE_string('result_format', 'icdar_2015_inc', 'Result file format')
tf.app.flags.DEFINE_string('result_suffix', time.strftime('_%Y%m%d_%H%M%S'), 'Result file suffix')
# post processing
tf.app.flags.DEFINE_float('bbox_scale', 1.0, 'Scale output bounding box')
tf.app.flags.DEFINE_float('bbox_min_area', 0, 'Minimum bounding box area')
# intermediate results
tf.app.flags.DEFINE_integer('load_intermediate', 0, 'Whether to load intermediate results.')
tf.app.flags.DEFINE_integer('save_intermediate', 0, 'Whether to load intermediate results.')
# useless flags, do not set
tf.app.flags.DEFINE_string('weight_init_method', 'xavier', 'Weight initialization method')

def evaluate():
  with tf.device('/cpu:0'):
    # input data
    streams = data.input_stream(FLAGS.test_dataset)
    pstreams = data.test_preprocess(streams)
    if FLAGS.test_resize_method == 'dynamic':
      # each test image is resized to a different size
      # test batch size must be 1
      assert(FLAGS.test_batch_size == 1)
      batches = tf.train.batch(pstreams,
                               FLAGS.test_batch_size,
                               capacity=1000,
                               num_threads=1,
                               dynamic_pad=True)
    else:
      # resize every image to the same size
      batches = tf.train.batch(pstreams,
                               FLAGS.test_batch_size,
                               capacity=1000,
                               num_threads=1)
    image_size = tf.shape(batches['image'])[1:3]

  fetches = {}
  fetches['images'] = batches['image']
  fetches['image_name'] = batches['image_name']
  fetches['resize_size'] = batches['resize_size']
  fetches['orig_size'] = batches['orig_size']

  # detector
  detector = model.SegLinkDetector()
  all_maps = detector.build_model(batches['image'])

  # decode local predictions
  all_nodes, all_links, all_reg = [], [], []
  for i, maps in enumerate(all_maps):
    cls_maps, lnk_maps, reg_maps = maps
    reg_maps = tf.multiply(reg_maps, data.OFFSET_VARIANCE)

    # segments classification
    cls_prob = tf.nn.softmax(tf.reshape(cls_maps, [-1, 2]))
    cls_pos_prob = cls_prob[:, model.POS_LABEL]
    cls_pos_prob_maps = tf.reshape(cls_pos_prob, tf.shape(cls_maps)[:3])
    # node status is 1 where probability is higher than threshold
    node_labels = tf.cast(tf.greater_equal(cls_pos_prob_maps, FLAGS.node_threshold),
                          tf.int32)

    # link classification
    lnk_prob = tf.nn.softmax(tf.reshape(lnk_maps, [-1, 2]))
    lnk_pos_prob = lnk_prob[:, model.POS_LABEL]
    lnk_shape = tf.shape(lnk_maps)
    lnk_pos_prob_maps = tf.reshape(lnk_pos_prob,
                                   [lnk_shape[0], lnk_shape[1], lnk_shape[2], -1])
    # link status is 1 where probability is higher than threshold
    link_labels = tf.cast(tf.greater_equal(lnk_pos_prob_maps, FLAGS.link_threshold),
                          tf.int32)

    all_nodes.append(node_labels)
    all_links.append(link_labels)
    all_reg.append(reg_maps)

    fetches['link_labels_%d' % i] = link_labels

  # decode segments and links
  segments, group_indices, segment_counts = ops.decode_segments_links(
    image_size, all_nodes, all_links, all_reg,
    anchor_sizes=list(detector.anchor_sizes))
  fetches['segments'] = segments
  fetches['group_indices'] = group_indices
  fetches['segment_counts'] = segment_counts

  # combine segments
  combined_rboxes, combined_counts = ops.combine_segments(
    segments, group_indices, segment_counts)
  fetches['combined_rboxes'] = combined_rboxes
  fetches['combined_counts'] = combined_counts

  sess_config = tf.ConfigProto()
  with tf.Session(config=sess_config) as sess:
    # load model
    model_loader = tf.train.Saver()
    model_loader.restore(sess, FLAGS.test_model)

    batch_size = FLAGS.test_batch_size
    n_batches = int(math.ceil(FLAGS.num_test / batch_size))

    # result directory
    result_dir = os.path.join(FLAGS.log_dir, 'results' + FLAGS.result_suffix)
    utils.mkdir_if_not_exist(result_dir)

    intermediate_result_path = os.path.join(FLAGS.log_dir, 'intermediate.pkl')
    if FLAGS.load_intermediate:
      all_batches = joblib.load(intermediate_result_path)
      logging.info('Intermediate result loaded from {}'.format(intermediate_result_path))
    else:
      # run all batches and store results in a list
      all_batches = []
      with slim.queues.QueueRunners(sess):
        for i in range(n_batches):
          logging.info('Evaluating batch %d/%d' % (i+1, n_batches))
          sess_outputs = sess.run(fetches)
          all_batches.append(sess_outputs)
      if FLAGS.save_intermediate:
        joblib.dump(all_batches, intermediate_result_path, compress=5)
        logging.info('Intermediate result saved to {}'.format(intermediate_result_path))

    # # visualize local rboxes (TODO)
    # if FLAGS.save_vis:
    #   vis_save_prefix = os.path.join(save_dir, 'localpred_batch_%d_' % i)
    #   pred_rboxes_counts = []
    #   for j in range(len(all_maps)):
    #     pred_rboxes_counts.append((sess_outputs['segments_det_%d' % j],
    #                               sess_outputs['segment_counts_det_%d' % j]))
    #   _visualize_layer_det(sess_outputs['images'],
    #                       pred_rboxes_counts,
    #                       vis_save_prefix)

    # # visualize joined rboxes (TODO)
    # if FLAGS.save_vis:
    #   vis_save_prefix = os.path.join(save_dir, 'batch_%d_' % i)
    #   # _visualize_linked_det(sess_outputs, save_prefix)
    #   _visualize_combined_rboxes(sess_outputs, vis_save_prefix)

    if FLAGS.result_format == 'icdar_2015_inc':
      postprocess_and_write_results_ic15(all_batches, result_dir)
    elif FLAGS.result_format == 'icdar_2013':
      postprocess_and_write_results_ic13(all_batches, result_dir)
    else:
      logging.critical('Unknown result format: {}'.format(FLAGS.result_format))
      sys.exit(1)
  
  logging.info('Evaluation done.')


def postprocess_and_write_results_ic15(all_batches, result_dir):
  test_count = 0

  for batch in all_batches:
    for i in range(FLAGS.test_batch_size):
      # the last batch may contain duplicates
      if test_count > FLAGS.num_test: break

      rboxes = batch['combined_rboxes'][i]
      count = batch['combined_counts'][i]
      rboxes = rboxes[:count, :]

      # post processings
      if FLAGS.bbox_scale > 1.0:
        rboxes[:, 3:5] *= FLAGS.bbox_scale

      # convert rboxes to polygons and find its coordinates on the original image
      orig_h, orig_w = batch['orig_size'][i]
      resize_h, resize_w = batch['resize_size'][i]
      polygons = utils.rboxes_to_polygons(rboxes)
      scale_y = float(orig_h) / float(resize_h)
      scale_x = float(orig_w) / float(resize_w)

      # confine polygons inside image
      polygons[:, ::2] = np.maximum(0, np.minimum(polygons[:, ::2] * scale_x, orig_w-1))
      polygons[:, 1::2] = np.maximum(0, np.minimum(polygons[:, 1::2] * scale_y, orig_h-1))
      polygons = np.round(polygons).astype(np.int32)

      # write results to text files
      image_name = batch['image_name'][i].decode('ascii')
      result_fname = 'res_{}.txt'.format(os.path.splitext(image_name)[0])
      orig_size = batch['orig_size'][i]
      save_path = os.path.join(result_dir, result_fname)
      with open(save_path, 'w') as f:
        lines = []
        for k in range(polygons.shape[0]):
          poly_str = list(polygons[k])
          poly_str = [str(o) for o in poly_str]
          poly_str = ','.join(poly_str)
          lines.append(poly_str)
        # remove duplicated lines
        lines = list(frozenset(lines))
        f.write('\r\n'.join(lines))
        logging.info('Detection results written to {}'.format(save_path))
        
        test_count += 1
  
  # compress results into a single zip file
  result_dir_name = 'results' + FLAGS.result_suffix
  cmd = "zip -rj {}.zip {}".format(os.path.join(result_dir, '..', result_dir_name),
                                   result_dir)
  logging.info('Executing {}'.format(cmd))
  os.system(cmd)


def postprocess_and_write_results_ic13(all_results):
  raise NotImplementedError('This function needs revision')
  
  for j in range(batch_size):
    # convert detection results
    rboxes = sess_outputs['combined_rboxes'][j]
    count = sess_outputs['combined_counts'][j]
    orig_h, orig_w = sess_outputs['orig_size'][j]
    resize_h, resize_w = sess_outputs['resize_size'][j]
    bboxes = utils.rboxes_to_bboxes(rboxes[:count, :])

    # bbox scaling trick
    bbox_scale = FLAGS.bbox_scale
    bboxes_width = bboxes[:,2] - bboxes[:,0]
    bboxes_height = bboxes[:,3] - bboxes[:,1]
    bboxes[:,0] -= 0.5 * bbox_scale * bboxes_width
    bboxes[:,1] -= 0.5 * bbox_scale * bboxes_height
    bboxes[:,2] += 0.5 * bbox_scale * bboxes_width
    bboxes[:,3] += 0.5 * bbox_scale * bboxes_height

    scale_y = float(orig_h) / float(resize_h)
    scale_x = float(orig_w) / float(resize_w)
    bboxes[:, ::2] = np.maximum(0, np.minimum(bboxes[:, ::2] * scale_x, orig_w-1))
    bboxes[:, 1::2] = np.maximum(0, np.minimum(bboxes[:, 1::2] * scale_y, orig_h-1))
    bboxes = np.round(bboxes).astype(np.int32)

    # write results to text files
    image_name = str(sess_outputs['image_name'][j])
    result_fname = 'res_' + os.path.splitext(image_name)[0] + '.txt'
    orig_size = sess_outputs['orig_size'][j]
    save_path = os.path.join(result_dir, result_fname)
    with open(save_path, 'w') as f:
      lines = []
      for k in range(bboxes.shape[0]):
        bbox_str = list(bboxes[k])
        bbox_str = [str(o) for o in bbox_str]
        bbox_str = ','.join(bbox_str)
        lines.append(bbox_str)
      # remove duplicated lines
      lines = list(set(lines))
      f.write('\r\n'.join(lines))
      logging.info('Detection results written to {}'.format(save_path))

    # save images and lexicon list for post-processing
    if FLAGS.save_image_and_lexicon:
      sess_outputs['']


if __name__ == '__main__':
  # create logging dir if not existed
  utils.mkdir_if_not_exist(FLAGS.log_dir)
  # set up logging
  log_file_name = FLAGS.log_prefix + time.strftime('%Y%m%d_%H%M%S') + '.log'
  log_file_path = os.path.join(FLAGS.log_dir, log_file_name)
  utils.setup_logger(log_file_path)
  utils.log_flags(FLAGS)
  utils.log_git_version()
  # run test
  evaluate()

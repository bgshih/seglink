import math
import os

import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import ops
import utils
import model_fctd
import data
import config
import visualizations as vis

FLAGS = tf.app.flags.FLAGS


def test_encode_decode_synth_data():
  batch_size = 50
  n_gt_max = 4
  image_h = 150
  image_w = 300
  image_size = [image_h, image_w]
  map_sizes = [[19, 38], [10, 19], [5, 10], [3, 5], [2, 3], [1, 1]]
  n_stages = len(map_sizes)
  # region_sizes = 300. * np.minimum(FLAGS.region_size_alpha / np.asarray([38, 19, 10, 5, 3, 1]), 0.95)
  region_sizes = [11.84210526, 23.68421053, 45., 90., 150., 285.]
  print(region_sizes)
  pos_thresh = 1.5
  neg_thresh = 2.0

  def _generate_random_gt(batch_size, n_gt_max):
    gt_cx = image_w * np.random.uniform(low=0.2, high=0.8, size=[batch_size, n_gt_max, 1])
    gt_cy = image_h * np.random.uniform(low=0.2, high=0.8, size=[batch_size, n_gt_max, 1])
    gt_w = image_w * np.random.uniform(low=0.2, high=1, size=[batch_size, n_gt_max, 1])
    gt_h = image_h * np.random.uniform(low=0.05, high=0.5, size=[batch_size, n_gt_max, 1])
    gt_theta = np.random.uniform(low=-0.5, high=0.5, size=[batch_size, n_gt_max, 1])
    gt_rboxes = np.concatenate([gt_cx, gt_cy, gt_w, gt_h, gt_theta], axis=2)
    return gt_rboxes

  def _visualize(ax, match_status, local_gt, gt_rboxes, gt_counts,
                 decoded_pred, decoded_counts, link_status=None):
    """
    Visualize encoded groundtruth
    ARGS
      ax: pyplot axis
      match_status: int [map_h, map_w] match status
      local_gt: [map_h, map_w, rbox_dim] encoded groundtruths
      gt_rboxes: [5]
      gt_counts: []
      decoded_pred: [n_decoded_pred_max, 5]
      decoded_counts: int []
      link_status: int [map_h, map_w, 8] link status
    """
    map_h, map_w, _ = local_gt.shape
    step_x = float(image_w) / map_w
    step_y = float(image_h) / map_h

    # visualize regions
    region_bboxes = []
    for p in range(map_h * map_w):
      px = p % map_w
      py = int(math.floor(p / map_w))
      grid_cx = (0.5 + px) * step_x
      grid_cy = (0.5 + py) * step_y
      region_bboxes.append([grid_cx, grid_cy, region_size, region_size, 0])
    region_bboxes = np.asarray(region_bboxes)
    # utils.visualize_rboxes(ax, region_bboxes, edgecolor='pink', facecolor='pink', alpha=0.5)

    # visualize groundtruth
    vis.visualize_rboxes(ax, gt_rboxes[:gt_counts, :],
        verbose=False, edgecolor='green', facecolor='none', linewidth=2)

    # visualize grid
    for p in range(map_h * map_w):
      px = p % map_w
      py = p // map_w
      grid_cx = (0.5 + px) * step_x
      grid_cy = (0.5 + py) * step_y
      match_status_p = match_status[py, px]

      # draw grid center point as a circle
      if match_status_p == 1: # positive
        circle_color = 'red'
      elif match_status_p == 0: # ignore
        circle_color = 'yellow'
      else: # negative
        circle_color = 'blue'
      circle = plt.Circle((grid_cx, grid_cy), 2, color=circle_color)
      ax.add_artist(circle)

      # # visualize decoded predictions
      # utils.visualize_rboxes(ax, decoded_pred[:decoded_counts, :],
      #     edgecolor='green', facecolor='green', alpha=0.5)

    if link_status is not None:
      # visulaize link status
      for p in range(map_h * map_w):
        px = p % map_w
        py = int(math.floor(p / map_w))
        grid_cx = (0.5 + px) * step_x
        grid_cy = (0.5 + py) * step_y
        link_status_p = link_status[py, px, :]

        idx = 0
        for ny in [py - 1, py, py + 1]:
          for nx in [px - 1, px, px + 1]:
            if ny == py and nx == px:
              # skip self link
              continue
            if link_status_p[idx] != -1:
              nb_cx = (0.5 + nx) * step_x
              nb_cy = (0.5 + ny) * step_y
              if link_status_p[idx] == 1:
                link_color = 'red'
              elif link_status_p[idx] == 0:
                link_color = 'yellow'
              else:
                raise('Internal error')
              ax.plot((grid_cx, nb_cx), (grid_cy, nb_cy),
                      color=link_color, alpha=0.5, linewidth=2)
            idx += 1

  # generate random number of random groundtruths
  gt_rboxes = _generate_random_gt(batch_size, n_gt_max)
  gt_counts = np.random.randint(low=1, high=n_gt_max, size=[batch_size])

  node_status_below = [[[]]]
  match_indices_below = [[[]]]

  # fetch encoding & decoding results on all stages
  fetches = {}
  for i in range(n_stages):
    map_size = map_sizes[i]
    region_size = region_sizes[i]
    match_status, link_status, local_gt, match_indices = ops.encode_groundtruth(
        gt_rboxes, gt_counts, map_size, image_size,
        node_status_below, match_indices_below,
        region_size=region_size,
        pos_scale_diff_thresh=pos_thresh,
        neg_scale_diff_thresh=neg_thresh,
        cross_links=False)
    decoded_pred, decoded_counts = ops.decode_prediction(
        match_status, local_gt, image_size, region_size=region_size)
    fetches['match_status_%d' % i] = match_status
    fetches['link_status_%d' % i] = link_status
    fetches['local_gt_%d' % i] = local_gt
    fetches['decoded_pred_%d' % i] = decoded_pred
    fetches['decoded_counts_%d' % i] = decoded_counts

  with tf.Session() as sess:
    sess_outputs = sess.run(fetches)
    fig = plt.figure()
    for i in range(batch_size):
      fig.clear()
      for j in range(n_stages):
        ax = fig.add_subplot(2, 3, j+1)
        ax.invert_yaxis()
        _visualize(ax,
            sess_outputs['match_status_%d' % j][i],
            sess_outputs['local_gt_%d' % j][i],
            gt_rboxes[i],
            gt_counts[i],
            sess_outputs['decoded_pred_%d' % j][i],
            sess_outputs['decoded_counts_%d' % j][i],
            # link_status=None)
            link_status=sess_outputs['link_status_%d' % j][i])
        ax.set_xlim(0, image_w)
        ax.set_ylim(0, image_h)
        ax.set_aspect('equal')
      save_path = os.path.join('../vis', 'local_gt_%d.png' % i)
      plt.savefig(save_path, dpi=200)
      print('Visualization saved to %s' % save_path)


def test_encode_decode_real_data():
  save_dir = '../vis/gt_link_node/'
  utils.mkdir_if_not_exist(save_dir)
  batch_size = 233

  streams = data.input_stream(FLAGS.train_record_path)
  pstreams = data.train_preprocess(streams)
  batch = tf.train.batch(pstreams, batch_size, num_threads=1, capacity=100)

  image_h = tf.shape(batch['image'])[1]
  image_w = tf.shape(batch['image'])[2]
  image_size = tf.pack([image_h, image_w])

  detector = model_fctd.FctdDetector()
  all_maps = detector.build_model(batch['image'])

  det_layers = ['det_conv4_3', 'det_fc7', 'det_conv6',
                'det_conv7', 'det_conv8', 'det_pool6']

  fetches = {}
  fetches['images'] = batch['image']
  fetches['image_size'] = image_size

  for i, det_layer in enumerate(det_layers):
    cls_maps, lnk_maps, reg_maps = all_maps[i]
    map_h, map_w = tf.shape(cls_maps)[1], tf.shape(cls_maps)[2]
    map_size = tf.pack([map_h, map_w])

    node_status_below = tf.constant([[[0]]], dtype=tf.int32)
    match_indices_below = tf.constant([[[0]]], dtype=tf.int32)
    cross_links = False # FIXME

    node_status, link_status, local_gt, match_indices = ops.encode_groundtruth(
        batch['rboxes'],
        batch['count'],
        map_size,
        image_size,
        node_status_below,
        match_indices_below,
        region_size=detector.region_sizes[i],
        pos_scale_diff_thresh=FLAGS.pos_scale_diff_threshold,
        neg_scale_diff_thresh=FLAGS.neg_scale_diff_threshold,
        cross_links=cross_links)

    fetches['node_status_%d' % i] = node_status
    fetches['link_status_%d' % i] = link_status
    fetches['local_gt_%d' % i] = local_gt

  def _visualize_nodes_links(ax, image, node_status, link_status, image_size):
    """
    Visualize nodes and links of one example.
    ARGS
      `node_status`: int [map_h, map_w]
      `link_status`: int [map_h, map_w, n_links]
      `image_size`: int [2]
    """
    ax.clear()
    image_display = vis.convert_image_for_visualization(
        image, mean_subtracted=True)
    ax.imshow(image_display)

    vis.visualize_nodes(ax, node_status, image_size)
    vis.visualize_links(ax, link_status, image_size)

  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    tf.train.start_queue_runners(sess=sess)

    sess_outputs = sess.run(fetches)

    fig = plt.figure()
    for i in xrange(batch_size):
      fig.clear()
      for j, det_layer in enumerate(det_layers):
        ax = fig.add_subplot(2, 3, j+1)
        _visualize_nodes_links(ax,
                               sess_outputs['images'][i],
                               sess_outputs['node_status_%d' % j][i],
                               sess_outputs['link_status_%d' % j][i],
                               sess_outputs['image_size'])

      save_path = os.path.join(save_dir, 'gt_node_link_%04d.jpg' % i)
      plt.savefig(save_path, dpi=200)
      print('Visualization saved to %s' % save_path)


def test_clip_rboxes():

  def _generate_random_rboxes(n_rboxes):
    rboxes = np.zeros((n_rboxes, 5))
    rboxes[:,0] = np.random.uniform(low=0.0, high=1.0, size=[n_rboxes])  # cx
    rboxes[:,1] = np.random.uniform(low=0.0, high=1.0, size=[n_rboxes])  # cy
    rboxes[:,2] = np.random.uniform(low=0.2, high=0.8, size=[n_rboxes])  # width
    rboxes[:,3] = np.random.uniform(low=0.0, high=0.3, size=[n_rboxes])  # height
    rboxes[:,4] = np.random.uniform(low=-1.0, high=1.0, size=[n_rboxes]) # theta
    return rboxes

  n_rboxes = 5
  rboxes = tf.constant(_generate_random_rboxes(n_rboxes), tf.float32)
  crop_bbox = tf.constant([0, 0, 1, 1], tf.float32)
  clipped_rboxes = ops.clip_rboxes(rboxes, crop_bbox)

  with tf.Session() as sess:
    fetches = {'rboxes': rboxes, 'clipped_rboxes': clipped_rboxes}
    sess_outputs = sess.run(fetches)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.invert_yaxis() # left-top is the origin
    ax.set_aspect('equal')
    ax.clear()

    # plot rboxes before & after clipping
    vis.visualize_rboxes(ax, sess_outputs['rboxes'],
        edgecolor='blue', facecolor='none', verbose=True)
    vis.visualize_rboxes(ax, sess_outputs['clipped_rboxes'],
        edgecolor='green', facecolor='none', verbose=True)

    save_path = os.path.join('../vis', 'clipped_rboxes.png')
    plt.savefig(save_path)
    print('Visualization saved to %s' % save_path)


def test_data_loading_and_preprocess():
  fig = plt.figure()
  ax = fig.add_subplot(111)

  def _visualize_example(save_path, image, gt_rboxes, mean_subtracted=True):
    ax.clear()
    # convert image
    image_display = vis.convert_image_for_visualization(
        image, mean_subtracted=mean_subtracted)
    # draw image
    ax.imshow(image_display)
    # draw groundtruths
    image_h = image_display.shape[0]
    image_w = image_display.shape[1]
    vis.visualize_rboxes(ax, gt_rboxes,
        edgecolor='yellow', facecolor='none', verbose=False)
    # save plot
    plt.savefig(save_path)

  n_batches = 10
  batch_size = 32

  save_dir = '../vis/example'
  utils.mkdir_if_not_exist(save_dir)

  streams = data.input_stream('../data/synthtext_train.tf')
  pstreams = data.train_preprocess(streams)
  batches = tf.train.shuffle_batch(pstreams, batch_size, capacity=2000, min_after_dequeue=20,
                                   num_threads=1)
  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    tf.train.start_queue_runners(sess=sess)
    for i in xrange(n_batches):
      fetches = {'images': batches['image'],
                 'gt_rboxes': batches['rboxes'],
                 'gt_counts': batches['count']}
      sess_outputs = sess.run(fetches)
      for j in xrange(batch_size):
        save_path = os.path.join(save_dir, '%04d_%d.jpg' % (i, j))
        gt_count = sess_outputs['gt_counts'][j]
        _visualize_example(save_path,
                           sess_outputs['images'][j],
                           sess_outputs['gt_rboxes'][j, :gt_count],
                           mean_subtracted=True)
        print('Visualization saved to %s' % save_path)


def test_max_pool_on_odd_sized_maps():
  size = 5
  x = np.random.rand(size, size).reshape(1,size,size,1).astype(np.float32)
  print(x[0,:,:,0])
  with tf.Session() as sess:
    y = tf.nn.max_pool(x, [1,2,2,1], [1,2,2,1], 'SAME')
    print(y.eval()[0,:,:,0])


def test_decode_combine_rboxes():
  x = [np.random.rand(4,4).astype(np.float32),
       np.random.rand(5,5).astype(np.float32),
       np.random.rand(6,6).astype(np.float32)]
  y, _ = ops.decode_combine_rboxes(x, x, x, [100, 100],
                                region_size=10, cell_size=10)
  import ipdb; ipdb.set_trace()
  with tf.Session() as sess:
    y.eval()
  pass


if __name__ == '__main__':
  # test_encode_decode_synth_data()
  test_encode_decode_real_data()
  # test_clip_rboxes()
  # test_data_loading_and_preprocess()
  # test_max_pool_on_odd_sized_maps()
  # test_decode_combine_rboxes()

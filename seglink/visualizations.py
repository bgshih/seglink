import os, sys, re, logging
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def convert_image_for_visualization(image_data, mean_subtracted=True):
  """
  Convert image data from tensorflow to displayable format
  """
  import data

  image = image_data
  if mean_subtracted:
    image = image + np.asarray(data.IMAGE_BGR_MEAN, np.float32)
  if FLAGS.image_channel_order == 'BGR':
    image = image[:,:,::-1] # BGR => RGB
  image = np.floor(image).astype(np.uint8)
  return image

def visualize_bboxes(images, bboxes, scores=None, score_threshold=None,
                     output_dir='', prefix='bboxes_', channels='RGB', layout='NCHW'):
  """
  Visualize bounding boxes on images
  ARGS
    images: uint8
    bboxes: [N,n,4], or [n,4], or list of [?,4], normalized bboxes
    scores: [N,n,n_classes] bounding boxes scores
    score_threshold: bboxes whose scores are below this will be ignored
    prefix: prefix for save file name
    output_dir: output directory
    channels: color channel order, 'RGB' or 'BGR'
  """
  batch_size = images.shape[0]
  for i in range(batch_size):
    if layout == 'NCHW':
      image_display = np.transpose(images[i], axes=[1,2,0])
    elif layout == 'NHWC':
      image_display = np.copy(images[i])
    else:
      raise 'Unknown layout: %s' % layout

    # image_display is now HWC
    image_h = image_display.shape[0]
    image_w = image_display.shape[1]

    if channels == 'BGR': # convert to RGB
      image_display = image_display[:,:,::-1]

    # normalize to [0,1]
    image_display = image_display / 255.0

    plt.clf()
    plt.imshow(image_display)

    # if scores is not None and score_threshold is not None:
    #   bboxes_display = scores[]

    if isinstance(bboxes, list):
      bboxes_display = np.copy(bboxes[i])
    elif bboxes.ndim == 3:
      bboxes_display = np.copy(bboxes[i])
    elif bboxes.ndim == 2:
      bboxes_display = np.copy(bboxes)
    else:
      assert(False)

    bboxes_display[:,0::2] *= image_w
    bboxes_display[:,1::2] *= image_h
    for j, bbox in enumerate(bboxes_display):
      plt.gca().add_patch(mpl.patches.Rectangle(
          (bbox[0],bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
          fill=False, edgecolor=np.random.rand(3,1), linewidth=2))

    save_path = os.path.join(output_dir, '%s%d.png' % (prefix, i))
    plt.savefig(save_path)
    logging.info('Visualization saved to %s' % save_path)
      # plt.text(bbox[i,0], bbox[i,1], '#%d' % (i+1), color='#00ff00')

def visualize_rboxes(ax, rboxes, colors=None, verbose=False, **kwargs):
  """
  Visualize rotated bounding boxes
  ARGS
    ax: pyplot axis
    rboxes: array [n_rboxes, 5] or [5], (cx, cy, w, h, theta)
    colors: rboxes colors
    verbose: print extra information
    kwargs: extra arguments passed to mpl.patches.Rectangle
  """
  if rboxes.ndim == 1:
    rboxes = np.expand_dims(rboxes, axis=0)
  for i in range(rboxes.shape[0]):
    cx, cy, w, h, theta = rboxes[i]
    trans = mpl.transforms.Affine2D().rotate_around(cx, cy, theta) + ax.transData
    ec = 'green' if colors is None else colors[i]
    rect = mpl.patches.Rectangle((cx - 0.5 * w, cy - 0.5 * h), w, h,
                                 facecolor='none', edgecolor=ec, **kwargs)
    rect.set_transform(trans)
    ax.add_patch(rect)
    # c = 'green' if colors is None else colors[i]
    # center_dot = mpl.patches.Circle((cx, cy), 0.005, color=c)
    # ax.add_patch(center_dot)

    if verbose:
      print('Plotted rbox: (%.2f %.2f %.2f %.2f %.2f)' % (cx, cy, w, h, theta))

def visualize_nodes(ax, node_status, image_size):
  """
  Visualize a grid of nodes with colored dots
  ARGS
    `ax`: pyplot axis
    `node_status`: int [map_h, map_w]
    `image_size`: int [2]
  """
  map_h, map_w = node_status.shape
  image_h, image_w = image_size
  step_x = float(image_w) / map_w
  step_y = float(image_h) / map_h
  for p in xrange(map_h * map_w):
    px, py = p % map_w, p // map_w
    grid_cx = (0.5 + px) * step_x
    grid_cy = (0.5 + py) * step_y
    node_status_p = node_status[py, px]

    # draw grid center point as a circle
    if node_status_p == 1: # positive
      circle_color = 'red'
    elif node_status_p == 0: # ignore
      circle_color = 'yellow'
    elif node_status_p == -1: # negative
      circle_color = 'blue'
    else:
      raise 'Internal error, node_status_p == %d' % node_status_p

    circle = plt.Circle((grid_cx, grid_cy), 2, color=circle_color)
    ax.add_artist(circle)

def visualize_links(ax, link_status, image_size,
                    link_status_below=None,
                    cross_stride=None):
  """
  Visualize links (local and cross) with colored lines
  ARGS
    `ax`: pyplot axis
    `link_status`: int [map_h, map_w, n_links]
    `image_size`: int [2]
    `link_status_below`: (optional) [below_h, below_w, n_links_below]
    `cross_stride`: (optional) int
  """
  map_h, map_w, n_links = link_status.shape
  image_h, image_w = image_size
  step_x = float(image_w) / map_w
  step_y = float(image_h) / map_h
  for p in xrange(map_h * map_w):
    px, py = p % map_w, p // map_w
    grid_cx = (0.5 + px) * step_x
    grid_cy = (0.5 + py) * step_y
    link_status_p = link_status[py, px]

    # visualize same-layer links
    idx = 0
    for ny in [py - 1, py, py + 1]:
      for nx in [px - 1, px, px + 1]:
        if ny == py and nx == px:
          # skip self link
          continue
        # if link_status_p[idx] != -1:
        if link_status_p[idx] == 1: # FIXME: see the line above
          nb_cx = (0.5 + nx) * step_x
          nb_cy = (0.5 + ny) * step_y
          if link_status_p[idx] == 1:
            link_color = 'red'
          elif link_status_p[idx] == 0:
            link_color = 'yellow'
          else:
            raise('Internal error, link_status_p[idx] == %d' % link_status_p[idx])
          ax.plot((grid_cx, nb_cx), (grid_cy, nb_cy),
                  color=link_color, alpha=0.3, linewidth=1)
        idx += 1
    assert(idx == FLAGS.n_local_links)

    # visualize cross-layer links
    if link_status_below is not None:
      assert(n_links > FLAGS.n_local_links)
      assert(cross_stride == 2) # FIXME: put into FLAGS
      below_h, below_w, _ = link_status_below.shape
      step_x_below = float(image_w) / below_w
      step_y_below = float(image_h) / below_h
      y_start = min(cross_stride * py, below_h - cross_stride)
      y_end = min(cross_stride * (py + 1), below_h)
      x_start = min(cross_stride * px, below_w - cross_stride)
      x_end = min(cross_stride * (px + 1), below_w)
      idx = 0
      for ny in range(y_start, y_end):
        for nx in range(x_start, x_end):
          l = link_status_p[FLAGS.n_local_links + idx]
          if l == 1:
            nb_cx = (0.5 + nx) * step_x_below
            nb_cy = (0.5 + ny) * step_y_below
            ax.plot((grid_cx, nb_cx), (grid_cy, nb_cy),
                    color='green', alpha=1, linewidth=1)
          idx += 1

def visualize_detection_each_layer(sess_outputs, save_dir):
  """
  Visualize local rboxes of each layer on separated axis.
  ARGS:
    `sess_outputs`: TF session outputs
    `save_dir`: saving directory
  """
  raise NotImplemented

  fig.clear()
  ax = fig.add_subplot(211)
  ax2 = fig.add_subplot(212) # draw original image

  # different colors for differnt conv detector
  det_names = ['conv4_3', 'fc7', 'conv6', 'conv7', 'conv8', 'pool6']
  color_list = ['green', 'blue', 'red', 'yellow', 'magenta', 'brown']

  images = sess_outputs['images']

  n_images = images.shape[0]
  for i in range(n_images):
    legend_handles = []
    image_i = images[i]
    # display image
    image_display = convert_image_for_visualization(image_i)
    ax.imshow(image_display)
    ax2.imshow(image_display)
    # display local predictions
    for j, pairs in enumerate(pred_rboxes_counts):
      pred_rboxes, pred_counts = pairs
      pred_rboxes_i = pred_rboxes[i,:pred_counts[i],:]
      # display local predictions
      utils.visualize_rboxes(ax, pred_rboxes_i,
        edgecolor=color_list[j], facecolor='none', verbose=False)
      # legends
      dummy_rect = mpl.patches.Rectangle((0, 0), 0, 0,
          edgecolor=color_list[j], facecolor='none', label=det_names[j])
      legend_handles.append(dummy_rect)
    ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1,0.5))
    # save figure
    save_path = prefix + '%d.png' % i
    plt.savefig(save_path, dpi=300, format='png', bbox_inches='tight')
    print('Visualization saved to %s' % save_path)

def visualize_segments_and_links(sess_outputs, save_dir):
  """
  Visualize linked local rboxes on one figure.
  """
  fig.clear()
  ax = fig.add_subplot(211)
  ax2 = fig.add_subplot(212) # for drawing original images

  images = sess_outputs['images']
  group_indices = sess_outputs['group_indices']
  rboxes = sess_outputs['local_rboxes']
  counts = sess_outputs['local_counts']

  # different colors for differnt conv detector
  color_list = []
  for name, _ in mpl.colors.cnames.iteritems():
    color_list.append(name)
  random.shuffle(color_list)

  n_images = images.shape[0]
  for i in range(n_images):
    legend_handles = []
    image_i = images[i]
    # display image
    image_display = convert_image_for_visualization(image_i)
    ax.imshow(image_display)
    ax2.imshow(image_display)
    # display local predictions
    colors = [color_list[j % len(color_list)] for j in group_indices[i, :counts[i]]]
    visualize_rboxes(ax, rboxes[i, :counts[i], :], colors=colors, verbose=False)
    # display links
    for j in xrange(len(detector.det_layers)):
      link_status = sess_outputs['link_status_%d' % j][i]
      link_status_below = None if j == 0 else sess_outputs['link_status_%d' % (j-1)][i]
      visualize_links(ax, link_status, image_size,
                      link_status_below=link_status_below,
                      cross_stride=2)
    # save
    image_id = str(sess_outputs['image_name'][i])
    save_path = os.path.join(save_dir, '{}.jpg'.format(image_id))
    plt.savefig(save_path, dpi=300, format='jpg', bbox_inches='tight')
    print('Visualization saved to %s' % save_path)

def visualize_combined_rboxes(sess_outputs, save_dir):
  """
  Visualize joined rboxes.
  """
  fig.clear()
  ax = fig.add_subplot(111)

  images = sess_outputs['images']
  rboxes = sess_outputs['combined_rboxes']
  counts = sess_outputs['combined_counts']

  n_images = images.shape[0]
  for i in range(n_images):
    legend_handles = []
    image_i = images[i]
    # display image
    image_display = convert_image_for_visualization(image_i)
    ax.imshow(image_display)
    # display local predictions
    visualize_rboxes(ax, rboxes[i, :counts[i], :], verbose=False)
    # save
    save_path = os.path.join(save_dir, '{}.jpg'.format(image_id))
    plt.savefig(save_path, dpi=300, format='jpg', bbox_inches='tight')
    print('Visualization saved to %s' % save_path)

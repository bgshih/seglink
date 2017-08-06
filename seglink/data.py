import tensorflow as tf
import numpy as np

import ops

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('sampling_overlap_mode', 'coverage', 'Sampling based on jaccard / coverage')
tf.app.flags.DEFINE_string('image_channel_order', 'BGR', 'Order of input image channels')
tf.app.flags.DEFINE_integer('max_num_gt', 300, 'Max number of groundtruths in one example, used for determining padding length')
tf.app.flags.DEFINE_string('test_resize_method', 'fixed', 'Image resizing method in testing {fixed, dynamic}')
tf.app.flags.DEFINE_integer('resize_longer_side', 512, 'Longer side of resized image')
tf.app.flags.DEFINE_integer('resize_step', 128, 'Width and height must be dividable by this number')
tf.app.flags.DEFINE_integer('image_height', 384, 'Resize image height')
tf.app.flags.DEFINE_integer('image_width', 384, 'Resize image width')


# constants
IMAGE_BGR_MEAN = np.array([104, 117, 123], dtype=np.float32)
RBOX_DIM = 5
OFFSET_DIM = 6
WORD_POLYGON_DIM = 8 # 4 vertices, each 2 coordinates
OFFSET_VARIANCE = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]


def input_stream(record_path, scope=None):
  """
  Input data stream
  ARGS
    `record_path`: tf records file path
  RETURN
    `streams`: data streams
  """
  with tf.device('/cpu:0'):
    with tf.variable_scope(scope or 'input_stream'):
      reader = tf.TFRecordReader()
      filename_queue = tf.train.string_input_producer([record_path], None)
      _, record_value = reader.read(filename_queue)
      features = tf.parse_single_example(record_value,
        {
          'image_jpeg': tf.FixedLenFeature([], tf.string),
          'image_name': tf.FixedLenFeature([], tf.string),
          'word_polygons': tf.VarLenFeature(tf.float32),
          # 'words': tf.VarLenFeature(tf.string) // FIXME: problem with parsing words
        })
      # decode jpeg image
      image = tf.cast(tf.image.decode_jpeg(features['image_jpeg'], channels=3), tf.float32)

      # extract bounding polygons
      word_polygons = tf.sparse_tensor_to_dense(features['word_polygons'])
      word_polygons = tf.reshape(word_polygons, [-1, WORD_POLYGON_DIM])

      # extract words
      # words = tf.sparse_tensor_to_dense(features['words'])

      # output streams
      streams = {'image': image,
                 'image_name': features['image_name'],
                 'image_jpeg': features['image_jpeg'],
                 'word_polygons': word_polygons}
      return streams


def train_preprocess(streams, scope=None):
  """
  Preprocess training images and groundtruths
  ARGS
    streams: input data streams
  RETURN
    pstreams: preprocessed data streams
  """
  with tf.variable_scope(scope or 'train_preprocess'):
    word_polygons = streams['word_polygons']
    image_shape = tf.shape(streams['image'])
    orig_h, orig_w = image_shape[0], image_shape[1]
    orig_size = tf.stack([orig_h, orig_w])

    # get envelope bounding boxes of words
    reshaped_polygons = tf.reshape(word_polygons, [-1, 4, 2])
    envelope_xmin = tf.reduce_min(reshaped_polygons[:,:,0], 1, keep_dims=True) # => [n,1]
    envelope_ymin = tf.reduce_min(reshaped_polygons[:,:,1], 1, keep_dims=True)
    envelope_xmax = tf.reduce_max(reshaped_polygons[:,:,0], 1, keep_dims=True)
    envelope_ymax = tf.reduce_max(reshaped_polygons[:,:,1], 1, keep_dims=True)
    envelope_bboxes = tf.concat([envelope_xmin, envelope_ymin, envelope_xmax, envelope_ymax],
                                axis=1)

    # sample an image crop
    min_overlaps = [0.1, 0.3, 0.5, 0.7, 0.9, 0.]
    full_crop_bbox = tf.cast(tf.stack([0, 0, orig_w-1, orig_h-1]), tf.float32)
    crop_bboxes = [full_crop_bbox]
    successes = [tf.constant(True, dtype=tf.bool)]
    for i, min_overlap in enumerate(min_overlaps):
      crop_bbox, success = ops.sample_crop_bbox(orig_size, envelope_bboxes,
        overlap_mode=FLAGS.sampling_overlap_mode, min_overlap=min_overlap,
        aspect_ratio_range=[0.5, 2.0], scale_ratio_range=[0.3, 1.0],
        max_trials=50, name=('sampler_%d' % i))
      crop_bboxes.append(crop_bbox)
      successes.append(success)
    crop_bboxes = tf.stack(crop_bboxes)
    successes = tf.stack(successes)
    # random select a valid crop_bbox
    crop_bboxes = tf.boolean_mask(crop_bboxes, successes)
    n_success = tf.shape(crop_bboxes)[0]
    random_crop_index = tf.random_uniform([],
        minval=0, maxval=n_success, dtype=tf.int32)
    crop_bbox = tf.slice(crop_bboxes,
        tf.stack([random_crop_index, 0]), [1, -1])[0,:] # => [4]

    # FIXME: experimental code
    if FLAGS.no_random_crop:
      crop_bbox = full_crop_bbox

    # slice begins and sizes
    slice_xmin = tf.cast(tf.round(crop_bbox[0]), tf.int32)
    slice_ymin = tf.cast(tf.round(crop_bbox[1]), tf.int32)
    slice_xmax = tf.cast(tf.round(crop_bbox[2]), tf.int32)
    slice_ymax = tf.cast(tf.round(crop_bbox[3]), tf.int32)
    slice_width = slice_xmax - slice_xmin + 1
    slice_height = slice_ymax - slice_ymin + 1
    slice_begin = tf.stack([slice_ymin, slice_xmin, 0])
    slice_size = tf.stack([slice_height, slice_width, -1])

    # crop image
    cropped_image = tf.slice(streams['image'], slice_begin, slice_size)

    # resize image with a random interpolation method
    interp_methods = [tf.image.resize_area,
                      tf.image.resize_bicubic,
                      tf.image.resize_bilinear,
                      tf.image.resize_nearest_neighbor]
    n_interp_method = len(interp_methods)
    pred_method_pairs = []
    interp_method_idx = tf.random_uniform(shape=[],
        minval=0, maxval=n_interp_method, dtype=tf.int64)
    resize_size = tf.stack([FLAGS.image_height, FLAGS.image_width])
    for i, interp_method in enumerate(interp_methods):
      pred_method_pairs.append((
          tf.equal(interp_method_idx, i),
          lambda: interp_method(tf.expand_dims(cropped_image, [0]),
                                resize_size)))
    default_resized_image = tf.image.resize_bilinear(
        tf.expand_dims(cropped_image, [0]), resize_size)
    resized_image = tf.case(pred_method_pairs,
                            lambda: default_resized_image, # should never be called
                            exclusive=False)
    resized_image = tf.squeeze(resized_image, [0])

    # image adjustments
    pass

    # project polygons
    projected_polygons, valid_mask = ops.project_polygons(
        word_polygons, crop_bbox, resize_size)
    valid_polygons = tf.boolean_mask(projected_polygons, valid_mask)
    valid_count = tf.shape(valid_polygons)[0]

    # convert polygons to rboxes
    rboxes = ops.polygons_to_rboxes(valid_polygons)

    # clip rboxes by boundaries
    boundary_bbox = tf.cast(tf.stack([0, 0, FLAGS.image_height, FLAGS.image_width]), tf.float32)
    clipped_rboxes = ops.clip_rboxes(rboxes, boundary_bbox)

    if FLAGS.image_channel_order == 'BGR':
      # convert from RGB to BGR
      normed_image = tf.reverse(resized_image, [2])
      # subtract mean
      normed_image = normed_image - IMAGE_BGR_MEAN
    else:
      raise 'Unknown channel order: ' + FLAGS.image_channel_order

    # convert data format
    normed_image.set_shape([FLAGS.image_height, FLAGS.image_width, 3])

    # pad groundtruth to fixed size
    pad_size = FLAGS.max_num_gt - valid_count
    padded_rboxes = tf.pad(clipped_rboxes,
        tf.stack([tf.stack([0, pad_size]), [0, 0]]), mode='CONSTANT')
    padded_rboxes.set_shape([FLAGS.max_num_gt, RBOX_DIM])

  pstreams = {'image': normed_image,
              'image_name': streams['image_name'],
              'rboxes': padded_rboxes,
              'count': valid_count,
              'orig_size': orig_size}
  return pstreams


def test_preprocess(streams, scope=None):
  """
  Preprocess test images and groundtruths
  ARGS
    streams: input data streams
  RETURN
    pstreams: preprocessed data streams
  """
  with tf.variable_scope(scope or 'test_preprocess'):
    # normalize ground truth
    word_polygons = streams['word_polygons']
    image_shape = tf.shape(streams['image'])
    orig_h, orig_w = image_shape[0], image_shape[1]
    orig_size = tf.stack([orig_h, orig_w])
    full_crop_bbox = tf.cast(tf.stack([0, 0, orig_w-1, orig_h-1]), tf.float32)

    # resize image
    if FLAGS.test_resize_method == 'fixed':
      resize_size = tf.stack([FLAGS.image_height, FLAGS.image_width])
    elif FLAGS.test_resize_method == 'dynamic':
      # widths and heights must be dividable by resize_step
      longer_side = tf.minimum(orig_h, orig_w) # FIXME
      resize_scale = tf.truediv(FLAGS.resize_longer_side, longer_side)
      resize_h = tf.cast(orig_h, tf.float64) * resize_scale
      resize_w = tf.cast(orig_w, tf.float64) * resize_scale
      # round to the nearest number dividable by resize_step
      resize_step = tf.cast(FLAGS.resize_step, tf.float64)
      resize_h = tf.cast(
          tf.round(tf.truediv(tf.cast(resize_h, tf.float64), resize_step)) * resize_step,
          dtype=tf.int32)
      resize_w = tf.cast(
          tf.round(tf.truediv(tf.cast(resize_w, tf.float64), resize_step)) * resize_step,
          dtype=tf.int32)
      resize_size = tf.stack([resize_h, resize_w])
    else:
      raise 'Unknown resize method: {}'.format(FLAGS.test_resize_method)
    resized_image = tf.image.resize_bilinear(
        tf.expand_dims(streams['image'], [0]), resize_size)
    resized_image = tf.squeeze(resized_image, [0])

    # project polyogns
    projected_polygons, valid_polygons = ops.project_polygons(
        word_polygons, full_crop_bbox, resize_size)
    valid_polygons = tf.boolean_mask(projected_polygons, valid_polygons)

    # convert polygons to rboxes
    rboxes = ops.polygons_to_rboxes(word_polygons)
    valid_count = tf.shape(rboxes)[0]

    # clip rboxes by boundaries
    clipped_rboxes = ops.clip_rboxes(
        rboxes, tf.constant([0., 0., 1., 1.], tf.float32))

    if FLAGS.image_channel_order == 'BGR':
      # convert from RGB to BGR
      normed_image = tf.reverse(resized_image, [2])
      # subtract mean
      normed_image = normed_image - IMAGE_BGR_MEAN
    else:
      raise 'Unknown channel order: ' + FLAGS.image_channel_order

    # pad groundtruth to fixed size
    pad_size = FLAGS.max_num_gt - valid_count
    padded_rboxes = tf.pad(rboxes,
        tf.stack([tf.stack([0, pad_size]), [0, 0]]), mode='CONSTANT')
    padded_rboxes.set_shape([FLAGS.max_num_gt, RBOX_DIM])

  pstreams = {'image': normed_image,
              'image_name': streams['image_name'],
              'image_jpeg': streams['image_jpeg'], # save images for post-processing
              'rboxes': padded_rboxes,
              'count': valid_count,
              'resize_size': resize_size,
              'orig_size': orig_size}
  return pstreams

import os
import sys
import glob
import random
import math
import re
import tensorflow as tf
import numpy as np
import scipy.io as sio
from tqdm import tqdm

# import utils


WORD_POLYGON_DIM = 8

# helper functions
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
def _bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
def _int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
def _float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def read_jpeg_check(image_path, forbid_grayscale=False):
  with open(image_path, 'rb') as f:
    image_jpeg = f.read()
  return image_jpeg
  # import imghdr
  # import numpy as np
  # # check path exists
  # if not os.path.exists(image_path):
  #   print('Image does not exist: {}'.format(image_path))
  #   return None
  # # check file not empty
  # with open(image_path, 'rb') as f:
  #   image_jpeg = f.read()
  # if image_jpeg is None:
  #   print('Image file is empty: {}'.format(image_path))
  #   return None
  # # check image type is jpeg
  # if imghdr.what(image_path) != 'jpeg':
  #   print('Image file is not jpeg: {}'.format(image_path))
  #   return None
  # # check image is decodable
  # image_buf = np.fromstring(image_jpeg, dtype=np.uint8)
  # image = cv2.imdecode(image_buf, cv2.IMREAD_UNCHANGED)
  # if image is None:
  #   print('Failed to decode image: {}'.format(image_path))
  # # check image is not zero-size
  # if image.shape[0] * image.shape[1] == 0:
  #   print('Image has zero size: {}'.format(image_path))
  #   return None
  # # check image is not grayscale
  # if forbid_grayscale:
  #   if image.ndim == 2 or image.shape[2] == 1:
  #     print('Image is gray-scale: {}'.format(image_path))
  #     return None
  # return image_jpeg


def create_synthtext_dataset(save_path, data_root, shuffle=False, n_max=None):
  """
  Create tf records for the VGG SynthText dataset
  ARGS
    save_path: path to save the TF record
    data_root: the root folder for the datasets
    list_name: list file name
    shuffle: bool, whether to shuffle examples
  """
  # if os.path.exists(save_path):
    # print('File already exists: {}'.format(save_path))
    # return

  # load gt.mat
  print('Loading gt.mat ...')
  gt = sio.loadmat(os.path.join(data_root, 'gt.mat'))
  n_samples = gt['wordBB'].shape[1]

  writer = tf.python_io.TFRecordWriter(save_path)
  print('Start writing to %s' % save_path)

  if n_max is not None:
    n_samples = min(n_max, n_samples)

  if shuffle:
    indices = np.random.permutation(n_samples)
  else:
    indices = np.arange(n_samples)

  for i in tqdm(range(n_samples)):
    idx = indices[i]
    image_rel_path = str(gt['imnames'][0, idx][0])
    image_path = os.path.join(data_root, image_rel_path)
    # load image jpeg data
    with open(image_path, 'rb') as f:
      image_jpeg = f.read()
    # word polygons
    word_polygons = gt['wordBB'][0, idx]
    if word_polygons.ndim == 2:
      word_polygons = np.expand_dims(word_polygons, axis=2)
    word_polygons = np.transpose(word_polygons, axes=[2,1,0])
    n_words = word_polygons.shape[0]
    word_polygons_flat = [float(o) for o in word_polygons.flatten()]
    # words
    text = gt['txt'][0, idx]
    words = []
    for text_line in text:
      text_line = str(text_line)
      line_words = ('\n'.join(text_line.split())).split('\n')
      words.extend(line_words)
    # convert to bytes
    words = [o.encode('ascii') for o in words]
    # write an example
    example = tf.train.Example(features=tf.train.Features(feature={
      'image_name': _bytes_feature(image_rel_path.encode('ascii')),
      'image_jpeg': _bytes_feature(image_jpeg),
      'num_words': _int64_feature(n_words),
      'words': _bytes_list_feature(words),
      'word_polygons': _float_list_feature(word_polygons_flat),
      }))
    writer.write(example.SerializeToString())


def create_icdar2015_incidental_dataset(save_path, image_dir, gt_dir=None,
                                        shuffle=False):
  if os.path.exists(save_path):
    print('File already exists: {}'.format(save_path))
    return

  writer = tf.python_io.TFRecordWriter(save_path)
  print('Start writing to %s' % save_path)
  image_paths = glob.glob(os.path.join(image_dir, '*.jpg'))

  if shuffle:
    random.shuffle(image_paths)

  for i, image_path in enumerate(image_paths):
    # read image binaries
    image_fname = os.path.basename(image_path)
    with open(image_path, 'rb') as f:
      image_jpeg = f.read()
    # read groundtruth
    words = []
    word_polygons = []
    if gt_dir is not None:
      gt_fname = 'gt_' + image_fname[:-4] + '.txt'
      gt_path = os.path.join(gt_dir, gt_fname)
      with open(gt_path, 'r') as f:
        lines = [o.decode('utf-8-sig').encode('utf-8').strip() for o in f.readlines()]
        for line in lines:
          splits = line.split(',')
          polygon = [float(int(o)) for o in splits[:8]]
          word = ','.join(splits[8:])
          words.append(word)
          word_polygons.extend(polygon)

    # words are utf-8 encoded
    words = [bytes(o, encoding='utf-8') for o in words]

    example = tf.train.Example(features=tf.train.Features(feature={
      'image_name': _bytes_feature(image_fname),
      'image_jpeg': _bytes_feature(image_jpeg),
      'num_words': _int64_feature(len(words)),
      'words': _bytes_list_feature(words),
      'word_polygons': _float_list_feature(word_polygons),
      }))
    writer.write(example.SerializeToString())

    if i > 0 and i % 100 == 0:
      print('Written %d / %d' % (i, len(image_paths)))


class DatasetCreator(object):
  def __init__(self, save_path):
    self.save_path = save_path
    self.example_indicies = None

  def _read_list(self):
    """
    Read image and groundtruth list.
    RETURN
      `image_paths`: list of image file paths
      `gt_paths`: list of groundtruth file paths
    """
    raise NotImplementedError

  def _read_image_binary(self, image_path):
    return read_jpeg_check(image_path, forbid_grayscale=True)

  def _parse_annotation(self, annot_file_path):
    """
    Parse groundtruth annotations.
    ARGS
      `annot_file_path`: annotation file path
    RETURN
      `annot_dict`: dictionary of groundtruth annotations
    """
    raise NotImplementedError

  def _make_sample(self, image_id, image_binaries, annot_dict):
    """
    Make a protobuf example.
    ARGS
      `image_binaries`: str, image jpeg binaries
      `annot_dict`: dict, annotations
    RETURN
      `example`: protobuf example
    """
    words = annot_dict['words']
    word_polygons = annot_dict['word_polygons']
    
    if image_binaries is None:
      example = None
    else:
      example = tf.train.Example(features=tf.train.Features(feature={
        'image_name': _bytes_feature(bytes(image_id, encoding='ascii')),
        'image_jpeg': _bytes_feature(image_binaries),
        'num_words': _int64_feature(len(words)),
        'words': _bytes_list_feature(words),
        'word_polygons': _float_list_feature(word_polygons),
        }))
    return example

  def _create_next_sample(self):
    # initialize index
    if not hasattr(self, 'indices'):
      if self.shuffle:
        self.indices = np.random.permutation(self.n_samples)
      else:
        self.indices = np.arange(self.n_samples)
      self.index = 0

    # create the next sample if it's valid
    example = None
    if self.index < self.n_samples:
      image_path = self.image_paths[self.index]
      gt_path = self.gt_paths[self.index] if self.gt_paths is not None else None
      image_id, _ = os.path.splitext(os.path.basename(image_path))
      image_jpeg = self._read_image_binary(image_path)
      if image_jpeg is not None:
        annot_dict = self._parse_annotation(gt_path)
        example = self._make_sample(image_id, image_jpeg, annot_dict)
      else:
        example = None
      self.index += 1

    return example

  def create(self):
    self._read_list()
    print('Start creating dataset with {} examples. Output path: {}'.format(
          self.n_samples, self.save_path))
    writer = tf.python_io.TFRecordWriter(self.save_path)
    count = 0
    for i in range(self.n_samples):
      example = self._create_next_sample()
      if example is not None:
        writer.write(example.SerializeToString())
        count += 1
      if i > 0 and i % 100 == 0:
        print('Progress %d / %d' % (i, self.n_samples))
    print('Done creating %d samples' % count)


class DatasetCreator_Icdar2015Incidental(DatasetCreator):
  def __init__(self, save_path, data_root, training=True, shuffle=True):
    self.save_path = save_path
    self.data_root = data_root
    self.training = training
    self.shuffle = shuffle

  def _read_list(self):
    if self.training:
      image_dir = os.path.join(self.data_root, 'ch4_training_images')
      gt_dir = os.path.join(self.data_root, 'ch4_training_localization_transcription_gt')
    else:
      image_dir = os.path.join(self.data_root, 'ch4_test_images')
      gt_dir = None

    self.image_paths = glob.glob(os.path.join(image_dir, '*.jpg'))
    if self.shuffle:
      random.shuffle(self.image_paths)
    if gt_dir is not None:
      self.gt_paths = [os.path.join(gt_dir, 'gt_{}.txt'.format(
          os.path.basename(o)[:-4])) for o in self.image_paths]
    else:
      self.gt_paths = None

    self.n_samples = len(self.image_paths)

  def _parse_annotation(self, gt_path):
    if gt_path is None:
      empty_annot_dict = {
        'words': [],
        'word_polygons': []
      }
      return empty_annot_dict

    with open(gt_path, 'r', encoding='utf-8-sig') as f:
      lines = [o.strip() for o in f.readlines()]
    word_polygons = []
    words = []
    for line in lines:
      splits = line.split(',')
      polygon = [float(int(o)) for o in splits[:8]]
      word = ','.join(splits[8:]) # in case that GT is splitted because it has ',' in it
      word = bytes(word, encoding='utf-8')
      words.append(word)
      word_polygons.extend(polygon)
    annot_dict = {
      'words': words,
      'word_polygons': word_polygons}
    return annot_dict


class DatasetCreator_Icdar2013(DatasetCreator):
  def __init__(self, save_path, data_root, training, shuffle=False):
    self.save_path = save_path
    self.data_root = data_root
    self.training = training
    self.shuffle = shuffle

  def _read_list(self):
    if self.training:
      image_dir = os.path.join(self.data_root, 'Challenge2_Training_Task12_Images')
      gt_dir = os.path.join(self.data_root, 'Challenge2_Training_Task1_GT')
    else:
      image_dir = os.path.join(self.data_root, 'Challenge2_Test_Task12_Images')
      gt_dir = os.path.join(self.data_root, 'Challenge2_Test_Task1_GT')

    # load image and groundtruth file list
    image_paths = glob.glob(os.path.join(image_dir, '*.jpg'))
    if self.shuffle:
      random.shuffle(image_paths)
    gt_paths = []
    for image_path in image_paths:
      image_id, _ = os.path.splitext(os.path.basename(image_path))
      gt_path = os.path.join(gt_dir, 'gt_%s.txt' % image_id)
      gt_paths.append(gt_path)

    self.image_paths = image_paths
    self.gt_paths = gt_paths
    self.n_samples = len(image_paths)

  def _parse_annotation(self, annot_file_path):
    with open(annot_file_path, 'r') as f:
      lines = [o.strip() for o in f.readlines()]
    p = re.compile('(\d+)[,\s]*?(\d+)[,\s]*?(\d+)[,\s]*?(\d+)[,\s]*?"(.*?)"')
    word_polygons = []
    words = []
    for line in lines:
      m = p.match(line)
      xmin = int(m.group(1))
      ymin = int(m.group(2))
      xmax = int(m.group(3))
      ymax = int(m.group(4))
      # convert bounding box to polygon
      word_polygons.extend([xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax])
      word = m.group(5)
      words.append(word)

    # pack annotations into a python dict
    annot_dict = {}
    annot_dict['words'] = words
    annot_dict['word_polygons'] = word_polygons

    return annot_dict


class DatasetCreator_Td500(DatasetCreator):
  def __init__(self, save_path, data_root, subset='train', include_difficult=True):
    self.save_path = save_path
    self.data_root = data_root
    self.subset = subset
    self.shuffle = True # shuffle is fixed in code
    self.include_difficult = include_difficult
    random.seed(123)

  def _read_list(self):
    if self.subset == 'train':
      image_gt_dir = os.path.join(self.data_root, 'train')
      self.image_paths = glob.glob(os.path.join(image_gt_dir, '*.JPG'))
      random.shuffle(self.image_paths)
      self.image_paths = self.image_paths[100:]
    elif self.subset == 'val':
      image_gt_dir = os.path.join(self.data_root, 'train')
      self.image_paths = glob.glob(os.path.join(image_gt_dir, '*.JPG'))
      random.shuffle(self.image_paths)
      self.image_paths = self.image_paths[:100]
    elif self.subset == 'test':
      image_gt_dir = os.path.join(self.data_root, 'test')
      self.image_paths = glob.glob(os.path.join(image_gt_dir, '*.JPG'))
    else:
      raise ValueError('subset = %s' % subset)
      
    self.gt_paths = [o[:-4] + '.gt' for o in self.image_paths]
    self.n_samples = len(self.image_paths)

  def _parse_annotation(self, gt_path):
    with open(gt_path, 'r') as f:
      lines = [o.decode('utf-8-sig').encode('utf-8').strip() for o in f.readlines()]

    words = []
    word_polygons = []
    for line in lines:
      splits = line.split(' ')
      difficult_label = bool(int(splits[1]))
      if self.include_difficult == False and difficult_label == True:
        continue

      x = float(splits[2])
      y = float(splits[3])
      w = float(splits[2])
      h = float(splits[3])
      theta = float(splits[4])
      cx = x + 0.5 * w
      cy = y + 0.5 * h
      x1 = cx - 0.5 * w * math.cos(theta) + 0.5 * h * math.sin(theta)
      y1 = cy - 0.5 * w * math.sin(theta) - 0.5 * h * math.cos(theta)
      x2 = cx + 0.5 * w * math.cos(theta) + 0.5 * h * math.sin(theta)
      y2 = cy + 0.5 * w * math.sin(theta) - 0.5 * h * math.cos(theta)
      x3 = cx + 0.5 * w * math.cos(theta) - 0.5 * h * math.sin(theta)
      y3 = cy + 0.5 * w * math.sin(theta) + 0.5 * h * math.cos(theta)
      x4 = cx - 0.5 * w * math.cos(theta) - 0.5 * h * math.sin(theta)
      y4 = cy - 0.5 * w * math.sin(theta) + 0.5 * h * math.cos(theta)
      word_polygons.extend([x1, y1, x2, y2, x3, y3, x4, y4])
      words.append('') # TD500 has no text annotations

    annot_dict = {
      'words': words,
      'word_polygons': word_polygons}
    return annot_dict


class DatasetCreator_Scut(DatasetCreator):
  def __init__(self, save_path, data_root, shuffle=True):
    self.save_path = save_path
    self.data_root = data_root
    self.shuffle = shuffle

  def _read_list(self):
    image_dir = os.path.join(self.data_root, 'word_img')
    gt_dir = os.path.join(self.data_root, 'word_annotation')
    image_paths = glob.glob(os.path.join(image_dir, '*.jpg'))
    image_paths.extend(glob.glob(os.path.join(image_dir, '*.JPG')))
    gt_paths = []
    for image_path in image_paths:
      image_id, _ = os.path.splitext(os.path.basename(image_path))
      gt_path = os.path.join(gt_dir, '%s.txt' % image_id)
      gt_paths.append(gt_path)
    self.image_paths = image_paths
    self.gt_paths = gt_paths
    self.n_samples = len(self.image_paths)

  def _parse_annotation(self, annot_file_path):
    with open(annot_file_path, 'r') as f:
      lines = [o.strip() for o in f.readlines()]
    word_polygons = []
    words = []
    for line in lines:
      splits = line.split(',')
      assert(len(splits) == 5)
      xmin = int(splits[0])
      ymin = int(splits[1])
      width = int(splits[2])
      height = int(splits[3])
      word = splits[4]
      xmax = xmin + width
      ymax = ymin + height
      # convert bounding box to polygon
      word_polygons.extend([xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax])
      words.append(word)

    # pack annotations into a python dict
    annot_dict = {}
    annot_dict['words'] = words
    annot_dict['word_polygons'] = word_polygons

    return annot_dict


def create_merge_multiple(save_path, creators, shuffle=True):
  n_sample_total = 0
  creator_indices = []
  for i, creator in enumerate(creators):
    creator._read_list()
    n_sample_total += creator.n_samples
    creator_indices.append(np.full((creator.n_samples), i, dtype=np.int))
  creator_indices = np.concatenate(creator_indices)

  if shuffle:
    np.random.shuffle(creator_indices)

  print('Start creating dataset with {} examples. Output path: {}'.format(
        n_sample_total, save_path))
  writer = tf.python_io.TFRecordWriter(save_path)
  count = 0
  for i in range(n_sample_total):
    creator = creators[creator_indices[i]]
    example = creator._create_next_sample()
    if example is not None:
      writer.write(example.SerializeToString())
      count += 1
    if i > 0 and i % 100 == 0:
      print('Progress %d / %d' % (i, n_sample_total))
  print('Done creating %d samples' % count)


if __name__ == '__main__':
  # SynthText all
  # data_root = '/mnt/datasets/scene_text/SynthText/SynthText/'
  # create_synthtext_dataset('../data/synthtext_10k.tf',
  #                          data_root,
  #                          n_max=10000,
  #                          shuffle=True)
  # create_synthtext_dataset('/mnt/datasets/scene_text/SynthText/synthtext_full.tf',
  #                          data_root,
  #                          shuffle=True)

  # ICDAR 2015 incidental
  ic15_data_root = '/mnt/datasets/scene_text/icdar_2015_incidental/'
  creator_ic15_train = DatasetCreator_Icdar2015Incidental(
      '../data/icdar_2015_incidental_train.tf',
      ic15_data_root,
      training=True,
      shuffle=True)
  creator_ic15_test = DatasetCreator_Icdar2015Incidental(
      '../data/icdar_2015_incidental_test.tf',
      ic15_data_root,
      training=False,
      shuffle=False)

  # # ICDAR 2013
  # ic13_root_dir = '/home/ubuntu/data/datasets/scene_text/icdar_2013/'
  # creator_ic13_train = DatasetCreator_Icdar2013('../data/icdar_2013_train.tf',
  #     os.path.join(ic13_root_dir, 'Ch2_Scenet_Text/Text Localization'),
  #     training=True, shuffle=True)
  # creator_ic13_test = DatasetCreator_Icdar2013('../data/icdar_2013_test.tf',
  #     os.path.join(ic13_root_dir, 'Ch2_Scenet_Text/Text Localization'),
  #     training=False, shuffle=False)

  # # TD500
  # td500_root_dir = '/home/ubuntu/data/datasets/scene_text/MSRA-TD500'
  # creator_td500_train = DatasetCreator_Td500('../data/td500_train.tf',
  #   td500_root_dir, subset='train', include_difficult=True)
  # creator_td500_val = DatasetCreator_Td500('../data/td500_val.tf',
  #   td500_root_dir, subset='val', include_difficult=True)
  # creator_td500_test = DatasetCreator_Td500('../data/td500_test.tf',
  #   td500_root_dir, subset='test', include_difficult=True)

  # individual datasets
  creator_ic15_train.create()
  creator_ic15_test.create()
  # creator_ic13_train.create()
  # creator_ic13_test.create()

  # creator_td500_train.create()
  # creator_td500_val.create()
  # creator_td500_test.create()

  pass

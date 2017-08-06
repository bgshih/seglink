import math
import tensorflow as tf
import numpy as np
import argparse
import skimage
import skimage.io, skimage.transform

import model_vgg16

parser = argparse.ArgumentParser(description='')
parser.add_argument('--ckpt_path', default='/tmp/VGG_ILSVRC_16_layers.ckpt',
                    help='Checkpoint save path.')
args = parser.parse_args()


def test_classify_image():

  def load_image_and_preprocess(fname):
    image = skimage.io.imread(fname)
    assert(image.ndim == 3)
    # scale image so that the shorter side is 224
    image_h, image_w = image.shape[0], image.shape[1]
    shorter_side = min(image_h, image_w)
    scale = 224.0 / shorter_side
    image = skimage.transform.rescale(image, scale)
    image_h, image_w = image.shape[0], image.shape[1]
    # center crop
    crop_x = (image_w - 224) / 2
    crop_y = (image_h - 224) / 2
    image = image[crop_y:crop_y+224,crop_x:crop_x+224,:]
    # RGB -> BGR
    image = image[:,:,::-1]
    image *= 255.0
    # subtract mean
    image_mean = np.array([103.939, 116.779, 123.68])
    image -= np.reshape(image_mean, [1,1,3])
    # HWC -> CHW
    image = np.transpose(image, [2,0,1])
    # add an extra dim
    image = np.expand_dims(image, axis=0)
    return image

  # construct model graph
  vgg16 = model_vgg16.Vgg16Model()
  input_images = tf.placeholder(tf.float32, shape=[1,3,224,224])
  prob = vgg16(input_images, scope='Vgg16')

  # load imge
  image = load_image_and_preprocess('cat.png')

  with tf.Session() as session:
    session.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    print('Restoring model')
    saver.restore(session, args.ckpt_path)
    print('Model restored')

    session_outputs = session.run([prob], {input_images.name: image})
    prob_value = session_outputs[0]
    top_5_indices = np.argsort(prob_value[0])[-5:][::-1]
    synsets = [line.rstrip('\n') for line in open('synset.txt')]
    print('Top 5 predictions:')
    for i in range(5):
      idx = top_5_indices[i]
      print('%f  %s' % (prob_value[0,idx], synsets[idx]))


if __name__ == '__main__':
  test_classify_image()

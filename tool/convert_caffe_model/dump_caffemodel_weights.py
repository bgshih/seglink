import numpy as np
import joblib
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--caffe_root', help='Caffe root directory.')
parser.add_argument('--prototxt_path', help='Model prototxt path.')
parser.add_argument('--caffemodel_path', help='Caffe model weights file (.caffemodel) path.')
parser.add_argument('--caffe_weights_path', default='./VGG_ILSVRC_16_layers_weights.pkl',
                    help='VGG16 weights dump path.')
args = parser.parse_args()

os.environ["GLOG_minloglevel"] = "2"
sys.path.append(os.path.join(args.caffe_root, 'python'))
import caffe


def dump_caffemodel_weights():
  net = caffe.Net(args.prototxt_path, args.caffemodel_path, caffe.TEST)
  weights = {}
  n_layers = len(net.layers)
  for i in range(n_layers):
    layer_name = net._layer_names[i]
    layer = net.layers[i]
    layer_blobs = [o.data for o in layer.blobs]
    weights[layer_name] = layer_blobs
  joblib.dump(weights, args.caffe_weights_path)


if __name__ == '__main__':
  dump_caffemodel_weights()

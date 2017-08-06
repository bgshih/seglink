#!/usr/bin/python3
import os
from os.path import join, exists, abspath
import sys
import json
import glob

SRC_DIR = abspath('./seglink')
SHARED_LIBRARY_NAME = 'libseglink.so'


def build_op():
  build_dir = join(SRC_DIR, 'cpp/build')
  if not exists(build_dir):
    os.mkdir(build_dir)
  os.chdir(build_dir)
  if not exists('Makefile'):
    os.system('cmake -DCMAKE_BUILD_TYPE=Release ..')
  os.system('make -j16')
  os.system('cp %s %s' % (SHARED_LIBRARY_NAME, join(SRC_DIR, SHARED_LIBRARY_NAME)))
  print('Building complete')


def clean_op():
  build_dir = join(SRC_DIR, 'cpp/build')
  print('Deleting recursively: %s' % build_dir)
  os.system('rm -rI %s' % build_dir)
  os.system('rm %s' % join(SRC_DIR, SHARED_LIBRARY_NAME))


def clear():
  if len(sys.argv) != 3:
    print('Usage ./manage.py clear <exp_dir>')
    return
  exp_dir = sys.argv[2]
  # get files to delete
  all_files = glob.glob(exp_dir + "/*")
  files_to_delete = [o for o in all_files if not o.endswith('.json')]
  # prompt
  print('Files to delete:')
  print('\n'.join(files_to_delete))
  print('Conitnue? (y/n)')
  user_input = input()
  if user_input != 'y':
    files_to_delete = []
  for fpath in files_to_delete:
    os.remove(fpath)
  print('Deleted {} files'.format(len(files_to_delete)))


def run_tf_program_with_json_config(program):
  if program == 'train':
    script = 'solver.py'
  elif program == 'test':
    script = 'evaluate.py'
  
  if len(sys.argv) != 4:
    print('Usage ./manage.py {} <exp_dir> <config_name>'.format(program))
    return
  exp_dir = abspath(sys.argv[2])
  config_name = sys.argv[3]
  if not exists(exp_dir):
    print('Directory not found: {0}'.format(exp_dir))
    return
  config_path = join(exp_dir, config_name + '.json')
  if not exists(config_path):
    print('Configuration file not found: {}'.format(config_path))
    return
  with open(config_path, 'r') as f:
    config = json.load(f)

  # construct command
  cmd = 'python3 {}'.format(script)
  if 'cuda_devices' in config:
    cuda_devices = config.pop('cuda_devices')
    n_gpu = len(cuda_devices.split(','))
    cmd = 'CUDA_VISIBLE_DEVICES={} '.format(cuda_devices) + cmd
  else:
    n_gpu = 1
  cmd += ' --log_dir {0}'.format(exp_dir)
  cmd += ' --n_gpu {0}'.format(n_gpu)
  for key, value in config.items():
    if isinstance(value, list):
      value_str = ','.join(['{}'.format(o) for o in value])
    else:
      value_str = value
    cmd += (' --{0} {1}'.format(key, value_str))
  print(cmd)

  # cmd must be run from the SRC_DIR
  os.chdir(SRC_DIR)
  os.system(cmd)


def train():
  run_tf_program_with_json_config('train')


def test():
  run_tf_program_with_json_config('test')


def start_tb():
  """
  Start Tensorboard at port 8000.
  """
  cmd = 'tensorboard --logdir ./exp --port 8000'
  os.system(cmd)


def upload_logs():
  """
  Upload experiment logs and events to remote server.
  """
  pass


if __name__ == '__main__':
  if len(sys.argv) < 2:
    print('Usage: python3 manage.py <function-name>')
  else:
    fn_name = sys.argv[1]
    eval(fn_name + "()")

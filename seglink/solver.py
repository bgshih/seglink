import os
import sys
import re
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.client import timeline
import logging
import numpy as np
from tqdm import trange

import model
import data
import utils

FLAGS = tf.app.flags.FLAGS
# logging
tf.app.flags.DEFINE_string('log_dir', '', 'Directory for saving checkpoints and log files')
tf.app.flags.DEFINE_string('log_prefix', '', 'Log file name prefix')
# training
tf.app.flags.DEFINE_string('resume', 'vgg16', 'Training from loading VGG16 parameters ("vgg16"), resume a checkpoint ("resume"), or finetune a pretrained model ("finetune")')
tf.app.flags.DEFINE_string('vgg16_model', '../data/VGG_ILSVRC_16_layers_ssd.ckpt', 'The pretrained VGG16 model checkpoint')
tf.app.flags.DEFINE_string('finetune_model', '', 'Finetuning model path')
tf.app.flags.DEFINE_string('train_datasets', '', 'Training datasets file names separated by semicolons')
tf.app.flags.DEFINE_string('weight_init_method', 'xavier', 'Weight initialization method')
tf.app.flags.DEFINE_integer('train_batch_size', 32, 'Training batch size')
tf.app.flags.DEFINE_integer('n_gpu', 1, 'Number of GPUs used in training')
tf.app.flags.DEFINE_float('hard_neg_ratio', 3.0, 'Ratio of hard negatives to positives')
tf.app.flags.DEFINE_integer('no_random_crop', 0, 'In data augmentation, do not crop image, i.e. use full images')
# optimizer
tf.app.flags.DEFINE_string('optimizer', 'sgd', 'Optimization algorithm')
tf.app.flags.DEFINE_float('base_lr', 1e-3, 'Base learning rate')
tf.app.flags.DEFINE_float('momentum', 0.9, 'SGD momentum')
tf.app.flags.DEFINE_float('weight_decay', 5e-4, 'SGD weight decay')
tf.app.flags.DEFINE_integer('max_steps', 60000, 'Maximum number of iterations.')
tf.app.flags.DEFINE_string('lr_policy', 'staircase', 'Learning rate decaying policy')
tf.app.flags.DEFINE_string('lr_breakpoints', '', 'Comma-separated breakpoints of learning rate decay')
tf.app.flags.DEFINE_string('lr_decays', '', 'Comma-separated decay values for every breakpoint')
tf.app.flags.DEFINE_integer('profiling', 0, 'Do profiling during training (profiling could slow down training significantly)')
tf.app.flags.DEFINE_integer('profiling_step', 21, 'Run profiling once at this step')
tf.app.flags.DEFINE_string('profiling_report', 'timeline.json', 'Profiling report filename')
# summaries and checkpoints
tf.app.flags.DEFINE_integer('brief_summary_period', 10, 'Period for brief summaries')
tf.app.flags.DEFINE_integer('detailed_summary_period', 200, 'Period for detailed summaries')
tf.app.flags.DEFINE_integer('checkpoint_period', 5000, 'Period for saving checkpoints')


class Solver:
  def __init__(self):
    self.detector = model.SegLinkDetector()

    # global TF variables
    with tf.device('/cpu:0'):
      self.global_step = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int64)
      tf.summary.scalar('global_step', self.global_step, collections=['brief'])

    # setup training graphs and summaries
    self._setup_train_net_multigpu()

    # if true the training process will be terminated in the next iteration
    self.should_stop = False

  def _tower_loss(self, train_batch):
    images = train_batch['image']

    # model and loss function
    outputs = self.detector.build_model(images)
    image_size = tf.shape(images)[1:]
    tower_loss = self.detector.build_loss(outputs, train_batch['rboxes'],
                                          train_batch['count'], image_size)
    return tower_loss

  def _average_gradients(self, tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
      grads = []
      for g, _ in grad_and_vars:
        expanded_g = tf.expand_dims(g, 0)
        grads.append(expanded_g)

      grad = tf.concat(grads, axis=0)
      grad = tf.reduce_mean(grad, 0)

      v = grad_and_vars[0][1]
      grad_and_var = (grad, v)
      average_grads.append(grad_and_var)
    return average_grads

  def _setup_train_net_multigpu(self):
    with tf.device('/cpu:0'):
      # learning rate decay
      with tf.name_scope('lr_decay'):
        if FLAGS.lr_policy == 'staircase':
          # decayed learning rate
          lr_breakpoints = [int(o) for o in FLAGS.lr_breakpoints.split(',')]
          lr_decays = [float(o) for o in FLAGS.lr_decays.split(',')]
          assert(len(lr_breakpoints) == len(lr_decays))
          pred_fn_pairs = []
          for lr_decay, lr_breakpoint in zip(lr_decays, lr_breakpoints):
            fn = (lambda o: lambda: tf.constant(o, tf.float32))(lr_decay)
            pred_fn_pairs.append((tf.less(self.global_step, lr_breakpoint), fn))
          lr_decay = tf.case(pred_fn_pairs, default=(lambda: tf.constant(1.0)))
        else:
          logging.error('Unkonw lr_policy: {}'.format(FLAGS.lr_policy))
          sys.exit(1)

        self.current_lr = lr_decay * FLAGS.base_lr
        tf.summary.scalar('lr', self.current_lr, collections=['brief'])

      # input data
      # batch_size = int(FLAGS.train_batch_size / FLAGS.n_gpu)
      with tf.name_scope('input_data'):
        batch_size = FLAGS.train_batch_size
        train_datasets = FLAGS.train_datasets.split(';')
        train_pstreams_list = []
        for i, dataset in enumerate(train_datasets):
          if not os.path.exists(dataset):
            logging.critical('Could not find dataset {}'.format(dataset))
            sys.exit(1)
          logging.info('Added training dataset #{}: {}'.format(i, dataset))
          train_streams = data.input_stream(dataset)
          train_pstreams = data.train_preprocess(train_streams)
          train_pstreams_list.append(train_pstreams)
        capacity = batch_size * 50
        min_after_dequeue = batch_size * 3
        train_batch = tf.train.shuffle_batch_join(train_pstreams_list,
                                                  batch_size,
                                                  capacity=capacity,
                                                  min_after_dequeue=min_after_dequeue)
        logging.info('Batch size {}; capacity: {}; min_after_dequeue: {}'.format(batch_size, capacity, min_after_dequeue))

        # split batch into sub-batches for each GPU
        sub_batch_size = int(FLAGS.train_batch_size / FLAGS.n_gpu)
        logging.info('Batch size is {} on each of the {} GPUs'.format(sub_batch_size, FLAGS.n_gpu))
        sub_batches = []
        for i in range(FLAGS.n_gpu):
          sub_batch = {}
          for k, v in train_batch.items():
            sub_batch[k] = v[i*sub_batch_size : (i+1)*sub_batch_size]
          sub_batches.append(sub_batch)

      if FLAGS.optimizer == 'sgd':
        optimizer = tf.train.MomentumOptimizer(self.current_lr, FLAGS.momentum)
        logging.info('Using SGD optimizer. Momentum={}'.format(FLAGS.momentum))
      elif FLAGS.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(self.current_lr)
        logging.info('Using ADAM optimizer.')
      elif FLAGS.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(self.current_lr)
        logging.info('Using RMSProp optimizer.')
      else:
        logging.critical('Unsupported optimizer {}'.format(FLAGS.optimizer))
        sys.exit(1)

      # construct towers
      tower_gradients = []
      tower_losses = []
      for i in range(FLAGS.n_gpu):
        logging.info('Setting up tower %d' % i)
        with tf.device('/gpu:%d' % i):
          # variables are shared
          with tf.variable_scope(tf.get_variable_scope(), reuse=(i > 0)):
            with tf.name_scope('tower_%d' % i):
              loss = self._tower_loss(sub_batches[i])
              # tf.get_variable_scope().reuse_variables()
              gradients = optimizer.compute_gradients(loss)
              tower_gradients.append(gradients)
              tower_losses.append(loss)

      # average loss and gradients
      self.loss = tf.truediv(tf.add_n(tower_losses), float(len(tower_losses)),
                             name='average_tower_loss')
      tf.summary.scalar('total_loss', self.loss, collections=['brief'])
      with tf.name_scope('average_gradients'):
        grads = self._average_gradients(tower_gradients)

      # update variables
      with tf.variable_scope('optimizer'):
        self.train_op = optimizer.apply_gradients(grads, global_step=self.global_step)

      # setup summaries
      for var in tf.all_variables():
        # remove the illegal ":x" part from the variable name
        summary_name = 'parameters/' + var.name.split(':')[0]
        tf.summary.histogram(summary_name, var, collections=['detailed'])
      
      self.brief_summary_op = tf.summary.merge_all(key='brief')
      self.detailed_summary_op = tf.summary.merge_all(key='detailed')
    

  def train_and_eval(self):
    # register handler for ctrl-c
    self._register_signal_handler()

    sess_config = tf.ConfigProto(log_device_placement=False,
                                 allow_soft_placement=True)
    with tf.Session(config=sess_config) as sess:
      # create summary writer and saver
      summary_writer = tf.summary.FileWriter(FLAGS.log_dir, graph=sess.graph)
      saver = tf.train.Saver(max_to_keep=20)

      # resume training, load pretrained model, or start training from scratch
      if FLAGS.resume == 'resume':
        latest_ckpt_path = tf.train.latest_checkpoint(FLAGS.log_dir)
        if latest_ckpt_path is None:
          logging.error('Failed to find the latest checkpoint from {}'.format(FLAGS.log_dir))
          sys.exit(1)
        else:
          model_loader = tf.train.Saver()
          model_loader.restore(sess, latest_ckpt_path)
          logging.info('Resuming checkpoint %s' % latest_ckpt_path)
      elif FLAGS.resume == 'finetune':
        ckpt_path = FLAGS.finetune_model
        model_loader = tf.train.Saver()
        model_loader.restore(sess, ckpt_path)
        tf.assign(self.global_step, 0).eval() # reset global_step
        logging.info('Loaded pretrained model from %s' % ckpt_path)
      elif FLAGS.resume == 'vgg16':
        logging.info('Initializing model')
        sess.run(tf.global_variables_initializer())
        vgg16_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='ssd/vgg16/')
        pretrained_loader = tf.train.Saver(var_list=vgg16_vars)
        pretrained_loader.restore(sess, FLAGS.vgg16_model)
        logging.info('VGG16 parameters loaded from {}'.format(FLAGS.vgg16_model))

      # checkpoint save path
      ckpt_save_path = os.path.join(FLAGS.log_dir, 'checkpoint')

      # start data loader threads
      with slim.queues.QueueRunners(sess):
        # training loop
        logging.info('Training loop started')

        start_step = self.global_step.eval()
        tqdm_range = trange(start_step, FLAGS.max_steps)
        for step in tqdm_range:
          need_brief_summary = step % FLAGS.brief_summary_period == 0
          need_detailed_summary = step % FLAGS.detailed_summary_period == 0
          need_save_checkpoint = step > 0 and step % FLAGS.checkpoint_period == 0
          need_profiling = FLAGS.profiling and step == FLAGS.profiling_step

          # construct train fetches
          train_fetches = {}
          train_fetches['train_op'] = self.train_op
          if need_brief_summary:
            train_fetches['loss'] = self.loss
            train_fetches['lr'] = self.current_lr
            train_fetches['brief_summary'] = self.brief_summary_op
          if need_detailed_summary:
            train_fetches['detailed_summary'] = self.detailed_summary_op
          if need_profiling:
            run_metadata = tf.RunMetadata()
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
          else:
            run_metadata = None
            run_options = None

          # run session
          time_start = time.time()
          outputs = sess.run(train_fetches,
                             options=run_options,
                             run_metadata=run_metadata)

          # brief summaries and progress display
          if need_brief_summary:
            summary_str = outputs['brief_summary']
            summary_writer.add_summary(summary_str, step)
            loss = outputs['loss']
            lr = outputs['lr']
            loss_str = '%.2f' % loss
            lr_str = '%.2e' % lr
            tqdm_range.set_postfix(loss=loss_str, lr=lr_str)

            # terminated due to divergence or nan loss
            if np.isnan(loss) or loss > 1e3:
              logging.critical('Training diverges. Terminating.')
              sys.exit(1)

          # detailed summaries
          if need_detailed_summary:
            summary_str = outputs['detailed_summary']
            summary_writer.add_summary(summary_str, step)

          # save profiling results
          if need_profiling:
            # write to timeline
            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            tl_write_path = os.path.join(FLAGS.log_dir, FLAGS.profiling_report)
            with open(tl_write_path, 'w') as f:
              f.write(ctf)
            logging.info('Profiling results written to {}'.format(tl_write_path))
            # write to summary
            summary_writer.add_run_metadata(run_metadata, 'step_{}'.format(step))
            logging.info('Profiling results written')

          # save checkpoint
          if need_save_checkpoint:
            saver.save(sess, ckpt_save_path, global_step=self.global_step)
            logging.info('Checkpoint saved to %s' % ckpt_save_path)

          # ctrl-c
          if self.should_stop == True:
            break

        logging.info('Training loop ended')
        saver.save(sess, ckpt_save_path, global_step=self.global_step.eval())
        logging.info('Checkpoint saved to %s' % ckpt_save_path)

  def _handle_ctrl_c(self, signal, frame):
    logging.info('Ctrl-C pressed, terminating training process')
    self.should_stop = True

  def _register_signal_handler(self):
    import signal
    signal.signal(signal.SIGINT, self._handle_ctrl_c)


if __name__ == '__main__':
  # create logging dir if not existed
  utils.mkdir_if_not_exist(FLAGS.log_dir)
  # set up logging
  log_file_name = FLAGS.log_prefix + time.strftime('%Y%m%d_%H%M%S') + '.log'
  log_file_path = os.path.join(FLAGS.log_dir, log_file_name)
  utils.setup_logger(log_file_path)
  utils.log_flags(FLAGS)
  utils.log_git_version()
  # run solver
  solver = Solver()
  solver.train_and_eval()

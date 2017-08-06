import tensorflow as tf

# basic
tf.app.flags.DEFINE_string('action', 'train', 'Action to take')
tf.app.flags.DEFINE_string('working_dir', '', 'Directory for saving checkpoints and log files')
tf.app.flags.DEFINE_string('log_file_prefix', 'fctd_', 'Prefix of logging file name')

# FCTD model
tf.app.flags.DEFINE_integer('pos_label', 1, 'Label for the background class')
tf.app.flags.DEFINE_integer('neg_label', 0, 'Label for the background class')
tf.app.flags.DEFINE_float('fctd_min_scale', 0.1, 'Minimum region size')
tf.app.flags.DEFINE_float('fctd_max_scale', 0.95, 'Maximum region size')
tf.app.flags.DEFINE_float('pos_scale_diff_threshold', 1.7, '')
tf.app.flags.DEFINE_float('neg_scale_diff_threshold', 2.0, '')
tf.app.flags.DEFINE_integer('fctd_n_scale', 6, 'Number of region scales')
tf.app.flags.DEFINE_integer('n_local_links', 8, 'Number of links of a grid node')
tf.app.flags.DEFINE_integer('n_cross_links', 4, 'Number of cross-layer links on each node')
tf.app.flags.DEFINE_string('link_clf_mode', 'softmax', 'Mode of classifying local links. Can be softmax or sigmoid')

# testing
tf.app.flags.DEFINE_integer('test_period', 5000, 'Period of on-the-fly testing')
tf.app.flags.DEFINE_string('test_model_path', '', 'Test model path')
tf.app.flags.DEFINE_string('test_record_path', '', 'Test tf-records path')
tf.app.flags.DEFINE_integer('test_batch_size', 32, 'Test batch size')
tf.app.flags.DEFINE_integer('num_test', 500, 'Number of test images')
tf.app.flags.DEFINE_float('node_threshold', 0.5, 'Confidence threshold for nodes')
tf.app.flags.DEFINE_float('link_threshold', 0.5, 'Confidence threshold for links')
tf.app.flags.DEFINE_integer('nms_top_k', 400, 'Apply NMS only to examples with top-k scores on each class')
tf.app.flags.DEFINE_integer('keep_top_k', 200, 'Keep examples with top-k scores after NMS')
tf.app.flags.DEFINE_integer('save_visualization', 0, 'Save visualization results')
tf.app.flags.DEFINE_string('result_format', 'icdar_2015_inc', 'Result file format')
tf.app.flags.DEFINE_float('bbox_scale_factor', 0, 'Bounding box scale trick')

# summaries and checkpoints
tf.app.flags.DEFINE_integer('brief_summary_period', 10, 'Period for brief summaries')
tf.app.flags.DEFINE_integer('detailed_summary_period', 500, 'Period for detailed summaries')
tf.app.flags.DEFINE_integer('checkpoint_period', 5000, 'Period for saving checkpoints')

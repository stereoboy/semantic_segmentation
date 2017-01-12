import numpy as np
import tensorflow as tf
import sys
import scipy.misc
import os
import random
from PIL import Image
from datetime import datetime, date, time
import getopt
import common

###########################################################################
#
# main reference: https://github.com/shelhamer/fcn.berkeleyvision.org.git
#
###########################################################################

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("channel", "3", "channel size")
tf.flags.DEFINE_string("directory", "./", "directory for TFRecords")
tf.flags.DEFINE_string("records", "train_VOC2012", "TFRecords filename")
tf.flags.DEFINE_integer("max_epoch", "10", "maximum iterations for training")
tf.flags.DEFINE_integer("max_itrs", "10000", "maximum iterations for training")
tf.flags.DEFINE_integer("img_size", "500", "sample image size")
tf.flags.DEFINE_string("save_dir", "fcn_voc2012_checkpoints", "dir for checkpoints")
tf.flags.DEFINE_integer("nrclass", "21", "size of class")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Momentum Optimizer")
tf.flags.DEFINE_float("beta1", "0.5", "beta1 for Adam optimizer")
tf.flags.DEFINE_float("momentum", "0.9", "momentum for Momentum Optimizer")
tf.flags.DEFINE_float("weight_decay", "0.0016", "Learning rate for Momentum Optimizer")
tf.flags.DEFINE_integer("num_threads", "6", "max thread number")
tf.flags.DEFINE_float("eps", "1e-5", "epsilon for various operation")

def idx2onehot(tensor):

  shape = tf.shape(tensor)
  onehot = tf.constant(np.eye(FLAGS.nrclass, dtype=np.float32))
  ret = tf.nn.embedding_lookup(onehot, tensor)
  #ret = tf.reshape(ret, shape=tf.pack([shape[0], shape[1], shape[2], FLAGS.nrclass]))

  return ret

def onehot2idx(tensor):

  tensor = tf.transpose(tensor, perm=[0, 2, 3, 1])
  tensor = tf.cast(tensor, tf.int32)
  ret = tf.argmax(tensor, axis=[0 ,1 ,2])

  return ret

palette = None
def save_label_png(filepath, array):

    print "test:", test
    img = Image.fromarray(array, mode='P')
    img.putpalette(palette.reshape(-1))
    img.save("./test.png","png")

def batch_norm_layer(tensors ,scope_bn, reuse):
  out = tf.contrib.layers.batch_norm(tensors, decay=0.9, center=True, scale=True,
      epsilon=FLAGS.eps,
      updates_collections=None,
      is_training=True,
      reuse=reuse,
      trainable=True,
      scope=scope_bn, data_format='NCHW')
  return out

def conv_relu(tensor, W, B):
  conved = tf.nn.conv2d(tensor, W, strides=[1, 1, 1, 1], padding='SAME', data_format='NCHW')
  biased = tf.nn.bias_add(conved, B, data_format='NCHW')
  relued = tf.nn.relu(biased)

  return relued

def init_VGG16(pretrained):
  def load_weight(name):
    print pretrained[name]['weights'].shape
    return tf.constant_initializer(value=pretrained[name]['weights'])

  def load_bias(name):
    print pretrained[name]['biases'].shape
    return tf.constant_initializer(value=pretrained[name]['biases'])

  # initialize with Pretrained model
  Ws = {
      "1_1":tf.get_variable('conv1_1', shape = [3, 3, FLAGS.channel, 64], initializer=load_weight('conv1_1')),
      "1_2":tf.get_variable('conv1_2', shape = [3, 3, 64, 64], initializer=load_weight('conv1_2')),

      "2_1":tf.get_variable('conv2_1', shape = [3, 3, 64, 128], initializer=load_weight('conv2_1')),
      "2_2":tf.get_variable('conv2_2', shape = [3, 3, 128, 128], initializer=load_weight('conv2_2')),

      "3_1":tf.get_variable('conv3_1', shape = [3, 3, 128, 256], initializer=load_weight('conv3_1')),
      "3_2":tf.get_variable('conv3_2', shape = [3, 3, 256, 256], initializer=load_weight('conv3_2')),
      "3_3":tf.get_variable('conv3_3', shape = [3, 3, 256, 256], initializer=load_weight('conv3_3')),

      "4_1":tf.get_variable('conv4_1', shape = [3, 3, 256, 512], initializer=load_weight('conv4_1')),
      "4_2":tf.get_variable('conv4_2', shape = [3, 3, 512, 512], initializer=load_weight('conv4_2')),
      "4_3":tf.get_variable('conv4_3', shape = [3, 3, 512, 512], initializer=load_weight('conv4_3')),

      "5_1":tf.get_variable('conv5_1', shape = [3, 3, 512, 512], initializer=load_weight('conv5_1')),
      "5_2":tf.get_variable('conv5_2', shape = [3, 3, 512, 512], initializer=load_weight('conv5_2')),
      "5_3":tf.get_variable('conv5_3', shape = [3, 3, 512, 512], initializer=load_weight('conv5_3')),
      }

  Bs = {
      "1_1":tf.get_variable('bias1_1', shape = [64], initializer=load_bias('conv1_1')),
      "1_2":tf.get_variable('bias1_2', shape = [64], initializer=load_bias('conv1_2')),

      "2_1":tf.get_variable('bias2_1', shape = [128], initializer=load_bias('conv2_1')),
      "2_2":tf.get_variable('bias2_2', shape = [128], initializer=load_bias('conv2_2')),

      "3_1":tf.get_variable('bias3_1', shape = [256], initializer=load_bias('conv3_1')),
      "3_2":tf.get_variable('bias3_2', shape = [256], initializer=load_bias('conv3_2')),
      "3_3":tf.get_variable('bias3_3', shape = [256], initializer=load_bias('conv3_3')),

      "4_1":tf.get_variable('bias4_1', shape = [512], initializer=load_bias('conv4_1')),
      "4_2":tf.get_variable('bias4_2', shape = [512], initializer=load_bias('conv4_2')),
      "4_3":tf.get_variable('bias4_3', shape = [512], initializer=load_bias('conv4_3')),

      "5_1":tf.get_variable('bias5_1', shape = [512], initializer=load_bias('conv5_1')),
      "5_2":tf.get_variable('bias5_2', shape = [512], initializer=load_bias('conv5_2')),
      "5_3":tf.get_variable('bias5_3', shape = [512], initializer=load_bias('conv5_3')),
      }
  return Ws, Bs

def init_weights():
  def init_with_normal():
    return tf.random_normal_initializer(mean=0.0, stddev=0.02)

  WEs = {

      "6":tf.get_variable('e_conv_6', shape = [7, 7, 512, 4096], initializer=init_with_normal()),
      "7":tf.get_variable('e_conv_7', shape = [1, 1, 4096, 4096], initializer=init_with_normal()),

      "8":tf.get_variable('e_conv_8', shape = [1, 1, 4096, 21], initializer=init_with_normal()),
      "9":tf.get_variable('e_conv_9', shape = [1, 1, 512, 21], initializer=init_with_normal()),
      "10":tf.get_variable('e_conv_10', shape = [1, 1, 256, 21], initializer=init_with_normal()),
      }

  BEs = {

      "6":tf.get_variable('e_bias_6', shape = [4096], initializer=init_with_normal()),
      "7":tf.get_variable('e_bias_7', shape = [4096], initializer=init_with_normal()),
      }

  WDs = {
      "1":tf.get_variable('d_conv_1', shape = [4, 4, 21, 21], initializer=init_with_normal()),
      "2":tf.get_variable('d_conv_2', shape = [4, 4, 21, 21], initializer=init_with_normal()),
      "3":tf.get_variable('d_conv_3', shape = [16, 16, 21, 21], initializer=init_with_normal()),
      }
  return WEs, BEs, WDs


def model_FCN8S(x, y, Ws, Bs, WEs, BEs, WDs, drop_prob = 0.5):

  mp_ksize= [1, 1, 2, 2]
  mp_strides=[1, 1, 2, 2]

  relued = conv_relu(x, Ws['1_1'], Bs['1_1'])
  relued = conv_relu(relued, Ws['1_2'], Bs['1_2'])
  pooled1 = tf.nn.max_pool(relued, ksize=mp_ksize, strides=mp_strides, padding='SAME', data_format='NCHW')

  relued = conv_relu(pooled1, Ws['2_1'], Bs['2_1'])
  relued = conv_relu(relued, Ws['2_2'], Bs['2_2'])
  pooled2 = tf.nn.max_pool(relued, ksize=mp_ksize, strides=mp_strides, padding='SAME', data_format='NCHW')

  relued = conv_relu(pooled2, Ws['3_1'], Bs['3_1'])
  relued = conv_relu(relued, Ws['3_2'], Bs['3_2'])
  relued = conv_relu(relued, Ws['3_3'], Bs['3_3'])
  pooled3 = tf.nn.max_pool(relued, ksize=mp_ksize, strides=mp_strides, padding='SAME', data_format='NCHW')

  relued = conv_relu(pooled3, Ws['4_1'], Bs['4_1'])
  relued = conv_relu(relued, Ws['4_2'], Bs['4_2'])
  relued = conv_relu(relued, Ws['4_3'], Bs['4_3'])
  pooled4 = tf.nn.max_pool(relued, ksize=mp_ksize, strides=mp_strides, padding='SAME', data_format='NCHW')

  relued = conv_relu(pooled4, Ws['5_1'], Bs['5_1'])
  relued = conv_relu(relued, Ws['5_2'], Bs['5_2'])
  relued = conv_relu(relued, Ws['5_3'], Bs['5_3'])
  pooled5 = tf.nn.max_pool(relued, ksize=mp_ksize, strides=mp_strides, padding='SAME', data_format='NCHW')

  relued = conv_relu(pooled5, WEs['6'], BEs['6'])
  dropouted = tf.nn.dropout(relued, drop_prob)

  relued = conv_relu(dropouted, WEs['7'], BEs['7'])
  dropouted = tf.nn.dropout(relued, drop_prob)

  score_fr = tf.nn.conv2d(dropouted, WEs['8'], strides=[1, 1, 1, 1], padding='SAME', data_format='NCHW')

  score_pool4 = tf.nn.conv2d(pooled4, WEs['9'], strides=[1, 1, 1, 1], padding='SAME', data_format='NCHW')
  shape_list = tf.shape(score_pool4)
  out_shape = tf.pack(shape_list)
  upscore2 = tf.nn.conv2d_transpose(score_fr, WDs['1'], out_shape, strides=[1, 1, 2, 2], padding='SAME', data_format='NCHW')
  fuse_pool4 = tf.add(upscore2, score_pool4)

  score_pool3 = tf.nn.conv2d(pooled3, WEs['10'], strides=[1, 1, 1, 1], padding='SAME', data_format='NCHW')
  shape_list = tf.shape(score_pool3)
  out_shape = tf.pack(shape_list)
  upscore_pool4 = tf.nn.conv2d_transpose(fuse_pool4, WDs['2'], out_shape, strides=[1, 1, 2, 2], padding='SAME', data_format='NCHW')
  fuse_pool3 = tf.add(upscore_pool4, score_pool3)

  shape_list = tf.shape(y)
  out_shape = tf.pack(shape_list)
  upscore_pool8 = tf.nn.conv2d_transpose(fuse_pool4, WDs['3'], out_shape, strides=[1, 1, 8, 8], padding='SAME', data_format='NCHW')

  final_score = upscore_pool8

  return final_score

def get_opt(loss, scope):
  var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

  print "============================"
  print scope
  print var_list
  for item in var_list:
    print item.name
  # Optimizer: set up a variable that's incremented once per batch and
  # controls the learning rate decay.
#  batch = tf.Variable(0, dtype=tf.int32)
  # Decay once per epoch, using an exponential schedule starting at 0.01.
#  learning_rate = tf.train.exponential_decay(
#      FLAGS.learning_rate,                # Base learning rate.
#      batch,  # Current index into the dataset.
#      1,          # Decay step.
#      FLAGS.weight_decay,                # Decay rate.
#      staircase=True)
  # Use simple momentum for the optimization.
#  optimizer = tf.train.MomentumOptimizer(learning_rate,
#                                         FLAGS.momentum).minimize(loss,
#                                                       var_list=var_list,
#                                                       global_step=batch)
#
#  return optimizer
  optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1)
  grads = optimizer.compute_gradients(loss, var_list=var_list)
  print grads
  return optimizer.apply_gradients(grads)

#
# reference: https://www.tensorflow.org/how_tos/reading_data/
#
def main(args):

  print "thanks to http://www.deeplearningmodel.net/"
  print "download pretrained VGG16 model"
  commom.maybe_download("./", "VGG_16.npy", "https://lexiondebug.blob.core.windows.net/mlmodel/models/VGG_16.npy")

  opts, args = getopt.getopt(sys.argv[1:], "s:", ["save_dir="])

  save_dir=FLAGS.save_dir

  for o, arg in opts:
    if o in ("-s", "--save_dir"):
      save_dir=arg
      print "checkpoint dir:", save_dir

  info_path="../data/VOCdevkit/VOC2012/ImageSets/Segmentation/"
  data_path="../data/VOCdevkit/VOC2012/JPEGImages/"
  annot_path="../data/VOCdevkit/VOC2012/SegmentationClass/"

  with open(os.path.join(info_path, "train.txt")) as f:
    filelist = f.readlines()

  colormap, palette = common.build_colormap_lookup(256)

    # Create a session for running operations in the Graph.
  _x = tf.placeholder(tf.float32)
  _y = tf.placeholder(tf.int32)


  mean = tf.constant(np.array((104.00698793,116.66876762,122.67891434)), dtype=np.float32)
  x = _x - mean
  y = _y

  x = tf.Print(x, [tf.shape(x)], message="x:")
  y = tf.Print(y, [tf.shape(y)], message="y:")

  y = idx2onehot(y)

  x = tf.transpose(x, perm=[2, 0, 1])
  y = tf.transpose(y, perm=[2, 0, 1])

  x = tf.Print(x, [tf.shape(x)], message="x_after:")
  y = tf.Print(y, [tf.shape(y)], message="y_after:")

  x = tf.expand_dims(x, 0)
  y = tf.expand_dims(y, 0)

  x = tf.Print(x, [tf.shape(x)[:2], tf.shape(x)[2:]], message="x_final:")
  y = tf.Print(y, [tf.shape(y)[:2], tf.shape(y)[2:]], message="y_final:")

  pretrained = commom.load_pretrained("./VGG_16.npy")

  with tf.variable_scope("VGG16") as scope:
    Ws, Bs = init_VGG16(pretrained)
  with tf.variable_scope("FCN8S") as scope:
    WEs, BEs, WDs = init_weights()

  out = model_FCN8S(x, y, Ws, Bs, WEs, BEs, WDs)

  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(out, y))

  opt = get_opt(loss, "FCN8S")
  valid = model_FCN8S(x, y, Ws, Bs, WEs, BEs, WDs, drop_prob=1.0)

  init_op = tf.group(tf.global_variables_initializer(),
                     tf.local_variables_initializer())
  with tf.Session() as sess:
    # Initialize the variables (the trained variables and the
    sess.run(init_op)

    # Start input enqueue threads.
    saver = tf.train.Saver()
    checkpoint = tf.train.latest_checkpoint(save_dir)
    print "checkpoint: %s" % checkpoint
    if checkpoint:
      print "Restoring from checkpoint", checkpoint
      saver.restore(sess, checkpoint)
    else:
      print "Couldn't find checkpoint to restore from. Starting over."
      dt = datetime.now()
      filename = "checkpoint" + dt.strftime("%Y-%m-%d_%H-%M-%S")
      checkpoint = os.path.join(save_dir, filename)

    for epoch in range(FLAGS.max_epoch):
      print "==================================================================="
      random.shuffle(filelist)
      for itr, filename in enumerate(filelist):
        print "---------------------------------------------------------------------"
        print "  ", filename[:-1]
        filename = filename[:-1] #remove "\n"
        jpegpath = os.path.join(data_path, filename + ".jpg")
        labelpath = os.path.join(annot_path, filename + ".png")

        _img = Image.open(jpegpath)
        _label = Image.open(labelpath)

        img, label = common.resize_if_need(_img, _label, FLAGS.img_size)
        img = np.array(img)
        label = np.array(label)

        #img = img.transpose(2, 0, 1)
        h = img.shape[0]
        w = img.shape[1]
        c = img.shape[2]
        print img.shape
        print label.shape, label.dtype

        label[label==255]=0

        if h < FLAGS.img_size and w < FLAGS.img_size:
          print "!!!!!!!!!!!!!!!!!!!!!!!!!!!"

        feed_dict = {_x: img, _y: label}
        _= sess.run([opt], feed_dict=feed_dict)
        images_value, labels_value = sess.run([x, y], feed_dict=feed_dict)

        filepath = os.path.join(save_dir, filename + "_est.png")
        #scipy.misc.imsave(filepath, contrastive_sample_vis)
        _img.close()
        _label.close()

        if itr > 1 and itr % 300 == 0:
          print "#######################################################"
          saver.save(sess, checkpoint)

if __name__ == "__main__":
  tf.app.run()

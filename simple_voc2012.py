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
import cv2

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
tf.flags.DEFINE_integer("img_size", "256", "sample image size")
tf.flags.DEFINE_string("save_dir", "simple_voc2012_checkpoints", "dir for checkpoints")
tf.flags.DEFINE_integer("nclass", "33", "size of class")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Momentum Optimizer")
tf.flags.DEFINE_float("beta1", "0.05", "beta1 for Adam optimizer")
tf.flags.DEFINE_float("momentum", "0.9", "momentum for Momentum Optimizer")
tf.flags.DEFINE_float("weight_decay", "0.0016", "Learning rate for Momentum Optimizer")
tf.flags.DEFINE_integer("num_threads", "6", "max thread number")
tf.flags.DEFINE_float("eps", "1e-5", "epsilon for various operation")

def onehot2idx(tensor):

  tensor = tf.transpose(tensor, perm=[0, 2, 3, 1])
  tensor = tf.cast(tensor, tf.int32)
  ret = tf.argmax(tensor, axis=[0 ,1 ,2])

  return ret

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

def init_Alexnet(pretrained):
  def load_weight(name):
    print pretrained[name][0].shape
    return tf.constant_initializer(value=pretrained[name][0])

  def load_bias(name):
    print pretrained[name][1].shape
    return tf.constant_initializer(value=pretrained[name][1])

  # initialize weights, biases for Encoder
  Ws = {
      "1":tf.get_variable('conv1', shape = [11, 11, FLAGS.channel, 96], initializer=load_weight('conv1')),

      "2":tf.get_variable('conv2', shape = [5, 5, 96, 256], initializer=load_weight('conv2')),

      "3":tf.get_variable('conv3', shape = [3, 3, 256, 384], initializer=load_weight('conv3')),

      "4":tf.get_variable('conv4', shape = [3, 3, 384, 384], initializer=load_weight('conv4')),

      "5":tf.get_variable('conv5', shape = [3, 3, 384, 256], initializer=load_weight('conv5')),
      }

  Bs = {
      "1":tf.get_variable('bias1', shape = [96], initializer=load_bias('conv1')),

      "2":tf.get_variable('bias2', shape = [256], initializer=load_bias('conv2')),

      "3":tf.get_variable('bias3', shape = [384], initializer=load_bias('conv3')),

      "4":tf.get_variable('bias4', shape = [384], initializer=load_bias('conv4')),

      "5":tf.get_variable('bias5', shape = [256], initializer=load_bias('conv5')),
      }

  return Ws, Bs

def init_Simple():
  def init_with_normal():
    return tf.truncated_normal_initializer(mean=0.0, stddev=0.02)

  # initialize weights, biases for Encoder
  Ws = {
      "1":tf.get_variable('conv1', shape = [11, 11, FLAGS.channel, 96], initializer=init_with_normal()),

      "2":tf.get_variable('conv2', shape = [5, 5, 96, 256], initializer=init_with_normal()),

      "3":tf.get_variable('conv3', shape = [3, 3, 256, 384], initializer=init_with_normal()),

      "4":tf.get_variable('conv4', shape = [3, 3, 384, 384], initializer=init_with_normal()),

      "5":tf.get_variable('conv5', shape = [3, 3, 384, 256], initializer=init_with_normal()),
      }

  Bs = {
      "1":tf.get_variable('bias1', shape = [96], initializer=init_with_normal()),

      "2":tf.get_variable('bias2', shape = [256], initializer=init_with_normal()),

      "3":tf.get_variable('bias3', shape = [384], initializer=init_with_normal()),

      "4":tf.get_variable('bias4', shape = [384], initializer=init_with_normal()),

      "5":tf.get_variable('bias5', shape = [256], initializer=init_with_normal()),
      }

  return Ws, Bs

def model(x, y, Ws, Bs, WDs, drop_prob = 0.5):
  def batch_normalization(tensor):
    mean, var = tf.nn.moments(tensor, [0, 1, 2])
    out = tf.nn.batch_normalization(tensor, mean, var, 0, 1, FLAGS.eps)
    return out

  def leaky_relu(tensor):
    return tf.maximum(tensor*0.2, tensor)

  mp_ksize= [1, 1, 2, 2]
  mp_strides=[1, 1, 2, 2]

  relued = conv_relu(x, Ws['1'], Bs['1'])
  pooled1 = tf.nn.max_pool(relued, ksize=mp_ksize, strides=mp_strides, padding='SAME', data_format='NCHW')

  relued = conv_relu(pooled1, Ws['2'], Bs['2'])
  pooled2 = tf.nn.max_pool(relued, ksize=mp_ksize, strides=mp_strides, padding='SAME', data_format='NCHW')

  relued = conv_relu(pooled2, Ws['3'], Bs['3'])
  pooled3 = tf.nn.max_pool(relued, ksize=mp_ksize, strides=mp_strides, padding='SAME', data_format='NCHW')

  relued = conv_relu(pooled3, Ws['4'], Bs['4'])
  pooled4 = tf.nn.max_pool(relued, ksize=mp_ksize, strides=mp_strides, padding='SAME', data_format='NCHW')

  relued = conv_relu(pooled4, Ws['5'], Bs['5'])
  pooled5 = tf.nn.max_pool(relued, ksize=mp_ksize, strides=mp_strides, padding='SAME', data_format='NCHW')

  shape_list = tf.shape(pooled4)
  print shape_list
  out_shape = tf.pack([shape_list[0], 256, shape_list[2], shape_list[3]])
  print out_shape
  deconved = tf.nn.conv2d_transpose(pooled5, WDs['1'], out_shape, strides=[1, 1, 2, 2], padding='SAME', data_format='NCHW')
  #normalized = batch_norm_layer(deconved, "discriminator/bnd1", False)
  normalized = batch_normalization(deconved)
  relued = leaky_relu(normalized)

  shape_list = tf.shape(pooled3)
  out_shape = tf.pack([shape_list[0], 256, shape_list[2], shape_list[3]])
  deconved = tf.nn.conv2d_transpose(relued, WDs['2'], out_shape, strides=[1, 1, 2, 2], padding='SAME', data_format='NCHW')
  #normalized = batch_norm_layer(deconved, "discriminator/bnd2", False)
  normalized = batch_normalization(deconved)
  relued = leaky_relu(normalized)

  shape_list = tf.shape(pooled2)
  out_shape = tf.pack([shape_list[0], 128, shape_list[2], shape_list[3]])
  deconved = tf.nn.conv2d_transpose(relued, WDs['3'], out_shape, strides=[1, 1, 2, 2], padding='SAME', data_format='NCHW')
  #normalized = batch_norm_layer(deconved, "discriminator/bnd3", False)
  normalized = batch_normalization(deconved)
  relued = leaky_relu(normalized)

  shape_list = tf.shape(pooled1)
  out_shape = tf.pack([shape_list[0], 64, shape_list[2], shape_list[3]])
  deconved = tf.nn.conv2d_transpose(relued, WDs['4'], out_shape, strides=[1, 1, 2, 2], padding='SAME', data_format='NCHW')
  #normalized = batch_norm_layer(deconved, "discriminator/bnd4", False)
  normalized = batch_normalization(deconved)
  relued = leaky_relu(normalized)

  shape_list = tf.shape(y)
  out_shape = tf.pack([shape_list[0], FLAGS.nclass, shape_list[2], shape_list[3]])
  deconved = tf.nn.conv2d_transpose(relued, WDs['5'], out_shape, strides=[1, 1, 2, 2], padding='SAME', data_format='NCHW')
  #normalized = batch_norm_layer(deconved, "discriminator/bnd4", False)
  normalized = deconved
  relued = leaky_relu(normalized)

  final_score = relued

  return final_score

def init_enc():
  def init_with_normal():
    return tf.truncated_normal_initializer(mean=0.0, stddev=0.1)

  def init_bias():
    return tf.constant_initializer(value=0.1)

  # initialize weights, biases for Encoder
  ksize = 7
  WEs = {
      1:tf.get_variable('e_conv1', shape = [ksize, ksize, FLAGS.channel, 64], initializer=init_with_normal()),

      2:tf.get_variable('e_conv2', shape = [ksize, ksize, 64, 128], initializer=init_with_normal()),

      3:tf.get_variable('e_conv3', shape = [ksize, ksize, 128, 256], initializer=init_with_normal()),

      4:tf.get_variable('e_conv4', shape = [ksize, ksize, 256, 256], initializer=init_with_normal()),

      5:tf.get_variable('e_conv5', shape = [ksize, ksize, 256, 256], initializer=init_with_normal()),
      }
  BEs = {
      1:tf.get_variable('e_bias1', shape = [64], initializer=init_bias()),

      2:tf.get_variable('e_bias2', shape = [128], initializer=init_bias()),

      3:tf.get_variable('e_bias3', shape = [256], initializer=init_bias()),

      4:tf.get_variable('e_bias4', shape = [256], initializer=init_bias()),

      5:tf.get_variable('e_bias5', shape = [256], initializer=init_bias()),
      }

  return WEs, BEs

def init_dec():
  def init_with_normal():
    return tf.truncated_normal_initializer(mean=0.0, stddev=0.1)

  def init_bias():
    return tf.constant_initializer(value=0.1)

  ksize = 7
  WDs = {
      1:tf.get_variable('d_conv1', shape = [ksize, ksize, 256, 256], initializer=init_with_normal()),

      2:tf.get_variable('d_conv2', shape = [ksize, ksize, 256, 256], initializer=init_with_normal()),

      3:tf.get_variable('d_conv3', shape = [ksize, ksize, 128, 256], initializer=init_with_normal()),

      4:tf.get_variable('d_conv4', shape = [ksize, ksize, 64, 128], initializer=init_with_normal()),

      5:tf.get_variable('d_conv5', shape = [ksize, ksize, FLAGS.nclass, 64], initializer=init_with_normal()),
      }

  BDs = {
      1:tf.get_variable('d_bias1', shape = [256], initializer=init_bias()),

      2:tf.get_variable('d_bias2', shape = [256], initializer=init_bias()),

      3:tf.get_variable('d_bias3', shape = [128], initializer=init_bias()),

      4:tf.get_variable('d_bias4', shape = [64], initializer=init_bias()),

      5:tf.get_variable('d_bias5', shape = [FLAGS.nclass], initializer=init_bias()),
      }
  return WDs, BDs

def model_tiny(x, y, WEs, BEs, WDs, BDs, drop_prob = 0.5):

  def batch_normalization(tensor):
    mean, var = tf.nn.moments(tensor, [0, 1, 2])
    out = tf.nn.batch_normalization(tensor, mean, var, 0, 1, FLAGS.eps)
    return out

  def leaky_relu(tensor):
    return tf.maximum(tensor*0.2, tensor)

  mp_strides=[1, 1, 2, 2]

  conved = tf.nn.conv2d(x, WEs[1], strides=mp_strides, padding='SAME', data_format='NCHW')
  conved = tf.nn.bias_add(conved, BEs[1], data_format='NCHW')
  # skip batch normalization by DCGAN
  #normalized = batch_normalization(conved)
  encoded1 = leaky_relu(conved)

  conved = tf.nn.conv2d(encoded1, WEs[2], strides=mp_strides, padding='SAME', data_format='NCHW')
  conved = tf.nn.bias_add(conved, BEs[2], data_format='NCHW')
  normalized = batch_normalization(conved)
  encoded2 = leaky_relu(normalized)

  conved = tf.nn.conv2d(encoded2, WEs[3], strides=mp_strides, padding='SAME', data_format='NCHW')
  conved = tf.nn.bias_add(conved, BEs[3], data_format='NCHW')
  normalized = batch_normalization(conved)
  encoded3 = leaky_relu(normalized)

  conved = tf.nn.conv2d(encoded3, WEs[4], strides=mp_strides, padding='SAME', data_format='NCHW')
  conved = tf.nn.bias_add(conved, BEs[4], data_format='NCHW')
  normalized = batch_normalization(conved)
  encoded4 = leaky_relu(normalized)

  conved = tf.nn.conv2d(encoded4, WEs[5], strides=mp_strides, padding='SAME', data_format='NCHW')
  conved = tf.nn.bias_add(conved, BEs[5], data_format='NCHW')
  normalized = batch_normalization(conved)
  encoded5 = leaky_relu(normalized)

  shape_list = tf.shape(encoded4)
  out_shape = tf.pack([shape_list[0], 256, shape_list[2], shape_list[3]])
  deconved = tf.nn.conv2d_transpose(encoded5, WDs[1], out_shape, strides=mp_strides, padding='SAME', data_format='NCHW')
  deconved = tf.nn.bias_add(deconved, BDs[1], data_format='NCHW')
  #normalized = batch_norm_layer(deconved, "discriminator/bnd1", False)
  normalized = batch_normalization(deconved)
  relued = tf.nn.relu(normalized)
  relued = tf.add(normalized, encoded4)

  shape_list = tf.shape(encoded3)
  out_shape = tf.pack([shape_list[0], 256, shape_list[2], shape_list[3]])
  deconved = tf.nn.conv2d_transpose(relued, WDs[2], out_shape, strides=mp_strides, padding='SAME', data_format='NCHW')
  deconved = tf.nn.bias_add(deconved, BDs[2], data_format='NCHW')
  #normalized = batch_norm_layer(deconved, "discriminator/bnd2", False)
  normalized = batch_normalization(deconved)
  relued = tf.nn.relu(normalized)
  relued = tf.add(normalized, encoded3)

  shape_list = tf.shape(encoded2)
  out_shape = tf.pack([shape_list[0], 128, shape_list[2], shape_list[3]])
  deconved = tf.nn.conv2d_transpose(relued, WDs[3], out_shape, strides=mp_strides, padding='SAME', data_format='NCHW')
  deconved = tf.nn.bias_add(deconved, BDs[3], data_format='NCHW')
  #normalized = batch_norm_layer(deconved, "discriminator/bnd3", False)
  normalized = batch_normalization(deconved)
  relued = tf.nn.relu(normalized)
  #relued = tf.add(normalized, encoded2)

  shape_list = tf.shape(encoded1)
  out_shape = tf.pack([shape_list[0], 64, shape_list[2], shape_list[3]])
  deconved = tf.nn.conv2d_transpose(relued, WDs[4], out_shape, strides=mp_strides, padding='SAME', data_format='NCHW')
  deconved = tf.nn.bias_add(deconved, BDs[4], data_format='NCHW')
  #normalized = batch_norm_layer(deconved, "discriminator/bnd4", False)
  normalized = batch_normalization(deconved)
  relued = tf.nn.relu(normalized)
  #relued = tf.add(normalized, encoded1)

  shape_list = tf.shape(y)
  out_shape = tf.pack([shape_list[0], FLAGS.nclass, shape_list[2], shape_list[3]])
  deconved = tf.nn.conv2d_transpose(relued, WDs[5], out_shape, strides=mp_strides, padding='SAME', data_format='NCHW')
  deconved = tf.nn.bias_add(deconved, BDs[5], data_format='NCHW')
  #normalized = batch_norm_layer(deconved, "discriminator/bnd4", False)
  normalized = deconved
  relued = tf.nn.relu(normalized)

  final_score = relued

  return final_score

def get_opt(loss, scope):
  var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

  print "============================"
  print scope
  for item in var_list:
    print item.name
  # Optimizer: set up a variable that's incremented once per batch and
  # controls the learning rate decay.
#  batch = tf.Variable(0, dtype=tf.int32)
#  # Decay once per epoch, using an exponential schedule starting at 0.01.
#  learning_rate = tf.train.exponential_decay(
#     FLAGS.learning_rate,                # Base learning rate.
#      batch,  # Current index into the dataset.
#      1,          # Decay step.
#      FLAGS.weight_decay,                # Decay rate.
#      staircase=True)
# # Use simple momentum for the optimization.
#  optimizer = tf.train.MomentumOptimizer(learning_rate,
#                                         FLAGS.momentum).minimize(loss,
#                                                       var_list=var_list,
#                                                       global_step=batch)
#
#  return optimizer
  #optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1)
  optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
  grads = optimizer.compute_gradients(loss, var_list=var_list)
  return optimizer.apply_gradients(grads)

#
# reference: https://www.tensorflow.org/how_tos/reading_data/
#
def main(args):

  print "thanks to https://github.com/guerzh/tf_weights"
  print "download pretrained Alexnet model"
  common.maybe_download("./", "bvlc_alexnet.npy", "http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy")

  opts, args = getopt.getopt(sys.argv[1:], "s:", ["save_dir="])

  save_dir=FLAGS.save_dir

  for o, arg in opts:
    if o in ("-s", "--save_dir"):
      save_dir=arg
      print "checkpoint dir:", save_dir

  colormap, palette = common.build_colormap_lookup(256)

  #datacenter = common.VOC2012(FLAGS.img_size)
  #datacenter = common.CamVid(FLAGS.img_size)
  datacenter = common.PASCALCONTEXT(FLAGS.img_size)

    # Create a session for running operations in the Graph.
  _x = tf.placeholder(tf.float32)
  _y = tf.placeholder(tf.int32)

  #mean = tf.constant(np.array((104.00698793,116.66876762,122.67891434)), dtype=np.float32)
  #x = _x - mean
  x = _x
  y = _y

  #x = tf.Print(x, [tf.shape(x)], message="x:")
  #y = tf.Print(y, [tf.shape(y)], message="y:")

  y = common.idx2onehot(y, FLAGS.nclass)

  x = tf.transpose(x, perm=[2, 0, 1])
  y = tf.transpose(y, perm=[2, 0, 1])

  #x = tf.Print(x, [tf.shape(x)], message="x_after:")
  #y = tf.Print(y, [tf.shape(y)], message="y_after:")

  x = tf.expand_dims(x, 0)
  y = tf.expand_dims(y, 0)

  #x = tf.Print(x, [tf.shape(x)[:2], tf.shape(x)[2:]], message="x_final:")
  #y = tf.Print(y, [tf.shape(y)[:2], tf.shape(y)[2:]], message="y_final:")

  pretrained = common.load_pretrained("./bvlc_alexnet.npy")

  with tf.variable_scope("Tiny") as scope:
    WEs, BEs = init_enc()
    WDs, BDs = init_dec()

  out = model_tiny(x, y, WEs, BEs, WDs, BDs)
  out = tf.transpose(out, perm=[0, 2, 3, 1])
  y = tf.transpose(y, perm=[0, 2, 3, 1])

  #y = tf.Print(y, [tf.shape(y)[:2], tf.shape(y)[2:]], message="y_restored:")
  #out = tf.Print(out, [tf.shape(out)[:2], tf.shape(out)[2:]], message="out_restored:")

  temp = tf.squeeze(y, axis=[0])
  restored = tf.argmax(temp, 2)

  logits = tf.reshape(out, shape=[-1, FLAGS.nclass])
  targets = tf.reshape(y, shape=[-1, FLAGS.nclass])
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, targets))

  opt = get_opt(loss, "Tiny")

  valid = model_tiny(x, y, WEs, BEs, WDs, BDs, drop_prob=1.0)
  valid = tf.transpose(valid, perm=[0, 2, 3, 1])

  temp = tf.squeeze(out, axis=[0])
  #temp = print_shape(temp, "temp")
  indexed_out = tf.argmax(temp, 2)

  accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(indexed_out, tf.int32), _y), tf.float32))

  init_op = tf.group(tf.global_variables_initializer(),
                     tf.local_variables_initializer())

  num_threads = FLAGS.num_threads
  with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=num_threads)) as sess:
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
      print "#####################################################################"
      datacenter.shuffle()
      for itr in range(datacenter.size):
        print "==================================================================="
        print "[", epoch, "]", "%d/%d"%(itr, datacenter.size)
        (filename, img, label) = datacenter.getPair(itr)

        h = img.shape[0]
        w = img.shape[1]
        c = img.shape[2]
        print "  ", filename, img.shape, label.shape

        #label[label==255]=0

        feed_dict = {_x: img, _y: label}
        _, loss_val = sess.run([opt, loss], feed_dict=feed_dict)
        accuracy_val = sess.run([accuracy], feed_dict=feed_dict)
        print "\tloss:", loss_val
        print "\taccuracy:", accuracy_val

        #images_value, labels_value = sess.run([x, y], feed_dict=feed_dict)
        restored_val, indexed_out_val = sess.run([restored, indexed_out], feed_dict=feed_dict)

        label_vis = common.convert_label2bgr(label, palette)
        #restored_vis = common.convert_label2bgr(restored_val, palette)
        out_vis = common.convert_label2bgr(indexed_out_val, palette)
        cv2.imshow('visualization', common.img_listup([img, label_vis, out_vis]))
        cv2.waitKey(2000)

        filepath = os.path.join(save_dir, filename + "_est.png")
        #scipy.misc.imsave(filepath, contrastive_sample_vis)

        if itr > 1 and itr % 300 == 0:
          print "#######################################################"
          saver.save(sess, checkpoint)

  cv2.destroyAllWindows()

if __name__ == "__main__":
  tf.app.run()

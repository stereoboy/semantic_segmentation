
import tensorflow as tf
import os
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin

object_table = [
	'blank',
	'aeroplane',
	'bicycle',
	'bird',
	'boat',
	'bottle',
	'bus',
	'car',
	'cat',
	'chair',
	'cow',
	'diningtable',
	'dog',
	'horse',
	'motorbike',
	'person',
	'pottedplant',
	'sheep',
	'sofa',
	'train',
	'tvmonitor',
]

def maybe_download(directory, filename, url):
  print 'Try to dwnloaded', url
  if not tf.gfile.Exists(directory):
    tf.gfile.MakeDirs(directory)
  filepath = os.path.join(directory, filename)
  if not tf.gfile.Exists(filepath):
    filepath, _ = urllib.request.urlretrieve(url, filepath)
    with tf.gfile.GFile(filepath) as f:
      size = f.size()
    print 'Successfully downloaded', filename, size, 'bytes.'
  return filepath

def load_pretrained(filepath):
  return np.load(filepath).item()

#
# reference: VOCdevkit/VOCcode/VOClabelcolormap.m
#
def build_colormap_lookup(N):

  colormap = {}
  palette = np.zeros((N, 3), np.uint8)

  for i in range(0, N):
    ID = i 
    r = 0
    g = 0
    b = 0
    for j in range(0, 8):
      r = r | (((ID&0x1)>>0) <<(7-j))
      g = g | (((ID&0x2)>>1) <<(7-j))
      b = b | (((ID&0x4)>>2) <<(7-j))
      ID = ID >> 3

    colormap[(r,g,b)] = i
    palette[i, 0] = r
    palette[i, 1] = g
    palette[i, 2] = b

  palette = np.array(palette, np.uint8).reshape(-1, 3)
  return colormap, palette

def resize_if_need(_img, _label, max_size):
  print "resize_if_need()"
  print _img
  w = _img.size[0]
  h = _img.size[1]
  print w, h

  if h <= FLAGS.img_size and w <= FLAGS.img_size:
    # do nothing
    return _img, _label

  if h > w:
    new_w = max_size*w/h
    new_h = max_size
  else:
    new_w = max_size
    new_h = max_size*h/w

  print "need to resize (%d, %d) to (%d, %d)" % (h, w, new_h, new_w)

  _resized_img = _img.resize((new_w, new_h))
  _resized_label = _label.resize((new_w, new_h), Image.NEAREST)
    
  return _resized_img, _resized_label


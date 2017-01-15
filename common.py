
import tensorflow as tf
import os
import numpy as np
from PIL import Image
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import random
import glob
import cv2
import scipy.io

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
hidden_palette = None
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
  global hidden_palette
  hidden_palette = palette
  return colormap, palette

def save_label_png(filepath, array, palette):

  img = Image.fromarray(array, mode='P')
  img.putpalette(palette.reshape(-1))
  img.save("./test.png","png")

def convert_label2bgr(indexed, palette):
  #print "convert_label2bgr()", indexed.shape
  
  ret = np.array(map((lambda x: palette[x]), indexed))
  ret = ret[:, :, ::-1]
  return ret

def resize_if_need(_img, _label, max_size):
  w = _img.size[0]
  h = _img.size[1]

  if h <= max_size and w <= max_size:
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

def img_listup(imgs):
  size = len(imgs)
  (h, w) = imgs[0].shape[:2]
  out = np.zeros((h, w*size, 3), np.uint8)

  offset = 0
  for i in range(size):
    out[:, offset: offset + w] = imgs[i]
    offset += w
 
  return out

class DataCenter(object):
  def nextPair(self):
    raise NotImplementedError

  def shuffle(self):
    raise NotImplementedError
  
  def getPair(self, idx):
    raise NotImplementedError
  
  @property
  def size(self):
    raise NotImplementedError

class VOC2012(DataCenter):
  def __init__(self, max_img_size):

    self.max_img_size = max_img_size

    info_path="../data/VOCdevkit/VOC2012/ImageSets/Segmentation/"
    data_path="../data/VOCdevkit/VOC2012/JPEGImages/"
    annot_path="../data/VOCdevkit/VOC2012/SegmentationClass/"

    with open(os.path.join(info_path, "train.txt")) as f:
      filelist = f.readlines()

    datapairs = []
    for filename in filelist:
      filename = filename[:-1] #remove "\n"
      imgpath = os.path.join(data_path, filename + ".jpg")
      labelpath = os.path.join(annot_path, filename + ".png")
      datapairs.append((imgpath, labelpath))

    self.datapairs = datapairs
    self._size = len(datapairs)

  def shuffle(self):
    random.shuffle(self.datapairs)

  @property
  def size(self):
    return self._size

  def getPair(self, idx):
    (jpegpath, labelpath) = self.datapairs[idx]
    filename = os.path.basename(jpegpath)

    _img = Image.open(jpegpath)
    _label = Image.open(labelpath)

    img, label = resize_if_need(_img, _label, self.max_img_size)
    img = np.array(img)
    label = np.array(label)
    _img.close()
    _label.close()
    return (filename, img, label)

class CamVid(DataCenter):
  def __init__(self, max_img_size):
    
    self.max_img_size = max_img_size

    camvidpath = "../data/SegNet-Tutorial/CamVid/"
    path1 = camvidpath + 'train/'
    path2 = camvidpath + 'trainannot/'

    trainimglist = glob.glob(path1 + '/*.png')
    trainannotlist = glob.glob(path2 + '/*.png')
    datapairs = zip(trainimglist, trainannotlist)

    self.datapairs = datapairs
    self._size = len(datapairs)

  def shuffle(self):
    random.shuffle(self.datapairs)

  @property
  def size(self):
    return self._size

  def getPair(self, idx):
    (jpegpath, labelpath) = self.datapairs[idx]
    filename = os.path.basename(jpegpath)

    _img = Image.open(jpegpath)
    _label = Image.open(labelpath)

    img, label = resize_if_need(_img, _label, self.max_img_size)
    img = np.array(img)
    label = np.array(label)
    _img.close()
    _label.close()
    return (filename, img, label)

class PASCALCONTEXT(DataCenter):
  def __init__(self, max_img_size):
    import scipy.io

    self.max_img_size = max_img_size

    full_label_path="../data/PASCALCONTEXT/labels.txt"
    label33_path = "../data/PASCALCONTEXT/33_labels.txt"

    lookup = {}
    full_label_text = None
    label33_text = None

    with open(full_label_path) as f:
      full_label_text = f.readlines()
    
    with open(label33_path) as f:
      label33_text = f.readlines()
 
    full_label_table = {}
    for label in full_label_text:
      n, c = label.rstrip().split(": ", 1)
      full_label_table[int(n)] = c
   
    label33_table = {0:'others'}
    for label in label33_text:
      if len(label.rstrip().split(": ", 1)) == 2:
        n, c = label.rstrip().split(": ", 1)
        label33_table[int(n)] = c

    self.label33_table = label33_table

    for k, v in full_label_table.items():
      if v in label33_table.values():
        key = label33_table.keys()[label33_table.values().index(v)]
        lookup[k] = key
        print (k, v), "->", (key, v)
      else:
        lookup[k] = 0

    #print lookup

    self.lookup = lookup
        
    data_path="../data/VOCdevkit/VOC2010/JPEGImages/"
    annot_path="../data/PASCALCONTEXT/trainval"
    
    trainannotlist = glob.glob(annot_path + '/*.mat')
    trainimglist = []

    for filepath in trainannotlist:
      filename = os.path.splitext(os.path.basename(filepath))[0]
      imgpath = os.path.join(data_path, filename + ".jpg")
      trainimglist.append(imgpath)
      
    datapairs = zip(trainimglist, trainannotlist)

    self.datapairs = datapairs
    self._size = len(datapairs)


  def shuffle(self):
    random.shuffle(self.datapairs)

  @property
  def size(self):
    return self._size

  def getPair(self, idx):
    (jpegpath, labelpath) = self.datapairs[idx]
    filename = os.path.basename(jpegpath)

    # load jpeg
    _img = Image.open(jpegpath)
    w = _img.size[0]
    h = _img.size[1]
    img = np.array(_img)
    _img.close()

    # load label data
    label = scipy.io.loadmat(labelpath)['LabelMap']
    #label = np.array(map(lambda x: self.lookup[x], label))
    label = np.fromiter([self.lookup[x] for x in label.reshape(-1)], np.uint16).reshape(h,w)
    
    label_set = set(label.reshape(-1))
    #print "Labels:", map(lambda x:self.label33_table[x], label_set)
    #print "Color:", map(lambda x:hidden_palette[x], label_set)

    # resize
    max_size = self.max_img_size
    if h <= max_size and w <= max_size:
      # do nothing
      return (filename, img, label)

    if h > w:
      new_w = max_size*w/h
      new_h = max_size
    else:
      new_w = max_size
      new_h = max_size*h/w

    print "need to resize (%d, %d) to (%d, %d)" % (h, w, new_h, new_w)
    
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    
    return (filename, img, label)

def idx2onehot(tensor, nclass):

  shape = tf.shape(tensor)
  onehot = tf.constant(np.eye(nclass, dtype=np.float32))
  ret = tf.nn.embedding_lookup(onehot, tensor)
  #ret = tf.reshape(ret, shape=tf.pack([shape[0], shape[1], shape[2], FLAGS.nclass]))

  return ret

def print_shape(tensor, name):
  tensor = tf.Print(tensor, [tf.shape(tensor)[:2], tf.shape(tensor)[2:]], message=name+":")
  return tensor


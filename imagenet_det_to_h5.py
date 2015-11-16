import json, os, glob, random, argparse, time
from Queue import Queue
from threading import Thread, Lock
import xml.etree.ElementTree as ET

import h5py

import numpy as np
import matplotlib.pyplot as plt

from scipy.misc import imread, imresize
from scipy.io import loadmat


parser = argparse.ArgumentParser()
parser.add_argument('--det_train_dir', default='data/imagenet/ILSVRC2014_DET_train')
parser.add_argument('--det_val_dir', default='data/imagenet/ILSVRC2013_DET_val')
parser.add_argument('--devkit_dir', default='data/imagenet/ILSVRC2014/ILSVRC2014_devkit')
parser.add_argument('--train_bbox_dir', default='data/imagenet/ILSVRC2014_DET_bbox_train')
parser.add_argument('--val_bbox_dir', default='data/imagenet/ILSVRC2013_DET_bbox_val')
parser.add_argument('--output_h5_file', default='data/ilsvrc14-det.h5')
parser.add_argument('--output_json_file', default='data/ilsvrc14-det.json')
parser.add_argument('--num_synsets', default=200, type=int)
parser.add_argument('--num_workers', default=10, type=int)
parser.add_argument('--image_size', default=224, type=int)
parser.add_argument('--debug_max_images', default=-1, type=int)
args = parser.parse_args()


# Load info about synsets from the meta_det.mat file.
def get_synset_info():
  meta_det_path = os.path.join(args.devkit_dir, 'data/meta_det.mat')
  print 'Loading meta_det from: ', meta_det_path
  meta_det = loadmat(meta_det_path)

  # This will be a list of dicts, each with the following three fields:
  # - det_id: Integer from 1 to 200 giving ID of the synset for detection data
  # - wnid: String giving WordNet ID of the synset
  # - name: String giving human-readable name of the synset
  synset_info = []

  # The meta_det struct contains info about 815 synsets including parents and
  # children of the 200 synsets actually used for detection. We only want to
  # grab info about the first 200.
  for i in xrange(args.num_synsets):
    synset = meta_det['synsets'][0, i]
    det_id = int(synset[0][0, 0])
    wnid = str(synset[1][0])
    assert synset[2].shape == (1,)
    name = str(synset[2][0])
    synset_info.append({
      'det_id': det_id,
      'wnid': wnid,
      'name': name,
    })
  return synset_info


# Now that we have info about all detection synsets, we need to read
# the text files in the det_lists directory to figure out which images
# correspond to each synset.
# This mutates synset_info and returns a set of all image ids.
def read_det_lists(synset_info):
  det_lists_dir = os.path.join(args.devkit_dir, 'data/det_lists')

  def read_image_ids(filename):
    with open(filename, 'r') as f:
      return [line.strip() for line in f]

  all_image_ids = set()

  for i, synset in enumerate(synset_info):
    if (i + 1) % 50 == 0:
      print 'Starting synset %d / %d' % (i + 1, len(synset_info))
    det_id = synset['det_id']
    pos_file = os.path.join(det_lists_dir, 'train_pos_%d.txt' % det_id)
    neg_file = os.path.join(det_lists_dir, 'train_neg_%d.txt' % det_id)
    part_file = os.path.join(det_lists_dir, 'train_part_%d.txt' % det_id)

    synset['pos_image_ids'] = read_image_ids(pos_file)
    synset['neg_image_ids'] = read_image_ids(neg_file)
    synset['part_image_ids'] = read_image_ids(part_file)

    all_image_ids.update(synset['pos_image_ids'])
    all_image_ids.update(synset['neg_image_ids'])
    all_image_ids.update(synset['part_image_ids'])

  # According to the devkit readme.txt, there should be 456567 images in the
  # detection training set. Make sure we found them all.
  assert len(all_image_ids) == 456567
  
  return all_image_ids


def build_dicts(synset_info, all_image_ids):
  image_id_to_idx = {image_id: idx for idx, image_id in enumerate(all_image_ids)}
  idx_to_image_id = {idx: image_id for idx, image_id in enumerate(all_image_ids)}

  name_to_det_id = {s['name']: s['det_id'] for s in synset_info}
  det_id_to_name = {s['det_id']: s['name'] for s in synset_info}
  
  wnid_to_det_id = {s['wnid']: s['det_id'] for s in synset_info}
  det_id_to_wnid = {s['det_id']: s['wnid'] for s in synset_info}

  dicts = {
    'image_id_to_idx': image_id_to_idx,
    'idx_to_image_id': idx_to_image_id,
    'name_to_det_id': name_to_det_id,
    'det_id_to_name': det_id_to_name,
    'wnid_to_det_id': wnid_to_det_id,
    'det_id_to_wnid': det_id_to_wnid,
  }
  return dicts


def add_train_label_arrays(synset_info, all_image_ids, dicts, h5_file):
  image_id_to_idx = dicts['image_id_to_idx']
  idx_to_image_id = dicts['idx_to_image_id']

  num_train_images = len(all_image_ids)
  train_labels = np.zeros((num_train_images, args.num_synsets), dtype=np.uint8)
  train_hardnegs = np.zeros((num_train_images, args.num_synsets), dtype=np.uint8)

  for synset in synset_info:
    det_id = synset['det_id']
    for image_id in synset['pos_image_ids']:
      idx = image_id_to_idx[image_id]
      train_labels[idx, det_id - 1] = 1
    for image_id in synset['neg_image_ids']:
      idx = image_id_to_idx[image_id]
      train_hardnegs[idx, det_id - 1] = 1

  h5_file.create_dataset('train_labels', data=train_labels)
  h5_file.create_dataset('train_hardnegs', data=train_hardnegs)

  
def add_images(idx_to_image_id, image_dir, dset_name, h5_file):
  num_images = len(idx_to_image_id)
  dset_size = (num_images, 3, args.image_size, args.image_size)
  if args.debug_max_images > 0:
    dset_size = (min(dset_size[0], args.debug_max_images),) + dset_size[1:]
  dset = h5_file.create_dataset(dset_name, dset_size, dtype=np.uint8)

  q = Queue()
  lock = Lock()
  for idx, image_id in sorted(idx_to_image_id.iteritems()):
  # for i, (image_id, idx) in enumerate(image_id_to_idx.iteritems()):
    if args.debug_max_images > 0 and idx >= args.debug_max_images: break
    filename = os.path.join(image_dir, '%s.JPEG' % image_id)
    q.put((idx, filename))

  def worker():
    images_read = 0
    while True:
      idx, filename = q.get()
      img = imread(filename)
      # Handle grayscale
      if img.ndim == 2:
        img = img[:, :, None][:, :, [0, 0, 0]]
      try:
        img = imresize(img, (args.image_size, args.image_size))
      except ValueError as e:
        lock.acquire()
        print 'Error reading file ', filename
        lock.release()
        continue
      # Swap RGB to BGR
      img = img[:, :, [2, 1, 0]]
      images_read += 1
      
      lock.acquire()
      if images_read % 50 == 0:
        pass
        # print 'worker has read %d images; queue has %d remaining' % (images_read, q.qsize())
      # print img.shape, img.sum(), 'writing to ', idx
      dset[idx] = img.transpose(2, 0, 1)  # Swap HWC -> CHW
      lock.release()
      q.task_done()
      
      if q.empty(): return

  def print_worker():
    while True:
      print 'queue has %d items remaining' % q.qsize()
      time.sleep(3)
      
  t = Thread(target=print_worker)
  t.daemon = True
  t.start()
  
  for i in xrange(args.num_workers):
    t = Thread(target=worker)
    t.daemon = True
    t.start()
  q.join()


def add_train_images(dicts, h5_file):
  add_images(dicts['idx_to_image_id'], args.det_train_dir, 'train_images', h5_file)


def add_val_images(dicts, h5_file):
  add_images(dicts['val_idx_to_image_id'], args.det_val_dir, 'val_images', h5_file)


  
# mutates dicts in-place to add val_image_id_to_idx and val_idx_to_image_id
def read_val_list(dicts):
  val_image_id_to_idx = {}
  val_idx_to_image_id = {}
  
  val_list_file = os.path.join(args.devkit_dir, 'data/det_lists/val.txt')
  with open(val_list_file, 'r') as f:
    for idx, line in enumerate(f):
      image_id, _ = line.strip().split()
      val_image_id_to_idx[image_id] = idx
      val_idx_to_image_id[idx] = image_id
      # xml_file = os.path.join(VAL_BBOX_DIR, '%s.xml' % image_id)
      # det_ids = read_val_xml(xml_file)
      
  dicts['val_image_id_to_idx'] = val_image_id_to_idx
  dicts['val_idx_to_image_id'] = val_idx_to_image_id

# Input: path to an XML file, and metadata dicts
# Returns: set of det_ids for objects in that image
def read_val_xml(xml_file, dicts):
  tree = ET.parse(xml_file)
  root = tree.getroot()
  det_ids = set()
  for obj in root.findall('object'):
    wnid = obj.find('name').text
    det_id = dicts['wnid_to_det_id'][wnid]
    det_ids.add(det_id)
  return det_ids



def add_labels(idx_to_image_id, dicts, xml_dir, dset_name, h5_file):
  num_images = len(idx_to_image_id)
  if args.debug_max_images > 0:
    num_images = min(num_images, args.debug_max_images)
  labels = np.zeros((num_images, args.num_synsets), dtype=np.uint8)
  for (idx, image_id) in sorted(idx_to_image_id.iteritems()):
    if args.debug_max_images > 0 and idx >= args.debug_max_images: break
    xml_file = os.path.join(xml_dir, '%s.xml' % image_id)
    if not os.path.isfile(xml_file):
      print 'WARNING: could not find XML file %s' % xml_file
      continue
    det_ids = read_val_xml(xml_file, dicts)
    for det_id in det_ids:
      labels[idx, det_id - 1] = 1
      
  h5_file.create_dataset(dset_name, data=labels)

  
def add_train_labels(dicts, h5_file):
  idx_to_image_id = dicts['idx_to_image_id']
  add_labels(idx_to_image_id, dicts, args.train_bbox_dir, 'train_labels', h5_file)


def add_val_labels(dicts, h5_file):
  idx_to_image_id = dicts['val_idx_to_image_id']
  add_labels(idx_to_image_id, dicts, args.val_bbox_dir, 'val_labels', h5_file)
  

def add_val_labels_old(dicts, h5_file):
  print 'Building label array for val images:'
  num_val_images = len(dicts['val_image_id_to_idx'])
  
  val_labels = np.zeros((num_val_images, args.num_synsets), dtype=np.uint8)
  for i, (idx, image_id) in enumerate(dicts['val_idx_to_image_id'].iteritems()):
    if (i + 1) % 1000 == 0:
      print 'Reading XML file for val image %d / %d' % (i + 1, num_val_images)
    xml_file = os.path.join(args.val_bbox_dir, '%s.xml' % image_id)
    det_ids = read_val_xml(xml_file, dicts)
    for det_id in det_ids:
      val_labels[idx, det_id - 1] = 1
      
  h5_file.create_dataset('val_labels', data=val_labels)


def main():
  synset_info = get_synset_info()
  all_image_ids = read_det_lists(synset_info)
  dicts = build_dicts(synset_info, all_image_ids)
  read_val_list(dicts)

  # Dump data to JSON output file
  with open(args.output_json_file, 'w') as f:
    json.dump(dicts, f)

  # Dump data to HDF5 output file
  with h5py.File(args.output_h5_file, 'w') as f:
    # add_train_label_arrays(synset_info, all_image_ids, dicts, f)
    add_train_labels(dicts, f)
    add_train_images(dicts, f)
    add_val_labels(dicts, f)
    add_val_images(dicts, f)


if __name__ == '__main__':
  main()


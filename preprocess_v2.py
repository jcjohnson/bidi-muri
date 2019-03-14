import argparse, os, json
import numpy as np
import h5py
from scipy.misc import imread, imresize
from scipy.io import loadmat


parser = argparse.ArgumentParser()
parser.add_argument('--input_mat',
          default='data/benchmark_images/benchmark.mat')
parser.add_argument('--image_dir', default='data/benchmark_images/coco_images')
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--output_h5', default='data/images.h5')
parser.add_argument('--output_json', default='data/images-json.json')
args = parser.parse_args()


if __name__ == '__main__':
  data = loadmat(args.input_mat)['BENCHMARK']
  split_to_idx_to_filename = {
    'train': [],
    'test': [],
  }
  category_name_to_idx = {}

  num_images = data.shape[1]

  # Assume an equal number of training and testing images
  num_train = num_images / 2
  num_test = num_images / 2

  # Set up numpy arrays for training images and labels
  X_train_shape = (num_train, 3, args.image_size, args.image_size)
  X_train = np.zeros(X_train_shape, dtype=np.uint8)
  y_train = np.zeros(num_train, dtype=np.int32)

  # Set up numpy arrays for testing images and labels
  X_test_shape = (num_test, 3, args.image_size, args.image_size)
  X_test = np.zeros(X_test_shape, dtype=np.uint8)
  y_test = np.zeros(num_test, dtype=np.int32)

  next_train_idx = 0
  next_test_idx = 0

  for i in xrange(num_images):
    print 'Starting image %d / %d' % (i + 1, num_images)
    row = data[0, i]
    image_id = row[0][0, 0]
    category_id = row[1][0, 0]
    filename = row[4][0, 0][0]
    height, width = row[5], row[6]
    category_name = row[7][0]
    split = row[9][0]
    
    if category_name not in category_name_to_idx:
      idx = len(category_name_to_idx) + 1
      category_name_to_idx[category_name] = idx

    category_idx = category_name_to_idx[category_name]

    img_path = os.path.join(args.image_dir, filename)
    img = imread(img_path)
    img = imresize(img, (args.image_size, args.image_size))

    # Transpose and convert RGB -> BGR
    img = img.transpose(2, 0, 1)[[2, 1, 0]]

    if split == 'train':
      X_train[next_train_idx] = img
      y_train[next_train_idx] = category_idx
      next_train_idx += 1
    elif split == 'test':
      X_test[next_test_idx] = img
      y_test[next_test_idx] = category_idx
      next_test_idx += 1

  # Write data and labels to HDF5 file
  with h5py.File(args.output_h5, 'w') as f:
    f.create_dataset('X_train', data=X_train)
    f.create_dataset('y_train', data=y_train)
    f.create_dataset('X_test', data=X_test)
    f.create_dataset('y_test', data=y_test)
    
  with open(args.output_json, 'w') as f:
    data = {
      'category_name_to_idx': category_name_to_idx,
    }
    json.dump(data, f)


import argparse, json, os
import numpy as np
import h5py
from scipy.io import savemat


parser = argparse.ArgumentParser()
parser.add_argument('--input_probs', required=True)
parser.add_argument('--input_json', default='data/images-json.json')
parser.add_argument('--output_mat', required=True)
args = parser.parse_args()


if __name__ == '__main__':
  with h5py.File(args.input_probs, 'r') as f:
    train_probs = np.asarray(f['train_probs'])
    train_labels = np.asarray(f['train_labels'])
    test_probs = np.asarray(f['test_probs'])
    test_labels = np.asarray(f['test_labels'])
    train_mask = np.asarray(f['train_mask']) == 255
    
  print (train_mask == 0).sum()
    
  with open(args.input_json, 'r') as f:
    json_data = json.load(f)
    category_name_to_idx = json_data['category_name_to_idx']
    print category_name_to_idx, len(category_name_to_idx), min(category_name_to_idx.values())
    category_idx_to_name = [None] * len(category_name_to_idx)
    for name, idx in category_name_to_idx.iteritems():
      category_idx_to_name[idx - 1] = name
    print category_idx_to_name
    
  mat_struct = {
    'train_probs': train_probs,
    'train_labels': train_labels,
    'test_probs': test_probs,
    'test_labels': test_labels,
    'train_mask': train_mask,
    'category_names': np.asarray(category_idx_to_name, dtype=np.object),
  }
  savemat(args.output_mat, mat_struct)

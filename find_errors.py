import argparse, json, os
import numpy as np
import h5py
from scipy.misc import imsave

parser = argparse.ArgumentParser()
parser.add_argument('--probs_h5', required=True)
parser.add_argument('--images_h5', default='data/images-fixed.h5')
parser.add_argument('--input_json', default='data/images-json.json')
parser.add_argument('--output_dir', default='data/errors')
args = parser.parse_args()


if __name__ == '__main__':
  with open(args.input_json, 'r') as f:
    category_name_to_idx = json.load(f)['category_name_to_idx']
    category_idx_to_name = {v - 1: k.replace(' ', '_') for k, v in category_name_to_idx.iteritems()}
  
  with h5py.File(args.probs_h5, 'r') as f:
    test_probs = np.asarray(f['test_probs'])
    test_labels = np.asarray(f['test_labels']) - 1
    test_preds = test_probs.argmax(axis=1)
    print (test_labels == test_preds).mean()

  with h5py.File(args.images_h5, 'r') as f:
    for i in xrange(test_preds.shape[0]):
      if test_preds[i] != test_labels[i]:
        img = np.asarray(f['X_train'][i])[[2, 1, 0]]
        print img.shape
        pred_label = category_idx_to_name[test_preds[i]]
        gt_label = category_idx_to_name[test_labels[i]]
        filename = '%d_P_%s_A_%s.png' % (i, pred_label, gt_label)
        imsave(os.path.join(args.output_dir, filename), img)
        print 'predicted: %s; actual: %s' % (pred_label, gt_label)
      
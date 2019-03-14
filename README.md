# bidi-muri

Dependencies:
- CUDA (not sure which version)
- cuDNN (not sure which version)
- Lua Torch
- Lua packages:
  - loadcaffe
  - torch-hdf5
  - torch cuDNN bindings
  - Some other Lua packages?
- python packages:
  - numpy
  - scipy
  - h5py

Pipeline:
- Preprocess data -- not sure how this worked? Maybe preprocess_v2.py?
- Start from pretrained model from caffe model zoo (need to find / download)
- Train the model with train_v2.lua; finetunes a pretrained model
- Dump preditions from the model to an HDF5 file with extract_scores.lua or test.lua, I'm not sure which
- Convert predictions from HDF5 to matlab format with probs_to_mat.py

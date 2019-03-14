local cjson = require 'cjson'
local utils = {}

function utils.read_json(path)
  local file = io.open(path, 'r')
  local text = file:read()
  file:close()
  local info = cjson.decode(text)
  return info
end

function utils.write_json(path, j)
  cjson.encode_sparse_array(true, 2, 10)
  local text = cjson.encode(j)
  local file = io.open(path, 'w')
  file:write(text)
  file:close()
end


function utils.subtract_vgg_mean(X)
  local vgg_mean = torch.FloatTensor{103.939, 116.779, 123.68}
  local vgg_mean_exp = vgg_mean:view(1, 3, 1, 1):expandAs(X)
  return X:float():add(-1, vgg_mean_exp)
end


--[[
-- Assume X has shape (1, 3, 256, 256)
-- Take four corner crops, center crop, plus flips; returns
-- output of shape (10, 3, 224, 224)
--]]
function utils.ten_crops(X, crop_size)
  crop_size = crop_size or 224
  assert(X:size(1) == 1)
  local H, W = X:size(3), X:size(4)
  local UL = X[{{}, {}, {1, crop_size}, {1, crop_size}}]
  local UR = X[{{}, {}, {1, crop_size}, {W - crop_size + 1, W}}]
  local BL = X[{{}, {}, {H - crop_size + 1, H}, {1, crop_size}}]
  local BR = X[{{}, {}, {H - crop_size + 1, H}, {W - crop_size + 1, W}}]
  local y0, x0 = (H - crop_size) / 2, (W - crop_size) / 2
  local C = X[{{}, {}, {y0 + 1, y0 + crop_size}, {x0 + 1, x0 + crop_size}}]

  local crops = X.new(10, X:size(2), crop_size, crop_size)
  crops[1] = UL
  crops[2] = UR
  crops[3] = BL
  crops[4] = BR
  crops[5] = C
  crops[6] = image.hflip(UL[1])
  crops[7] = image.hflip(UR[1])
  crops[8] = image.hflip(BL[1])
  crops[9] = image.hflip(BR[1])
  crops[10] = image.hflip(C[1])

  return crops
end

function utils.load_data(opt)
  local f = hdf5.open(opt.input_h5, 'r')
  local dset = {}
  local X_train = f:read('/X_train'):all()
  local y_train = f:read('/y_train'):all()
  local N = X_train:size(1)
  local V = math.floor(N * opt.val_frac)
  local val_idx = torch.multinomial(torch.ones(N), V, false)
  local mask = torch.zeros(N)
  mask:indexFill(1, val_idx, 1)
  mask:add(-1)
  local train_idx = mask:nonzero():view(-1)
    
  dset.X_train_orig = X_train
  dset.y_train_orig = y_train
  dset.train_mask = mask:byte()
  dset.X_train = X_train:index(1, train_idx)
  dset.y_train = y_train:index(1, train_idx)
  dset.X_val = X_train:index(1, val_idx)
  dset.y_val = y_train:index(1, val_idx)
  dset.X_test = f:read('/X_test'):all()
  dset.y_test = f:read('/y_test'):all()
  
  -- This was only here as a sanity check to make sure that the test data
  -- was prepared in the same was as the training data; check to make sure
  -- that val and a small subset of test have similar accuracies after 100 iterations
  --[[
  local N_test = dset.X_test:size(1)
  local idx_test = torch.multinomial(torch.ones(N_test), 100, false)
  dset.X_test_sub = dset.X_test:index(1, idx_test)
  dset.y_test_sub = dset.y_test:index(1, idx_test)
  --]]
  
  return dset
end


return utils

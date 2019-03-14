require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'optim'
require 'hdf5'
require 'loadcaffe'
require 'image'
require 'cudnn'
cudnn.benchmark = true

local utils = require 'utils'


local cmd = torch.CmdLine()
cmd:option('-checkpoint', '')
cmd:option('-input_h5', 'data/images-fixed.h5')
cmd:option('-output_h5', 'data/probs.h5')
cmd:option('-gpu', 0)
local opt = cmd:parse(arg)


local function get_probs(model, X, y)
  local softmax = nn.SoftMax()
  local N, C = X:size(1), y:max()
  local all_probs = torch.zeros(N, C)
  local num_correct = 0
  print(N, C)
  for i = 1, N do
    if i % 25 == 0 then
      local msg = 'Starting %d / %d'
      print(string.format(msg, i, N))
    end
    local crops = utils.ten_crops(X[{{i, i}}])
    crops = utils.subtract_vgg_mean(crops)
    local scores = model:forward(crops:cuda())
    local probs = softmax:forward(scores:double()):mean(1)
    all_probs[i]:copy(probs)
    local _, y_pred = probs:max(2)
    if y_pred[{1, 1}] == y[i] then
      num_correct = num_correct + 1
    end
  end
  print(num_correct, N)
  local acc = num_correct / N
  return all_probs, acc
end


local function main()
  cutorch.setDevice(opt.gpu + 1)
  local checkpoint = torch.load(opt.checkpoint)
  if opt.input_h5 ~= '' then
    checkpoint.opt.input_h5 = opt.input_h5
  end
  torch.manualSeed(checkpoint.opt.seed)
  local model = checkpoint.model
  model:evaluate()
  
  local dset = utils.load_data(checkpoint.opt)
  
  local train_probs, train_acc = get_probs(model, dset.X_train_orig, dset.y_train_orig)
  local test_probs, test_acc = get_probs(model, dset.X_test, dset.y_test)
  
  print('training accuracy: ', train_acc)
  print('testing accuracy: ', test_acc)
  
  local f = hdf5.open(opt.output_h5, 'w')
  f:write('/train_probs', train_probs)
  f:write('/train_labels', dset.y_train_orig)
  f:write('/test_probs', test_probs)
  f:write('/test_labels', dset.y_test)
  f:write('/train_mask', dset.train_mask)
  f:close()
end


main()
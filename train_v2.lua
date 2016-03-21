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
cmd:option('-input_h5', 'data/images.h5')
cmd:option('-prototxt', 'data/models/vgg-16/VGG_ILSVRC_16_layers_deploy.prototxt')
cmd:option('-caffemodel', 'data/models/vgg-16/VGG_ILSVRC_16_layers.caffemodel')
cmd:option('-reg', 0)
cmd:option('-batch_size', 64)
cmd:option('-val_frac', 0.1)
cmd:option('-check_acc_every', 10)
cmd:option('-num_iterations', 1000)
cmd:option('-checkpoint_name', 'data/checkpoint')
cmd:option('-checkpoint_every', 100)
local opt = cmd:parse(arg)


local function load_data(opt)
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
    
  dset.X_train = X_train:index(1, train_idx)
  dset.y_train = y_train:index(1, train_idx)
  dset.X_val = X_train:index(1, val_idx)
  dset.y_val = y_train:index(1, val_idx)
  dset.X_test = f:read('/X_test'):all()
  dset.y_test = f:read('/y_test'):all()
  return dset
end


local function build_model(opt)
  local cnn = loadcaffe.load(opt.prototxt, opt.caffemodel)

  -- TODO: These values are hardcoded for VGG-16 right now
  local first_part = nn.Sequential()
  for i = 1, 38 do
    first_part:add(cnn:get(i))
  end
  local second_part = nn.Linear(4096, 10)
  
  local model = nn.Sequential()
  model:add(first_part)
  model:add(second_part)
  cnn = nil
  collectgarbage()

  cudnn.convert(model, cudnn)
  return model
end


local function get_minibatch(X, y, batch_size)
  local mask = torch.LongTensor(batch_size):random(X:size(1))
  local X_batch = X:index(1, mask)
  local y_batch = y:index(1, mask)
  return X_batch, y_batch
end


--[[
-- Assume X has shape (1, 3, 256, 256)
-- Take four corner crops, center crop, plus flips; returns
-- output of shape (10, 3, 224, 224)
--]]
local function ten_crops(X, crop_size)
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


local function random_crop_flip(X)
  -- TODO: 224 hardcoded for VGG-16
  local crop_size = 224
  local H, W = X:size(3), X:size(4)

  local h, w = H - crop_size, W - crop_size
  local x0 = torch.random(H - crop_size)
  local y0 = torch.random(W - crop_size)
  local cropped = X[{{}, {}, {y0, y0 + crop_size - 1}, {x0, x0 + crop_size - 1}}]

  if torch.random(2) == 1 then
    for i = 1, cropped:size(1) do
      cropped[i] = image.hflip(cropped[i])
    end
  end

  return cropped
end


local function subtract_vgg_mean(X)
  local vgg_mean = torch.FloatTensor{103.939, 116.779, 123.68}
  local vgg_mean_exp = vgg_mean:view(1, 3, 1, 1):expandAs(X)
  return X:float():add(-1, vgg_mean_exp)
end


local function check_accuracy(model, X, y)
  model:evaluate()
  local num_correct = 0
  for i = 1, X:size(1) do
    local crops = ten_crops(X[{{i, i}}])
    local scores = model:forward(crops:cuda()):mean(1)
    local _, y_pred = scores:max(2)
    if y_pred[{1, 1}] == y[i] then
      num_correct = num_correct + 1
    end
  end
  local acc = num_correct / X:size(1)
  print(string.format('%d / %d = %f', num_correct, X:size(1), acc))
  return acc
end


-- Build model and criterion
local dset = load_data(opt)
local model = build_model(opt)
local crit = nn.CrossEntropyCriterion()
model:cuda()
crit:cuda()
model:training()

local params, grad_params = model:get(2):getParameters()
local state = {learningRate=1e-3}
local finetuning = false

-- Callback for optim
local function f(w)
  assert(w == params)

  -- Get a minibatch of training data, with data augmentation
  local X_train, y_train = dset.X_train, dset.y_train
  local X_batch, y_batch = get_minibatch(X_train, y_train, opt.batch_size)
  X_batch = subtract_vgg_mean(X_batch)
  
  X_batch = random_crop_flip(X_batch)
  X_batch, y_batch = X_batch:cuda(), y_batch:cuda()

  -- Run the model forward
  local scores = model:forward(X_batch)
  local loss = crit:forward(scores, y_batch)

  -- Run the model backward
  grad_params:zero()
  local grad_scores = crit:backward(scores, y_batch)
  if finetuning then
    model:backward(X_batch, grad_scores)
  else
    local first_out = model:get(1).output
    model:get(2):backward(first_out, grad_scores)
  end

  grad_params:add(opt.reg, params)

  return loss, grad_params
end


local loss_history = {}
local val_acc_history = {}
local train_acc_history = {}
local acc_ts = {}

for t = 1, opt.num_iterations do
  local _, loss = optim.adam(f, params, state)
  table.insert(loss_history, loss)
  local msg = 'Iteration %d, loss = %f'
  print(string.format(msg, t, loss[1]))
  
  if t % opt.check_acc_every == 0 then
    -- Grab a random subset of training images to check training accuracy
    local p = torch.ones(dset.X_train:size(1))
    local idx = torch.multinomial(p, dset.X_val:size(1), false)
    local X_train_sub = dset.X_train:index(1, idx)
    local y_train_sub = dset.y_train:index(1, idx)
    
    print 'Checking training accuracy ...'
    local train_acc = check_accuracy(model, X_train_sub, y_train_sub)
    print 'Checking validation accuracy ...'
    local val_acc = check_accuracy(model, dset.X_val, dset.y_val)
    print(train_acc, val_acc)
    table.insert(val_acc_history, val_acc)
    table.insert(train_acc_history, train_acc)
    table.insert(acc_ts, t)
    model:training()
  end
  
  if t % opt.checkpoint_every == 0 then
    -- First save a JSON checkpoint
    local checkpoint = {
      opt=opt,
      loss_history=loss_history,
      train_acc_history=train_acc_history,
      val_acc_history=val_acc_history,
    }
    local name = string.format(opt.checkpoint_name .. '.json')
    utils.write_json(name, checkpoint)
    
    -- Now save a torch checkpoint
    name = string.format(opt.checkpoint_name .. '.t7')
    checkpoint.model = model
    torch.save(name, checkpoint)
  end
end


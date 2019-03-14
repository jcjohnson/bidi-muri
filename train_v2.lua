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
cmd:option('-input_h5', 'data/images-fixed.h5')
cmd:option('-prototxt', 'data/models/vgg-16/VGG_ILSVRC_16_layers_deploy.prototxt')
cmd:option('-caffemodel', 'data/models/vgg-16/VGG_ILSVRC_16_layers.caffemodel')
cmd:option('-reg', 0)
cmd:option('-batch_size', 32)
cmd:option('-val_frac', 0.1)
cmd:option('-check_acc_every', 100)
cmd:option('-num_iterations', 1000)
cmd:option('-checkpoint_name', 'data/checkpoint')
cmd:option('-checkpoint_every', 100)
cmd:option('-learning_rate', 1e-3)
cmd:option('-finetune_after', 200)
cmd:option('-ft_learning_rate', 1e-4)
cmd:option('-seed', 0)
cmd:option('-random_crop', 1)
cmd:option('-random_flip', 1)
cmd:option('-lr_decay_every', 5000)
cmd:option('-lr_decay_factor', 0.1)
cmd:option('-dropout', 0.8)
cmd:option('-gpu', 0)

-- ResNet options
cmd:option('-resnet_model', '')
cmd:option('-resnet_layer', 10)

local opt = cmd:parse(arg)



local function build_model(opt)
  local first_part = nn.Sequential()
  local second_part = nn.Sequential()
  if opt.resnet_model == '' then
    local cnn = loadcaffe.load(opt.prototxt, opt.caffemodel)

    -- TODO: These values are hardcoded for VGG-16 right now;
    -- skipping the last dropout layer, will insert into second part
    for i = 1, 37 do
      first_part:add(cnn:get(i))
    end
    second_part:add(nn.Dropout(opt.dropout))
    second_part:add(nn.Linear(4096, 14))
  else
    local cnn = torch.load(opt.resnet_model)
    for i = 1, opt.resnet_layer do
      first_part:add(cnn:get(i))
    end
    second_part:add(nn.Dropout(opt.dropout))
    second_part:add(nn.Linear(2048, 14))
  end
  
  local model = nn.Sequential()
  model:add(first_part)
  model:add(second_part)
  cnn = nil
  collectgarbage()

  cudnn.convert(model, cudnn)
  print(model)
  return model
end


local function get_minibatch(X, y, batch_size)
  local mask = torch.LongTensor(batch_size):random(X:size(1))
  local X_batch = X:index(1, mask)
  local y_batch = y:index(1, mask)
  return X_batch, y_batch
end


local function random_crop_flip(X)
  -- TODO: 224 hardcoded for VGG-16
  local crop_size = 224
  local H, W = X:size(3), X:size(4)

  local h, w = H - crop_size, W - crop_size
  local x0 = torch.random(H - crop_size)
  local y0 = torch.random(W - crop_size)
  if opt.random_crop == 0 then
    x0, y0 = 16, 16
  end
  local cropped = X[{{}, {}, {y0, y0 + crop_size - 1}, {x0, x0 + crop_size - 1}}]

  if opt.random_flip == 1 and torch.random(2) == 1 then
    for i = 1, cropped:size(1) do
      cropped[i] = image.hflip(cropped[i])
    end
  end

  return cropped
end


local function check_accuracy(model, X, y)
  local softmax = nn.SoftMax():cuda()
  model:evaluate()
  local num_correct = 0
  for i = 1, X:size(1) do
    local crops = utils.ten_crops(X[{{i, i}}])
    crops = utils.subtract_vgg_mean(crops)
    local scores = model:forward(crops:cuda())
    local probs = softmax:forward(scores):mean(1)
    local _, y_pred = probs:max(2)
    if y_pred[{1, 1}] == y[i] then
      num_correct = num_correct + 1
    end
  end
  local acc = num_correct / X:size(1)
  print(string.format('%d / %d = %f', num_correct, X:size(1), acc))
  return acc
end

torch.manualSeed(opt.seed)
cutorch.setDevice(opt.gpu + 1)

-- Build model and criterion
local dset = utils.load_data(opt)
local model = build_model(opt)
local crit = nn.CrossEntropyCriterion()
model:cuda()
crit:cuda()
model:training()

local params, grad_params = model:get(2):getParameters()
local state = {learningRate=opt.learning_rate}
local finetuning = false

-- Callback for optim
local function f(w)
  assert(w == params)

  -- Get a minibatch of training data, with data augmentation
  local X_train, y_train = dset.X_train, dset.y_train
  local X_batch, y_batch = get_minibatch(X_train, y_train, opt.batch_size)
  X_batch = utils.subtract_vgg_mean(X_batch)
  
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
  local msg = 'Iteration %d, loss = %f, lr = %f'
  print(string.format(msg, t, loss[1], state.learningRate))
  
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
      acc_ts=acc_ts,
    }
    local name = string.format(opt.checkpoint_name .. '.json')
    utils.write_json(name, checkpoint)
    
    -- Now save a torch checkpoint
    name = string.format(opt.checkpoint_name .. '.t7')
    model:clearState()
    checkpoint.model = model
    torch.save(name, checkpoint)
  end
  
  if t == opt.finetune_after then
    print 'Starting to finetune'
    finetuning = true
    params, grad_params = model:getParameters()
    state = {learningRate=opt.ft_learning_rate}
  end
  
  if t % opt.lr_decay_every == 0 then
    print 'decaying learning rate'
    local new_lr = state.learningRate * opt.lr_decay_factor
    state = {learningRate=new_lr}
  end
end


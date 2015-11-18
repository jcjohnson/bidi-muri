require 'torch'
require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'loadcaffe'

require 'DataLoader'
require 'MulticlassLogisticCriterion'
require 'optim_updates'
local eval_utils = require 'eval_utils'
local utils = require 'utils'


cmd = torch.CmdLine()
cmd:option('-model', 'vgg-16')
cmd:option('-h5_file', 'data/ilsvrc14-50k-det.h5')
cmd:option('-json_file', 'data/ilsvrc14-50k-det.json')
cmd:option('-num_trainval', 1000)
cmd:option('-backend', 'cudnn')
cmd:option('-batch_size', 32)
cmd:option('-learning_rate', 1e-4)
cmd:option('-finetune_learning_rate', 1e-6)
cmd:option('-optim_alpha', 0.9)
cmd:option('-num_iterations', -1)
cmd:option('-finetune_after', 5000)
cmd:option('-eval_trainval_every', 100)
cmd:option('-save_checkpoint_every', 500)
cmd:option('-checkpoint_name', 'data/checkpoint.t7')
cmd:option('-gpu', 0)


local function build_model(opt, num_classes)
  local proto_file, caffemodel_file, model_dir
  local chop_idx, output_dim
  if opt.model == 'vgg-16' then
    model_dir = 'data/models/vgg-16'
    proto_file = 'VGG_ILSVRC_16_layers_deploy.prototxt'
    caffemodel_file = 'VGG_ILSVRC_16_layers.caffemodel'
    chop_idx = 38
    output_dim = 4096
  else
    error('Unrecognized model ' .. opt.model)
  end
  proto_file = paths.concat(model_dir, proto_file)
  caffemodel_file = paths.concat(model_dir, caffemodel_file)
  local cnn = loadcaffe.load(proto_file, caffemodel_file, opt.backend)

  local cnn_part = nn.Sequential()
  for i = 1, chop_idx do
    cnn_part:add(cnn:get(i))
  end

  local new_part = nn.Sequential()
  new_part:add(nn.Linear(output_dim, num_classes))

  local model = nn.Sequential()
  model:add(cnn_part)
  model:add(new_part)
  model:cuda()
  return model
end


local function eval_split(split, model, loader)
  model:evaluate()
  loader:resetSplit(split)
  local scores_list, labels_list = {}, {}
  for i = 1, 50 do
    local images, labels = loader:getBatch(split)
    local scores = model:forward(images:cuda())
    -- cat only works on DoubleTensor so we have to cast to double
    table.insert(scores_list, scores:double():clone())
    table.insert(labels_list, labels:double():clone())
  end
  local all_scores = torch.cat(scores_list, 1):float()
  local all_labels = torch.cat(labels_list, 1):byte()
  local stats = eval_utils.multiclass_eval(all_scores, all_labels)
  return stats
end


local function main()
  local opt = cmd:parse(arg)
  cutorch.setDevice(opt.gpu + 1)
  
  local model = build_model(opt, 200)
  local crit = nn.MulticlassLogisticCriterion():cuda()
  local loader = DataLoader(opt)

  local params, grad_params = model:get(2):getParameters()
  local finetuning = false
  print(params:nElement())

  local iter = 1
  local optim_state = {}
  local train_loss_history = {}
  local trainval_stat_history = {}
  while true do
    if iter % opt.eval_trainval_every == 1 then
      print('Evaluating trainval performance:')
      local stats = eval_split('trainval', model, loader)
      print(stats)
      trainval_stat_history[iter] = stats
      model:training()
    end
    
    local images, labels, info = loader:getBatch('train')
    images = images:cuda()

    local scores = model:forward(images)
    local loss = crit:forward(scores, labels)
    local dscores = crit:backward(scores, labels)
    
    local learning_rate
    if finetuning then
      model:backward(images, dscores)
      learning_rate = opt.finetune_learning_rate
    else
      learning_rate = opt.learning_rate
      model:get(2):backward(model:get(1).output, dscores)
    end
    
    print(iter, loss, info.start_idx, info.end_idx)
    sgdm(params, grad_params, learning_rate, opt.optim_alpha, optim_state)
    table.insert(train_loss_history, loss)

    if iter > 1 and iter % opt.save_checkpoint_every == 1 then
      print('Saving checkpoint to ' .. opt.checkpoint_name)
      local checkpoint = {}
      checkpoint.train_loss_history = train_loss_history
      checkpoint.trainval_stat_history = trainval_stat_history
      utils.write_json(opt.checkpoint_name .. '.json', checkpoint)
      checkpoint.model = model
      checkpoint.opt = opt
      torch.save(opt.checkpoint_name, checkpoint)
    end
    
    if iter == opt.finetune_after then
      print 'Starting to finetune now'
      optim_state = {}
      params, grad_params = model:getParameters()
      finetuning = true
    end

    if iter == opt.num_iterations then break end
    iter = iter + 1
  end
end

main()


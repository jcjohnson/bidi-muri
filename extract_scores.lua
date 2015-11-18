require 'torch'
require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'loadcaffe'
require 'hdf5'

require 'DataLoader'
require 'MulticlassLogisticCriterion'


cmd = torch.CmdLine()
cmd:option('-h5_file', 'data/ilsvrc14-50k-det.h5')
cmd:option('-json_file', 'data/ilsvrc14-50k-det.json')
cmd:option('-num_trainval', 1000)
cmd:option('-batch_size', 32)
cmd:option('-split', 'val')
cmd:option('-output_h5_file', 'data/scores.h5')
cmd:option('-backend', 'cudnn')
cmd:option('-checkpoint', 'data/checkpoint.t7')
cmd:option('-gpu', 0)


local function main()
  local opt = cmd:parse(arg)
  cutorch.setDevice(opt.gpu + 1)
  local checkpoint = torch.load(opt.checkpoint)
  local model = checkpoint.model
  local loader = DataLoader(opt)
  
  local scores_list = {}
  local labels_list = {}
  model:evaluate()
  while true do
    local images, labels, info = loader:getBatch(opt.split)
    print(info.start_idx, info.end_idx, loader.epoch_counters[opt.split])
    local scores = model:forward(images:cuda()):double():clone()
    table.insert(scores_list, scores)
    table.insert(labels_list, labels:double():clone())
    if loader.epoch_counters[opt.split] > 1 then
      break
    end
  end
  local all_scores = torch.cat(scores_list, 1):float()
  local all_labels = torch.cat(labels_list, 1):byte()
  
  local f = hdf5.open(opt.output_h5_file, 'w')
  f:write('/val_scores', all_scores)
  f:write('/val_labels', all_labels)
  f:close()
end

main()

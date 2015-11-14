require 'torch'
require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'loadcaffe'

require 'DataLoader'

cmd = torch.CmdLine()
cmd:option('-model', 'vgg-16')
cmd:option('-h5_file', 'data/ilsvrc14-det.h5')
cmd:option('-json_file', 'data/ilsvrc14-det.json')
cmd:option('-backend', 'cudnn')
cmd:option('-batch_size', 64)


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


local function main()
  local opt = cmd:parse(arg)
  local model = build_model(opt, 200)
  local loader = DataLoader(opt)

  for t = 1, 5 do
    local images, labels = loader:getBatch('train')
    print(images:size())
    print(labels:size())
    print(images:sum(), labels:sum())
  end
end

main()


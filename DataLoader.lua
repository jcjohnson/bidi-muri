require 'hdf5'
local cjson = require 'cjson'

local DataLoader = torch.class('DataLoader')


local function read_json(path)
  local file = assert(io.open(path, 'r'))
  local text = file:read()
  file:close()
  return cjson.decode(text)
end


function DataLoader:__init(opt)
  assert(opt.h5_file, 'Must provide h5_file')
  assert(opt.json_file, 'Must provide json_file')
  assert(opt.batch_size, 'Must provide batch_size')
  assert(opt.num_trainval, 'Must provide num_trainval')
  self.h5_file = hdf5.open(opt.h5_file, 'r')
  self.json_file = opt.json_file
  self.json_dicts = read_json(opt.json_file)
  self.batch_size = opt.batch_size
  self.split_idxs = {
    train = 1,
    trainval = 1,
    val = 1,
  }
  local train_size = self.h5_file:read('/train_images'):dataspaceSize()
  self.num_channels = train_size[2]
  self.image_height = train_size[3]
  self.image_width = train_size[4]
  self.start_idxs = {
    train = 1,
    trainval = train_size[1] - opt.num_trainval + 1,
    val = 1,
  }
  self.end_idxs = {
    train = train_size[1] - opt.num_trainval,
    trainval = train_size[1],
    val = self.h5_file:read('/val_images'):dataspaceSize()[1],
  }
  for k, sidx in pairs(self.start_idxs) do
    print(string.format('Split %s starts at %d and ends at %d',
          k, sidx, self.end_idxs[k]))
  end
  self.epoch_counters = {
    train = 1,
    trainval = 1,
    val = 1,
  }
  self.num_classes = self.h5_file:read('/train_labels'):dataspaceSize()[2]
  self.vgg_mean = torch.FloatTensor{103.939, 116.779, 123.68}
end


function DataLoader:resetSplit(split)
  self.split_idxs[split] = self.start_idxs[split]
  self.epoch_counters[split] = 1
end


function DataLoader:getBatch(split)
  local image_dset, label_dset
  if split == 'train' or split == 'trainval' then
    image_dset = '/train_images'
    label_dset = '/train_labels'
  elseif split == 'val' then
    image_dset = '/val_images'
    label_dset = '/val_labels'
  end
  local start_idx = self.split_idxs[split]
  local end_idx = math.min(start_idx + self.batch_size - 1, self.end_idxs[split])

  local images = self.h5_file:read(image_dset):partial(
                    {start_idx, end_idx},
                    {1, self.num_channels},
                    {1, self.image_height},
                    {1, self.image_width})
  local labels = self.h5_file:read(label_dset):partial(
                    {start_idx, end_idx},
                    {1, self.num_classes})
  
  -- Cast images to float and subtract VGG mean
  images = images:float()
  images:add(-1, self.vgg_mean:view(1, 3, 1, 1):expandAs(images))

  local info = {
    image_dset = image_dset,
    start_idx = start_idx,
    end_idx = end_idx,
  }
  
  -- Advance iterator for this split, maybe rolling back to the start
  self.split_idxs[split] = end_idx + 1
  if self.split_idxs[split] > self.end_idxs[split] then
    self.split_idxs[split] = self.start_idxs[split]
    self.epoch_counters[split] = self.epoch_counters[split] + 1
  end

  return images, labels, info
end


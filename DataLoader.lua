require 'hdf5'

local DataLoader = torch.class('DataLoader')

function DataLoader:__init(opt)
  assert(opt.h5_file, 'Must provide h5_file')
  assert(opt.json_file, 'Must provide json_file')
  assert(opt.batch_size, 'Must provide batch_size')
  self.h5_file = hdf5.open(opt.h5_file, 'r')
  self.json_file = opt.json_file
  self.batch_size = opt.batch_size
  self.split_idxs = {
    train = 1,
    val = 1,
  }
  local train_size = self.h5_file:read('/train_images'):dataspaceSize()
  self.num_channels = train_size[2]
  self.image_height = train_size[3]
  self.image_width = train_size[4]
  self.start_idxs = {
    train = 1,
    val = 1,
  }
  self.end_idxs = {
    train = train_size[1],
    val = self.h5_file:read('/val_images'):dataspaceSize()[1],
  }
  self.epoch_counters = {
    train = 1,
    val = 1,
  }
  self.num_classes = self.h5_file:read('/train_labels'):dataspaceSize()[2]
end


function DataLoader:resetSplit(split)
  self.split_idxs[split] = self.start_idxs[split]
  self.epoch_counters[split] = 1
end


function DataLoader:getBatch(split)
  local image_dset, label_dset
  if split == 'train' then
    image_dset = '/train_images'
    label_dset = '/train_labels'
  elseif split == 'val' then
    image_dst = '/val_images'
    label_dset = '/val_labels'
  end
  local start_idx = self.split_idxs[split]
  local end_idx = math.min(start_idx + self.batch_size, self.end_idxs[split])

  local images = self.h5_file:read(image_dset):partial(
                    {start_idx, end_idx},
                    {1, self.num_channels},
                    {1, self.image_height},
                    {1, self.image_width})
  local labels = self.h5_file:read(label_dset):partial(
                    {start_idx, end_idx},
                    {1, self.num_classes})
  
  -- TODO: cast images to float and subtract VGG mean

  -- Advance iterator for this split, maybe rolling back to the start
  self.split_idxs[split] = end_idx + 1
  if self.split_idxs[split] > self.end_idxs[split] then
    self.split_idxs[split] = self.start_idxs[split]
    self.epoch_counters[split] = self.epoch_counters[split] + 1
  end

  return images, labels
end


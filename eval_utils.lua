require 'torch'

local eval_utils = {}


--[[
- scores: Tensor of shape N x C giving class scores for N images
- labels: ByteTensor of shape N x C giving labels
--]]
function eval_utils.multiclass_eval(scores, labels)
  local _, idx = scores:max(2)
  local top_hit = labels:gather(2, idx)
  local stats = {}
  stats.recall_at_1 = top_hit:sum() / labels:sum()
  stats.recall_at_1_ubound = top_hit:nElement() / labels:sum()
  return stats
end


return eval_utils
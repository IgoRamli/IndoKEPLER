import torch
from datasets import load_dataset

class RoundRobinDataset(torch.utils.data.Dataset):
  def __init__(self, dataset_a, dataset_b, label_a, label_b, max_len=-1):
    self.ds_a = dataset_a
    self.ds_b = dataset_b
    self.label_a = label_a
    self.label_b = label_b
    self.max_len = max_len

  def __getitem__(self, idx):
    idx_a = idx%len(self.ds_a)
    idx_b = idx%len(self.ds_b)
    return {
      self.label_a: self.ds_a[idx_a],
      self.label_b: self.ds_b[idx_b]
    }

  def __len__(self):
    if self.max_len != -1:
      return self.max_len
    return max(len(self.ds_a), len(self.ds_b))


import torch
from datasets import load_dataset

class RoundRobinDataset(torch.utils.data.Dataset):
  def __init__(self, dataset_mlm, dataset_ke, max_len=-1):
    self.mlm = dataset_mlm
    self.ke = dataset_ke
    self.max_len = max_len

  def __getitem__(self, idx):
    idx_mlm = idx%len(self.mlm)
    idx_ke = idx%len(self.ke)
    return {
      'mlm': self.mlm[idx_mlm]['input_ids'],
      'heads': self.ke[idx_ke]['heads'],
      'tails': self.ke[idx_ke]['tails'],
      'heads_r': self.ke[idx_ke]['heads_r'],
      'tails_r': self.ke[idx_ke]['tails_r'],
      'nHeads': self.ke[idx_ke]['nHeads'],
      'nTails': self.ke[idx_ke]['nTails'],
      'relations': self.ke[idx_ke]['relations']
    }

  def __len__(self):
    if self.max_len != -1:
      return self.max_len
    return max(len(self.ds_a), len(self.ds_b))


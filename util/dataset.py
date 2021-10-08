import torch
from datasets import load_dataset

class RoundRobinDataset(torch.utils.data.Dataset):
	def __init__(self, dataset_a, dataset_b, label_a, label_b):
		self.ds_a = dataset_a
		self.ds_b = dataset_b

	def __getitem__(self, idx):
		idx_a = idx%len(self.ds_a)
		idx_b = idx%len(self.ds_b)
		return {
			self.label_a: self.ds_a[idx_a],
			self.label_b: self.ds_b[idx_b]
		}

	def __len__(self):
		return max(len(self.idx_a), len(self.idx_b))

mlm = load_dataset('./IndoWiki/indowiki_text.txt', split='train')
ke = load_from_disk('ke_dataset')

train_dataset = RoundRobinDataset(mlm, ke['train'], 'mlm', 'ke')
test_dataset = RoundRobinDataset(mlm, ke['test'], 'mlm', 'ke')


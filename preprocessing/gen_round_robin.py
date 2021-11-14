import numpy as np
import math
import gc
import os

from argparse import ArgumentParser
from datasets import load_from_disk, Dataset, concatenate_datasets, DatasetDict
from tqdm import tqdm
from pathlib import Path

parser = ArgumentParser(description='Map entity ids into their tokenized form')
parser.add_argument('--mlm-dir', default='mlm', help='Directory of MLM dataset')
parser.add_argument('--ke-dir', default='ke', help='Directory of KE dataset')
parser.add_argument('--out-dir', default='indokepler', help='Output directory to save datasets')

def gen_round_robin(dataset_mlm, dataset_ke, max_len=-1):
  total_len = max(len(dataset_mlm), len(dataset_ke))
  dataset_dict = {
    'mlm': [],
    'heads': [],
    'tails': [],
    'heads_r': [],
    'tails_r': [],
    'nHeads': [],
    'nTails': [],
    'relations': []
  }

  for idx in tqdm(range(total_len)):
    idx_mlm = idx%len(dataset_mlm)
    idx_ke = idx%len(dataset_ke)

    dataset_dict['mlm'].append(dataset_mlm[idx_mlm]['input_ids'])
    dataset_dict['heads'].append(dataset_ke[idx_ke]['heads'])
    dataset_dict['tails'].append(dataset_ke[idx_ke]['tails'])
    dataset_dict['heads_r'].append(dataset_ke[idx_ke]['heads_r'])
    dataset_dict['tails_r'].append(dataset_ke[idx_ke]['tails_r'])
    dataset_dict['nHeads'].append(dataset_ke[idx_ke]['nHeads'])
    dataset_dict['nTails'].append(dataset_ke[idx_ke]['nTails'])
    dataset_dict['relations'].append(dataset_ke[idx_ke]['relations'])
  return dataset_dict
  
def fetch_dataset(path):
  splits = os.listdir(path)
  dataset_dict = {
    split: load_from_disk('{}/{}'.format(path, split)) \
    for split in splits
  }
  return DatasetDict(dataset_dict)
  
def fetch_sharded_dataset(path):
  splits = os.listdir(path)
  dataset_dict = {
    split: concatenate_datasets([
      load_from_disk('{}/{}/{}'.format(path, split, shard_dir)) \
      for shard_dir in os.listdir('{}/{}'.format(path, split)) \
      if os.path.isdir('{}/{}/{}'.format(path, split, shard_dir))
    ])
    for split in splits
  }
  return DatasetDict(dataset_dict)
  
def compile_dataset(mlm_data, ke_data):
  dataset_dict = {
    'train': Dataset.from_dict(gen_round_robin(mlm_data['train'], ke_data['train'])),
    'valid': Dataset.from_dict(gen_round_robin(mlm_data['valid'], ke_data['valid'])),
    'test': Dataset.from_dict(gen_round_robin(mlm_data['test'], ke_data['test']))
  }
  return DatasetDict(dataset_dict)

if __name__ == '__main__':
  args = parser.parse_args()

  print('| Fetching MLM dataset')
  mlm = fetch_dataset(args.mlm_dir)
  print('| Fetching KE datasets')
  ke = fetch_sharded_dataset(args.ke_dir)
  print('| Combining datasets in round robin fashion')
  dataset = compile_dataset(mlm, ke)
  dataset.save_to_disk(args.out_dir)

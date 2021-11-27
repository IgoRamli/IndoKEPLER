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
parser.add_argument('--max-len', default=-1, type=int, help='Maximum number of data to be stored. Use this to sample the first bunch of data')
parser.add_argument('--num-proc', default=1, type=int, help='Number of threads to process dataset')

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

def compile_dataset(args, mlm_data, ke_data):
  pair_with_mlm = {
    split: lambda example, index : {
      'mlm': mlm_data[split][index%len(mlm_data[split])]['input_ids'],
      **example
    }
    for split in ['train', 'valid', 'test']
  }

  # Number of entities is usually smaller than number of knowledge triplets
  dataset_dict = {
    split: ke_data[split].map(pair_with_mlm[split], with_indices=True, num_proc=args.num_proc)
    for split in ['train', 'valid', 'test']
  }
  return DatasetDict(dataset_dict)

if __name__ == '__main__':
  args = parser.parse_args()

  print('| Fetching MLM dataset')
  mlm = fetch_dataset(args.mlm_dir)
  print('| Fetching KE datasets')
  ke = fetch_sharded_dataset(args.ke_dir)
  print('| Combining datasets in round robin fashion')
  dataset = compile_dataset(args, mlm, ke)
  print(f"| Final columns {dataset['train']}")
  print('| Saving dataset')
  dataset.save_to_disk(args.out_dir)
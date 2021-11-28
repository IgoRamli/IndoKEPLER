import numpy as np
import math
import gc
import os

from argparse import ArgumentParser
from datasets import load_from_disk, load_dataset
from tqdm import tqdm
from pathlib import Path

parser = ArgumentParser(description='Generate negative sampling')
parser.add_argument('entity_dir', help='Path to directory containing tokenized entity')
parser.add_argument('--data-dir', default='', help='Directory of shards to be processed')
parser.add_argument('--start', required=False, type=int, help='Index of the first shard to be processed')
parser.add_argument('--end', required=False, type=int, help='Index of the last shard to be processed (inclusive)')
parser.add_argument('--ns', default=1, type=int, help='Number of negative sampling to be generated')
parser.add_argument('--true-heads', default='true_heads.txt', help='Output file to save true heads')
parser.add_argument('--true-tails', default='true_tails.txt', help='Output file to save true tails')
parser.add_argument('--num-proc', default=1, type=int, help='Number of threads to work in parallel')
parser.add_argument('--out-dir', default='', help='Output directory to save dataset to')

TRESHOLD = 10000  # Minimum number of rows to start consider multiprocessing

def map_txt_to_dict(file):
  mapping = {}
  for line in tqdm(file):
    ids = [ int(i) for i in line.replace('\n', '').split(' ') ]
    k = (ids[0], ids[1])
    v = ids[2:]
    # print(k)
    mapping[k] = v
  return mapping

def fetch_true_triplets(heads_file, tails_file):
  print('| Fetching true heads')
  true_heads = map_txt_to_dict(heads_file)
  print('| Fetching true tails')
  true_tails = map_txt_to_dict(tails_file)
  return true_heads, true_tails

def gen_negative_sampling(old_ds, new_ds, args, entities, true_head, true_tail):
  print('| Loading dataset "{}"'.format(old_ds))
  dataset = load_from_disk(old_ds)

  print('| Generating negative samples')
  print('| Training size: ({})'.format(dataset.num_rows))
  print('| Negative samples for each triplet: {}'.format(args.ns))
  def gen_triplets_with_negative_sampling(row):
    head = row['head']
    relation = row['relation']
    tail = row['tail']

    ns = {}
    for mode in ['head-batch', 'tail-batch']:
      negative_sample_list = []
      while len(negative_sample_list) < args.ns:
        negative_sample = np.random.choice(entities, size=args.ns*2)
        if mode == 'head-batch':
          mask = np.in1d(
            negative_sample, 
            true_head[(relation, tail)], 
            assume_unique=True, 
            invert=True
          )
        elif mode == 'tail-batch':
          mask = np.in1d(
            negative_sample, 
            true_tail[(head, relation)], 
            assume_unique=True, 
            invert=True
          )
        negative_sample = negative_sample[mask]
        negative_sample_list += negative_sample.tolist()
        mask, negative_sample = None, None  # Optimize RAM usage
      ns[mode] = negative_sample_list[:args.ns]
      negative_sample_list = None  # Optimize RAM usage
      assert(len(ns[mode]) == args.ns)
    for false_head in ns['head-batch']:
      assert(false_head not in true_head[(relation, tail)])
    for false_tail in ns['tail-batch']:
      assert(false_tail not in true_tail[(head, relation)])

    return {
      'nHeads': ns['head-batch'],
      'nTails': ns['tail-batch'],
    }
  if dataset.num_rows >= TRESHOLD:
    dataset = dataset.map(gen_triplets_with_negative_sampling,
                          num_proc=args.num_proc)
  else:
    dataset = dataset.map(gen_triplets_with_negative_sampling)
  print('| Saving dataset to "{}"'.format(new_ds))
  Path(new_ds).mkdir(parents=True, exist_ok=True)
  dataset.save_to_disk(new_ds)
  print('| Dataset saved')

def get_num_entities(txt_file):
  return sum(1 for i in open(txt_file, 'r'))

if __name__ == '__main__':
  args = parser.parse_args()

  with open(args.true_heads, 'r') as heads_file:
    with open(args.true_tails, 'r') as tails_file:
      true_heads, true_tails = fetch_true_triplets(heads_file,
                                                   tails_file)

  ds_mapping = []
  ds_splits = os.listdir(args.data_dir)
  for split in ds_splits:
    old_ds = '{}/{}'.format(args.data_dir, split)
    new_ds = '{}/{}'.format(args.out_dir, split)
    for slice in range(args.start, args.end+1):
      mapping = ('{}/{}'.format(old_ds, slice), '{}/{}'.format(new_ds, slice))
      ds_mapping.append(mapping)

  print('| The following datasets will be processed:')
  for ds in ds_mapping:
    print('| - {}\t=>\t{}'.format(ds[0], ds[1]))

  for ds in ds_mapping:
    print('| Getting entity candidates')
    entities = []
    with open(f'{ds[0]}/../entities.txt', 'r') as f:
      for line in f:
        entities.append(int(line))
    print('| {} entities found'.format(len(entities)))
    gen_negative_sampling(ds[0], ds[1], args, entities, true_heads, true_tails)
    gc.collect()
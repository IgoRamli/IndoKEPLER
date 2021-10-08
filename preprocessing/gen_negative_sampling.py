import numpy as np
import math
import gc

from argparse import ArgumentParser
from datasets import load_from_disk
from tqdm import tqdm
from pathlib import Path

parser = ArgumentParser(description='Generate negative sampling')
parser.add_argument('entity_dir', help='Path to directory containing tokenized entity')
parser.add_argument('--data-dirs', default='', help='Directories of dataset, separated by commas (,)')
parser.add_argument('--split', required=False, type=int, help='Number of splits in dataset')
parser.add_argument('--start', required=False, type=int, help='Index of the first shard to be processed')
parser.add_argument('--end', required=False, type=int, help='Index of the last shard to be processed (inclusive)')
parser.add_argument('--ns', default=1, type=int, help='Number of negative sampling to be generated')
parser.add_argument('--true-heads', default='true_heads.txt', help='Output file to save true heads')
parser.add_argument('--true-tails', default='true_tails.txt', help='Output file to save true tails')
parser.add_argument('--num-proc', default=1, type=int, help='Number of threads to work in parallel')
parser.add_argument('--out-dirs', default='', help='Output directory to save dataset to')

def gen_triplets_with_negative_sampling(row,
                                        num_entities,
                                        true_head,
                                        true_tail):
  head = row['head']
  relation = row['relation']
  tail = row['tail']

  ns = {}
  num_rows = num_entities
  for mode in ['head-batch', 'tail-batch']:
    negative_sample_list = []
    while len(negative_sample_list) < args.ns:
      negative_sample = np.random.randint(num_rows, size=args.ns*2)
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

  return {
    'nHeads': ns['head-batch'],
    'nTails': ns['tail-batch'],
  }

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

def gen_negative_sampling(old_ds, new_ds, args, num_entities, true_head, true_tail):
  print('| Loading dataset "{}"'.format(old_ds))
  dataset = load_from_disk(old_ds)

  print('| Generating negative samples')
  print('| Training size: ({})'.format(dataset.num_rows))
  print('| Negative samples for each triplet: {}'.format(args.ns))
  gen_ns_func = lambda row: gen_triplets_with_negative_sampling(row,
                                                                num_entities,
                                                                true_head,
                                                                true_tail)
  dataset = dataset.map(gen_ns_func, num_proc=args.num_proc)
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
  print('| Getting number of entities')
  num_entities = load_from_disk(args.entity_dir).num_rows
  print('| {} entities found'.format(num_entities))

  ds_mapping = []
  for ds, new_ds in zip(args.data_dirs.split(','), args.out_dirs.split(',')):
    for split in range(args.start, args.end+1):
      mapping = ('{}/{}'.format(ds, split), '{}/{}'.format(new_ds, split))
      ds_mapping.append(mapping)

  print('| The following datasets will be processed:')
  for ds in ds_mapping:
    print('| - {}\t=>\t{}'.format(ds[0], ds[1]))

  for ds in ds_mapping:
    gen_negative_sampling(ds[0], ds[1], args, num_entities, true_heads, true_tails)
    gc.collect()
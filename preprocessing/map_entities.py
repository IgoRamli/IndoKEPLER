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
parser.add_argument('--num-proc', default=1, type=int, help='Number of threads to work in parallel')
parser.add_argument('--out-dirs', default='', help='Output directories to save dataset to, separated by commas (,)')

def map_entities(batch, ke_entities):
  return {
    'heads': ke_entities[batch['head']],
    'tails': ke_entities[batch['tail']],
    'relations': batch['relation'],
    'heads_r': ke_entities[batch['head']],
    'tails_r': ke_entities[batch['tail']],
    'nHeads': [ ke_entities[i] for i in batch['nHeads'] ],
    'nTails': [ ke_entities[i] for i in batch['nTails'] ],
  }

def process_dataset(old_ds, new_ds, entities):
  print('| Loading dataset "{}"'.format(old_ds))
  dataset = load_from_disk(old_ds)

  print('| Get tokenized representations for each KE entities')
  dataset = dataset.map(lambda row: map_entities(row, entities),
                        remove_columns=['head', 'relation', 'tail'],
                        num_proc=args.num_proc)
  print('| Saving dataset to "{}"'.format(new_ds))
  Path(new_ds).mkdir(parents=True, exist_ok=True)
  dataset.save_to_disk(new_ds)
  print('| Dataset saved')

if __name__ == '__main__':
  args = parser.parse_args()

  print('| Getting entities')
  entities = load_from_disk(args.entity_dir)

  ds_mapping = []
  for ds, new_ds in zip(args.data_dirs.split(','), args.out_dirs.split(',')):
    for split in range(args.start, args.end+1):
      mapping = ('{}/{}'.format(ds, split), '{}/{}'.format(new_ds, split))
      ds_mapping.append(mapping)

  print('| The following datasets will be processed:')
  for ds in ds_mapping:
    print('| - {}\t=>\t{}'.format(ds[0], ds[1]))

  for ds in ds_mapping:
    process_dataset(ds[0], ds[1], entities)
    gc.collect()

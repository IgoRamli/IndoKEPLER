import numpy as np
import math
import gc
import os

from argparse import ArgumentParser
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from pathlib import Path

parser = ArgumentParser(description='Map entity ids into their tokenized form')
parser.add_argument('entity_dir', help='Path to directory containing tokenized entity')
parser.add_argument('--data-dir', default='step-2', help='Directory of dataset to be processed')
parser.add_argument('--start', required=False, type=int, help='Index of the first shard to be processed')
parser.add_argument('--end', required=False, type=int, help='Index of the last shard to be processed (inclusive)')
parser.add_argument('--num-proc', default=1, type=int, help='Number of processes to be generated')
parser.add_argument('--out-dir', default='', help='Output directory to save datasets')


def process_dataset(args, old_ds, new_ds, entities):
  print('| Loading dataset "{}"'.format(old_ds))
  dataset = load_from_disk(old_ds)

  print('| Get tokenized representations for each KE entities')
  def map_entities(batch):
  #  print('Map triplet ({},{},{})'.format(batch['head'], batch['tail'], batch['relation']))
  #  print('{} {}'.format(batch['nHeads'], batch['nTails']))
    mapped_dict = {
      'heads': entities[batch['head']],
      'tails': entities[batch['tail']],
      'relations': batch['relation'],
      'heads_r': entities[batch['head']],
      'tails_r': entities[batch['tail']],
      'nHeads': [ entities[int(i)] for i in batch['nHeads'] ],
      'nTails': [ entities[int(i)] for i in batch['nTails'] ],
    }
    return mapped_dict
  dataset = dataset.map(map_entities,
                        remove_columns=['head', 'relation', 'tail'])
  print('| Saving dataset to "{}"'.format(new_ds))
  Path(new_ds).mkdir(parents=True, exist_ok=True)
  dataset.save_to_disk(new_ds)
  print('| Dataset saved')

if __name__ == '__main__':
  args = parser.parse_args()

  print('| Getting entities')
  entities = load_from_disk(args.entity_dir)
  entity_ids = []
  def extract_entities(row):
    entity_ids.append(row)
    return None
  entities.map(extract_entities)

  ds_mapping = []
  ds_splits = os.listdir(args.data_dir)
  for ds_split in ds_splits:
    old_ds = '{}/{}'.format(args.data_dir, ds_split)
    new_ds = '{}/{}'.format(args.out_dir, ds_split)
    for split in range(args.start, args.end+1):
      mapping = ('{}/{}'.format(old_ds, split), '{}/{}'.format(new_ds, split))
      ds_mapping.append(mapping)

  print('| The following datasets will be processed:')
  for ds in ds_mapping:
    print('| - {}\t=>\t{}'.format(ds[0], ds[1]))

  for ds in ds_mapping:
    process_dataset(args, ds[0], ds[1], entity_ids)
    gc.collect()

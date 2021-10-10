import os

from argparse import ArgumentParser
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

parser = ArgumentParser(description='Tokenize KE entities')
parser.add_argument('split', type=int, help='Number of splits to be made')
parser.add_argument('--data-dir', default='./csv', help='Directory of CSV files to be split')
parser.add_argument('--out-dir', default='./step-1', help='Output directory to save dataset to')

def slice_dataset(in_file, out_dir, split):
  dataset = load_dataset('csv',
                         split='train',
                         data_files=in_file)
  print('| Sharding')
  datasets = [ dataset.shard(num_shards=split, index=i) for i in tqdm(range(split)) ]
  print('| Saving to disk')
  for idx, shard in tqdm(enumerate(datasets), total=split):
    sub_dir = '{}/{}'.format(out_dir, idx+1)
    shard.save_to_disk(sub_dir)

  print('| Listing entities')
  entities = set()
  def fetch_entities(row):
    h, t = row['head'], row['tail']
    entities.add(h)
    entities.add(t)
  dataset.map(fetch_entities)
  print('| {} entities found'.format(len(entities)))
  with open('{}/entities.txt'.format(out_dir), 'w+') as ent_file:
    for entity in tqdm(entities):
      ent_file.write(str(entity) + '\n')

if __name__ == '__main__':
  args = parser.parse_args()
  in_files = [ '{}/{}'.format(args.data_dir, i) for i in  os.listdir(args.data_dir) ]
  out_dirs = [ '{}/{}'.format(args.out_dir, '.'.join(i.split('.')[:-1])) for i in  os.listdir(args.data_dir) ]

  print('| Slicing the following datasets into {} directories each'.format(args.split))
  for in_file, out_dir in zip(in_files, out_dirs):
    print('| - {}\t=>\t{}'.format(in_file, out_dir))

  for in_file, out_dir in zip(in_files, out_dirs):
    print('| Slicing "{}"'.format(in_file))
    slice_dataset(in_file, out_dir, args.split)

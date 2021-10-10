import os

from argparse import ArgumentParser
from util.functions import fetch_shards, compile_dataset

parser = ArgumentParser(description='Combine MLM and KE datasets in a round robin fashion')
parser.add_argument('mlm_dir', help='Path to tokenized MLM datasets')
parser.add_argument('ke_dir', help='Path to tokenized KE datasets')
parser.add_argument('--num-proc', default=1, type=int, help='Number of processes to be generated')
parser.add_argument('--out-dir', default='indokepler', help='Output directory to save processed datasets')

if __name__ == '__main__':
  args = parser.parse_args()

  mlm_splits = set(os.listdir(args.mlm_dir))
  ke_splits = set(os.listdir(args.ke_dir))
  splits = mlm_splits.intersection(ke_splits)

  for split in splits:
    print('| Fetching MLM dataset')
    mlm = fetch_shards('{}/{}'.format(args.mlm_dir, split))
    print('| Fetching KE datasets')
    ke = fetch_shards('{}/{}'.format(args.ke_dir, split))
    print('| Combining datasets in round robin fashion')
    dataset = compile_dataset(mlm, ke)

    print('| Saving dataset')
    dataset.save_to_disk('{}/{}'.format(args.out_dir, split))
    print('| "{}" Dataset saved to "{}"'.format(split, args.out_dir))

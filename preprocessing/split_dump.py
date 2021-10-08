from argparse import ArgumentParser
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

parser = ArgumentParser(description='Tokenize KE entities')
parser.add_argument('split', type=int, help='Number of splits to be made')
parser.add_argument('--files', default='', help='List of files to be split, separated by commas (,)')
parser.add_argument('--out-dir', default='', help='Output directory to save dataset to, separated by commas (,)')

def split_dataset(in_file, out_dir, split):
	dataset = load_dataset('csv',
		                   split='train',
		                   data_files=in_file)
	print('Sharding')
	datasets = [ dataset.shard(num_shards=split, index=i) for i in tqdm(range(split)) ]
	print('Saving to disk')
	for idx, shard in tqdm(enumerate(datasets), total=split):
		sub_dir = '{}/{}'.format(out_dir, idx+1)
		shard.save_to_disk(sub_dir)

if __name__ == '__main__':
	args = parser.parse_args()
	in_files = args.files.split(',')
	out_dirs = args.out_dir.split(',')

	if len(in_files) != len(out_dirs):
		raise ValueError('Number of directories must match number of files')

	print('| Splitting the following datasets into directories')
	for in_file, out_dir in zip(in_files, out_dirs):
		print('| - {}\t=>\t{}'.format(in_file, out_dir))

	for in_file, out_dir in zip(in_files, out_dirs):
		print('| Splitting "{}"'.format(in_file))
		split_dataset(in_file, out_dir, args.split)

import csv
import os

from argparse import ArgumentParser
from tqdm import tqdm

parser = ArgumentParser(description='Generate true heads and tails from list of triplets')
parser.add_argument('--data-dir', default='./csv', help='Directory that contains CSV files to be processed')
parser.add_argument('--out-file-heads', default='true_heads.txt', help='Output file to save true heads')
parser.add_argument('--out-file-tails', default='true_tails.txt', help='Output file to save true tails')

def generate_true_heads_and_tails(row, true_head, true_tail):
  head = int(row['head'])
  relation = int(row['relation'])
  tail = int(row['tail'])

  if (head, relation) not in true_tail:
    true_tail[(head, relation)] = []
  true_tail[(head, relation)].append(tail)

  if (relation, tail) not in true_head:
    true_head[(relation, tail)] = []
  true_head[(relation, tail)].append(head)

if __name__ == '__main__':
	args = parser.parse_args()
	files = [ '{}/{}'.format(args.data_dir, i) for i in  os.listdir(args.data_dir) ]

	true_head = {}
	true_tail = {}
	print('| True heads and tails will be generated from the following files')
	for file in files:
		print('| -', file)

	for file in files:
		print('| Processing "{}"'.format(file))
		with open(file, 'r') as f:
			csv_reader = csv.DictReader(f)
			for row in tqdm(csv_reader):
				generate_true_heads_and_tails(row,
																		  true_head,
																		  true_tail)
	print('| Saving true heads')
	with open(args.out_file_heads, 'w+') as out_heads:
		for k, v in tqdm(true_head.items()):
			flat_row = [ k[0], k[1] ] + v
			line = ' '.join([ str(i) for i in flat_row ])
			out_heads.write(line + '\n')

	print('| Saving true tails')
	with open(args.out_file_tails, 'w+') as out_tails:
		for k, v in tqdm(true_tail.items()):
			flat_row = [ k[0], k[1] ] + v
			line = ' '.join([ str(i) for i in flat_row ])
			out_tails.write(line + '\n')

	print('| True triplets saved')
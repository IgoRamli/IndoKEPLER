import json
from tqdm import tqdm
import csv
from pathlib import Path
from argparse import ArgumentParser

parser = ArgumentParser(description='Convert entities and triplet mappings into a format process-able by preprocess.py')
parser.add_argument('text_file', help='Text file containing entity descriptions')
parser.add_argument('train_file', help='Text file containing training data')
parser.add_argument('valid_file', help='Text file containing validation data')
parser.add_argument('test_file', help='Text file containing testing data')
parser.add_argument('--out-entity-file', default='./entities.txt', help='File name of entity descriptions')
parser.add_argument('--out-dir', default='./csv', help='Directory to store CSV files')

def get_entity_mappings(in_file, out_file):
  q_id = {}
  for idx, line in tqdm(enumerate(in_file)):
    data = line.split('\t')
    assert len(data) >= 2
    assert data[0].startswith('Q')
    desc = '\t'.join(data[1:]).strip()
    q_id[data[0]] = idx
    out_file.write(desc + '\n')
  return q_id


class TripletConverter():
  def __init__(self, q_id):
    self.q_id = q_id
    self.p_id = {}

  def convert_triplets(self, in_file, out_file):
    triplet_fields = ['head', 'relation', 'tail']
    writer = csv.DictWriter(out_file,
                            fieldnames=triplet_fields)
    writer.writeheader()

    for line in tqdm(in_file):
      data = line.replace('\n', '').split('\t')
      assert len(data) == 3
      if data[1] not in self.p_id:
        self.p_id[data[1]] = len(self.p_id)

      tmp = [
        self.q_id[data[0]],
        self.p_id[data[1]],
        self.q_id[data[2]]
      ]
      out_line = { k: v for k,v in zip(triplet_fields, tmp)}
      writer.writerow(out_line)


if __name__ == '__main__':
  args = parser.parse_args()

  print('| Getting ID mappings of entities')
  with open(args.text_file, 'r') as text_file:
    with open(args.out_entity_file, 'w+') as ent_file:
      q_id = get_entity_mappings(text_file, ent_file)
  print()

  Path(args.out_dir).mkdir(parents=True, exist_ok=True)
  converter = TripletConverter(q_id)
  print('| Converting training file:')
  with open(args.train_file, 'r') as in_file:
    with open(args.out_dir + '/train.csv', 'w+') as out_file:
      converter.convert_triplets(in_file, out_file)
  print()
  print('| Converting validation file:')
  with open(args.valid_file, 'r') as in_file:
    with open(args.out_dir + '/valid.csv', 'w+') as out_file:
      converter.convert_triplets(in_file, out_file)
  print()
  print('| Converting test file:')
  with open(args.test_file, 'r') as in_file:
    with open(args.out_dir + '/test.csv', 'w+') as out_file:
      converter.convert_triplets(in_file, out_file)
  print()

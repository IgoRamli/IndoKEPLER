import json
from tqdm import tqdm
import csv
from argparse import ArgumentParser

parser = ArgumentParser(description='Convert entities and triplet mappings into a format process-able by preprocess.py')
parser.add_argument('text_file', help='Text file containing entity descriptions')
parser.add_argument('train_file', help='Text file containing training data')
parser.add_argument('test_file', help='Text file containing testing data')
parser.add_argument('--cache-dir', default='./.cache', help='Directory to store intermediate files')

def get_entity_mappings(in_file, out_file):
  writer = csv.DictWriter(out_file, fieldnames=['text'])
  writer.writeheader()
  q_id = {}
  for idx, line in tqdm(enumerate(in_file)):
    data = line.split('\t')
    assert len(data) >= 2
    assert data[0].startswith('Q')
    desc = '\t'.join(data[1:]).strip()
    q_id[data[0]] = idx-1
    writer.writerow({ 'text': data[1] })
  return q_id

class TripletConverter():
  def __init__(self, q_id):
    self.q_id = q_id
    self.p_id = {}

  def convert_triplets(self, in_file, out_file):
    p_id = {}
    triplets = []

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

  print('Getting ID mappings of entities')
  with open(args.text_file, 'r') as text_file:
    with open(args.cache_dir + '/text.csv', 'w+') as text_csv:
      q_id = get_entity_mappings(text_file, text_csv)
  print()

  converter = TripletConverter(q_id)
  print('Converting train file:')
  with open(args.train_file, 'r') as in_file:
    with open(args.cache_dir + '/train.csv', 'w+') as out_file:
      converter.convert_triplets(in_file, out_file)
  print()
  print('Converting test file:')
  with open(args.test_file, 'r') as in_file:
    with open(args.cache_dir + '/test.csv', 'w+') as out_file:
      converter.convert_triplets(in_file, out_file)
  print()

from argparse import ArgumentParser
from datasets import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

parser = ArgumentParser(description='Tokenize KE entities')
parser.add_argument('text_file', help='Text file containing entity descriptions')
parser.add_argument('--tokenizer', default='distilbert-base-uncased', help='Name of tokenizer to be used')
parser.add_argument('--max-length', default=512, type=int, help='Maximum length of an entity description to be tokenized')
parser.add_argument('--num-proc', default=1, type=int, help='Number of threads to work in parallel')
parser.add_argument('--out-dir', default='tokenized_text', help='Output directory to save dataset to')

if __name__ == '__main__':
  args = parser.parse_args()
  print('| Fetching tokenizer')
  tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

  def tokenize_text(samples):
    tokenized = tokenizer(samples['text'],
                          max_length=args.max_length,
                          padding='max_length',
                          truncation=True)
    return tokenized['input_ids']

  print('| Tokenizing dataset')
  entities = []
  with open(args.text_file, 'r') as f:
    for line in tqdm(f):
      entities.append(tokenize_text(f))
  ds = Dataset.from_dict({ 'input_ids': entities })
  print('| Saving dataset')
  ds.save_to_disk(args.out_dir)
  print('| Tokenized entities saved to "{}"'.format(args.out_dir))

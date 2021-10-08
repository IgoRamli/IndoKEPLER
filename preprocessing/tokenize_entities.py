from argparse import ArgumentParser
from transformers import AutoTokenizer
from datasets import load_dataset

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
    return { 'input_ids': tokenized['input_ids'] }

  print('| Reading dataset')
  raw_dataset = load_dataset('csv',
                             split='train',
                             data_files=args.text_file)
  print('| Tokenizing KE entities')
  print('| Number of entities: {}'.format(raw_dataset.num_rows))
  entities = raw_dataset.map(tokenize_text,
                             batched=True,
                             num_proc=args.num_proc,
                             remove_columns=['text'])
  print('| Saving dataset')
  entities.save_to_disk(args.out_dir)
  print('| Tokenized entities saved to "{}"'.format(args.out_dir))

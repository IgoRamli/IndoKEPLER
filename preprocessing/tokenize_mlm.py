from argparse import ArgumentParser
from transformers import AutoTokenizer
from datasets import load_dataset

parser = ArgumentParser(description='Tokenize MLM dataset')
parser.add_argument('dataset_name_or_path', help='Text file containing entity descriptions. If multiple positional arguments are needed, separated by commas (,)')
parser.add_argument('--tokenizer', default='distilbert-base-uncased', help='Name of tokenizer to be used')
parser.add_argument('--size', default='', help='MAximum number of data to be preprocessed')
parser.add_argument('--num-proc', default=1, type=int, help='Number of threads to work in parallel')
parser.add_argument('--out-dir', default='tokenized_text', help='Output directory to save dataset to')

if __name__ == '__main__':
  args = parser.parse_args()
  print('| Fetching tokenizer')
  tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

  def tokenize_text(samples):
    return tokenizer(samples['text'], padding='max_length', truncation=True)

  print('| Reading dataset')
  pos_args = args.dataset_name_or_path.split(',')
  raw_dataset = load_dataset(*pos_args,
                             split='train[:{}]'.format(args.size),
                             data_files=args.text_file)
  print('| Tokenizing dataset')
  print('| Number of entities: {}'.format(raw_dataset.num_rows))
  dataset = raw_dataset.map(tokenize_text,
                             batched=True,
                             num_proc=args.num_proc,
                             remove_columns=['text'])
  print('| Saving dataset')
  dataset.save_to_disk(args.out_dir)
  print('| Tokenized dataset saved to "{}"'.format(args.out_dir))

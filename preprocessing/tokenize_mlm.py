from argparse import ArgumentParser
from transformers import AutoTokenizer
from datasets import load_dataset
from pathlib import Path

parser = ArgumentParser(description='Tokenize MLM dataset')
parser.add_argument('entities_file', help='Text file containing Wikipedia content.')
parser.add_argument('--tokenizer', default='distilbert-base-uncased', help='Name of tokenizer to be used')
parser.add_argument('--block-size', default=512, type=int, help='Length of one MLM dataset')
parser.add_argument('--num-proc', default=1, type=int, help='Number of threads to work in parallel')
parser.add_argument('--valid-size', default=0.01, type=float, help='Proportion of dataset that should belong to validation set')
parser.add_argument('--test-size', default=0.01, type=float, help='Proportion of dataset that should belong to testing set')
parser.add_argument('--out-dir', default='tokenized_text', help='Output directory to save dataset to')

if __name__ == '__main__':
  args = parser.parse_args()
  print('| Fetching tokenizer')
  tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

  def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= args.block_size:
      total_length = (total_length // args.block_size) * args.block_size
    # Split by chunks of max_len.
    result = {
      k: [t[i : i + args.block_size] for i in range(0, total_length, args.block_size)]
      for k, t in concatenated_examples.items()
    }
    return result
  def tokenize_text(samples):
    return tokenizer(samples['text'], return_special_tokens_mask=True)

  print('| Reading dataset')
  raw_dataset = load_dataset('text', split='train', data_files=args.entities_file)
  print('| Tokenizing dataset')
  print('| Number of entities: {}'.format(raw_dataset.num_rows))
  dataset = raw_dataset.map(tokenize_text,
                            batched=True,
                            num_proc=args.num_proc,
                            remove_columns=['text'])
  print('| Grouping dataset into chunks of size {}'.format(args.block_size))
  dataset = dataset.map(group_texts,
                        batched=True,
                        num_proc=args.num_proc)
  
  print('| Splitting datasets to train, validation, and test sets')
  train_valid_test = dataset.train_test_split(test_size=args.test_size)
  train_valid = train_valid_test['train'].train_test_split(test_size=args.valid_size)
  train = train_valid['train']
  valid = train_valid['test']
  test = train_valid_test['test']

  print('| Final dataset size:')
  print('|  - Training size:', train.num_rows)
  print('|  - Validation size:', valid.num_rows)
  print('|  - Testing size:', test.num_rows)

  # Flatten indices to save memory (See https://discuss.huggingface.co/t/saving-dataset-in-the-current-state-without-cache/5892/9)
  train.flatten_indices()
  valid.flatten_indices()
  test.flatten_indices()

  print('| Saving dataset')
  Path(args.out_dir).mkdir(parents=True, exist_ok=True)
  print('|  - Training set')
  train.save_to_disk(args.out_dir + '/train')
  print('|  - Validation set')
  valid.save_to_disk(args.out_dir + '/valid')
  print('|  - Testing set')
  test.save_to_disk(args.out_dir + '/test')
  print('| Tokenized datasets saved to "{}"'.format(args.out_dir))

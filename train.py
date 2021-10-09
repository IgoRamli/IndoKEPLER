from os import listdir
from transformers import HfArgumentParser, TrainingArguments

from util.functions import prepare_trainer

parser = HfArgumentParser(TrainingArguments)
parser.add_argument('--model-name-or-path', required=True, help='Model and tokenizer name')
parser.add_argument('--mlm-dirs', required=True, help='Path to MLM dataset directories, separated by commas (,)')
parser.add_argument('--ke-dirs', required=True, help='Path to KE dataset directories, separated by commas (,)')

if __name__ == '__main__':
    training_args, args = parser.parse_args_into_dataclasses()
    print(args)
    trainer = prepare_trainer(training_args, args)
    if len(listdir(training_args.output_dir)) > 0:
      trainer.train(resume_from_checkpoint=True)
    else:
      trainer.train()
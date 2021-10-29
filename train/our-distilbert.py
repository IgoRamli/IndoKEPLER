import torch
from os import listdir
from transformers import HfArgumentParser, TrainingArguments

from utils.functions import prepare_trainer

parser = HfArgumentParser(TrainingArguments)
parser.add_argument('--model-name-or-path', required=True, help='Model and tokenizer name')
parser.add_argument('--dataset', required=True, help='Path to dataset directories')

if __name__ == '__main__':
  training_args, args = parser.parse_args_into_dataclasses()
  training_args.label_names = []  # run validation no matter what
  print(args)
  
  trainer = prepare_trainer(training_args, args)
  if len(listdir(training_args.output_dir)) > 0:
    trainer.train(resume_from_checkpoint=True)
  else:
    trainer.train()

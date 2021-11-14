import torch
from os import listdir
from transformers import HfArgumentParser, TrainingArguments

from utils.functions import (
  prepare_trainer_for_our_distilbert,
  prepare_trainer_for_indokepler,
)

parser = HfArgumentParser(TrainingArguments)
parser.add_argument('--model-name-or-path', required=True, help='Model and tokenizer name')
parser.add_argument('--dataset', required=True, help='Path to dataset directories')
parser.add_argument('--type', required=True, choices=['our_distilbert', 'indokepler'], help='Type of model to be trained')

if __name__ == '__main__':
  training_args, args = parser.parse_args_into_dataclasses()
  training_args.label_names = []  # run validation no matter what
  print(args)
  
  if args.type == 'our_distilbert':
    trainer = prepare_trainer_for_our_distilbert(training_args, args)
  else:
    trainer = prepare_trainer_for_indokepler(training_args, args)
    
  if len(listdir(training_args.output_dir)) > 0:
    trainer.train(resume_from_checkpoint=True)
  else:
    trainer.train()
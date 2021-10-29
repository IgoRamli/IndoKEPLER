import torch
from os import listdir
from transformers import HfArgumentParser, TrainingArguments

from utils.functions import prepare_trainer

parser = HfArgumentParser(TrainingArguments)
parser.add_argument('--model-name-or-path', required=True, help='Model and tokenizer name')
parser.add_argument('--mlm-dir', required=True, help='Path to MLM dataset directories')
parser.add_argument('--ke-dir', required=True, help='Path to KE dataset directories')

def list_gpu():
  # If there's a GPU available...
  if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
  # If not...
  else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

if __name__ == '__main__':
  training_args, args = parser.parse_args_into_dataclasses()
  training_args.label_names = []  # run validation no matter what
  print(args)
  
  trainer = prepare_trainer(training_args, args)
  if len(listdir(training_args.output_dir)) > 0:
    trainer.train(resume_from_checkpoint=True)
  else:
    trainer.train()
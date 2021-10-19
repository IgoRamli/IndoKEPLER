import os
import torch
from datasets import load_from_disk, concatenate_datasets, DatasetDict
from transformers.data.data_collator import DataCollatorForLanguageModeling, default_data_collator
from transformers import (
  AutoTokenizer,
  AutoModel,
  AutoModelForMaskedLM,
  Trainer
)
from model.modeling_kepler import KeplerModel
from model.configuration_kepler import KeplerConfig

from .data_collator import DataCollatorForKnowledgeEmbedding
from .dataset import RoundRobinDataset


def get_data_collator(tokenizer):
  def custom_data_collator(features):
    mlm_data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)
    return {
      'mlm': mlm_data_collator(features['mlm']),
      'heads': default_data_collator(features['heads']),
      'tails': default_data_collator(features['tails']),
      'heads_r': default_data_collator(features['heads_r']),
      'tails_r': default_data_collator(features['tails_r']),
      'nHeads': default_data_collator(features['nHeads']),
      'nTails': default_data_collator(features['nTails']),
      'relations': torch.IntTensor(features['relations'])
    }
  return custom_data_collator

def load_model(model_name_or_path):
  if 'distilbert' in model_name_or_path:
    print('| Loading model "{}"'.format(model_name_or_path))
    mlm_model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
    base_model = mlm_model.distilbert

    config = KeplerConfig(embedding_size=base_model.config.dim)
    model = KeplerModel(config, base_model, mlm_model)
    return model
  else:
    raise NotImplementedError('Only distilbert models can be loaded into KEPLER')

def load_tokenizer(model_name_or_path):
  tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
  return tokenizer
  
def fetch_dataset(path):
  splits = os.listdir(path)
  return {
    split: load_from_disk('{}/{}'.format(path, split)) \
    for split in splits
  }
  
def fetch_sharded_dataset(path):
  splits = os.listdir(path)
  dataset_dict = {
    split: concatenate_datasets([
      load_from_disk('{}/{}/{}'.format(path, split, shard_dir)) \
      for shard_dir in os.listdir('{}/{}'.format(path, split)) \
      if os.path.isdir('{}/{}/{}'.format(path, split, shard_dir))
    ])
    for split in splits
  }
  return DatasetDict(dataset_dict)
  
def compile_dataset(mlm_data, ke_data):
  dataset_dict = {
    'train': RoundRobinDataset(mlm_data['train'], ke_data['train'], 'mlm_data', 'ke_data'),
    'valid': RoundRobinDataset(mlm_data['valid'], ke_data['valid'], 'mlm_data', 'ke_data'),
    'test': RoundRobinDataset(mlm_data['test'], ke_data['test'], 'mlm_data', 'ke_data')
  }
  return DatasetDict(dataset_dict)

def prepare_trainer(training_args, args):
  tokenizer = load_tokenizer(args.model_name_or_path)

  print('| Fetching MLM dataset')
  mlm = fetch_dataset(args.mlm_dir)
  print('| Fetching KE datasets')
  ke = fetch_sharded_dataset(args.ke_dir)
  print('| Combining datasets in round robin fashion')
  dataset = compile_dataset(mlm, ke)

  trainer = Trainer(
    model=load_model(args.model_name_or_path),
    args=training_args,
    data_collator=get_data_collator(tokenizer),
    train_dataset=dataset['train'],
    eval_dataset=dataset['valid'])
  return trainer

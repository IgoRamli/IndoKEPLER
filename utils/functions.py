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

def prepare_trainer_for_indokepler(training_args, args):
  tokenizer = load_tokenizer(args.model_name_or_path)
  
  dataset = load_from_disk(args.dataset)

  trainer = Trainer(
    model=load_model(args.model_name_or_path),
    args=training_args,
    data_collator=get_data_collator(tokenizer),
    train_dataset=dataset['train'],
    eval_dataset=dataset['valid'])
  return trainer
  

def prepare_trainer_for_our_distilbert(training_args, args):
  tokenizer = load_tokenizer(args.model_name_or_path)
  
  dataset = load_from_disk(args.dataset)
  model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path)

  trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer),
    train_dataset=dataset['train'],
    eval_dataset=dataset['valid'])
  return trainer

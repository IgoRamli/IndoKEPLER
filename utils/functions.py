import os
import torch
from datasets import load_from_disk, concatenate_datasets, DatasetDict, Dataset
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
  def slice(data, column, new_col_name=None):
    return [ d[column] for d in data ]

  def custom_data_collator(features):
    mlm_data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)
    new_features = default_data_collator(features)
    mlm_feature = mlm_data_collator(slice(features, 'mlm'))
    new_features['mlm'] = mlm_feature
    # new_features['mlm_input_ids'] = torch.tensor(mlm_feature['input_ids'])
    # new_features['mlm_labels'] = torch.tensor(mlm_feature['labels'])

    encoding_size = new_features['heads'].shape[1]
    new_features['nHeads'] = new_features['nHeads'].view((-1, encoding_size))
    new_features['nTails'] = new_features['nTails'].view((-1, encoding_size))
    return new_features
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
  dataset_dict = {
    split: load_from_disk('{}/{}'.format(path, split)) \
    for split in splits
  }
  return DatasetDict(dataset_dict)

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
  
  dataset = fetch_dataset(args.dataset)
  model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path)

  trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer),
    train_dataset=dataset['train'],
    eval_dataset=dataset['valid'])
  return trainer
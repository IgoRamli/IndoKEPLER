from datasets import load_from_disk, concatenate_datasets
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers import AutoModel, AutoTokenizer, training_args
from transformers import (
  AutoTokenizer,
  AutoModel,
  Trainer
)
from model.modeling_kepler import KeplerModel
from model.configuration_kepler import KeplerConfig

from .data_collator import DataCollatorForKnowledgeEmbedding
from .dataset import RoundRobinDataset


def get_data_collator(tokenizer):
  def custom_data_collator(features):
    mlm_data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)
    ke_data_collator = DataCollatorForKnowledgeEmbedding(tokenizer=tokenizer, ns_size=1)
    return {
      'mlm_data': mlm_data_collator([ feature['mlm_data']['input_ids'] for feature in features ]),
      'ke_data': ke_data_collator([ feature['ke_data'] for feature in features ])
    }
  return custom_data_collator

def load_model(model_name_or_path):
  base_model = AutoModel.from_pretrained(model_name_or_path)

  config = KeplerConfig(embedding_size=base_model.config.dim)
  model = KeplerModel(config, base_model)
  return model

def load_tokenizer(model_name_or_path):
  tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
  return tokenizer

def fetch_shards(dirs):
  shards = [ load_from_disk(d) for d in dirs ]
  return concatenate_datasets(shards)

def compile_dataset(mlm_data, ke_data):
  train_dataset = RoundRobinDataset(mlm_data['train'], ke_data['train'], 'mlm_data', 'ke_data')
  eval_dataset = RoundRobinDataset(mlm_data['test'], ke_data['test'], 'mlm_data', 'ke_data')
  return train_dataset, eval_dataset

def prepare_trainer(training_args, args):
  tokenizer = load_tokenizer(args.model_name_or_path)

  print('| Fetching MLM dataset')
  mlm = fetch_shards(args.mlm_dirs.split(','))
  print('| Fetching KE datasets')
  ke = fetch_shards(args.ke_dirs.split(','))
  print('| Combining datasets in round robin fashion')
  train, eval = compile_dataset(mlm, ke)

  trainer = Trainer(
    model=load_model(args.model_name_or_path),
    args=training_args,
    data_collator=get_data_collator(tokenizer),
    train_dataset=train,
    eval_dataset=eval)
  return trainer

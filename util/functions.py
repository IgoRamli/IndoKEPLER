from datasets import load_from_disk, concatenate_datasets
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers import AutoModel, AutoTokenizer, training_args
from transformers import (
  AutoTokenizer,
  AutoModel,
  Trainer
)
from transformers.utils.dummy_pt_objects import AutoModelForMaskedLM
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

def fetch_shards(dirs):
  shards = []
  for d in dirs:
    print('|   Loading {}'.format(d))
    shards.append(load_from_disk(d))
  print('|   Concatenating shards')
  dataset = concatenate_datasets(shards)
  print('|   Dataset loaded')
  return dataset

def compile_dataset(mlm_data, ke_data):
  train_dataset = RoundRobinDataset(mlm_data, ke_data, 'mlm_data', 'ke_data')
  return train_dataset

def prepare_trainer(training_args, args):
  tokenizer = load_tokenizer(args.model_name_or_path)

  print('| Fetching MLM dataset')
  mlm = fetch_shards(args.mlm_dirs.split(','))
  print('| Fetching KE datasets')
  ke = fetch_shards(args.ke_dirs.split(','))
  print('| Combining datasets in round robin fashion')
  train = compile_dataset(mlm, ke)

  trainer = Trainer(
    model=load_model(args.model_name_or_path),
    args=training_args,
    data_collator=get_data_collator(tokenizer),
    train_dataset=train)
  return trainer
